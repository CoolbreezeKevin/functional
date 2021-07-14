import numpy as np
import tensorflow as tf
import random as rn
import os, sys
import cv2
# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  For reprodocable results
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras, glob
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from sklearn.utils import compute_class_weight
from FgSegNet_M_S_module import FgSegNet_M_S_module
from keras.utils.data_utils import get_file

# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')

# =============================================================================
# Few frames, load into memory directly
# =============================================================================
def generateData(dataset_dir, scene, method_name):
    assert method_name in ['FgSegNet_M', 'FgSegNet_S'], 'method_name is incorrect'
    
    void_label = -1.
    
    # Given ground-truths, load training frames
    # ground-truths end with '*.png'
    # training frames end with '*.jpg'
    
    # given ground-truths, load inputs  
    # Y_list = glob.glob(os.path.join(train_dir, '*.png'))
    X_list= glob.glob(os.path.join(dataset_dir, 'input','*.jpg'))


    if len(X_list)<=0:
        raise ValueError('System cannot find the dataset path or ground-truth path. Please give the correct path.')   
        
    # X must be corresponded to Y
    X_list = sorted(X_list)

    # load training data
    X = []

    for i in range(len(X_list)):
        x = kImage.load_img(X_list[i])
        x = kImage.img_to_array(x)
        X.append(x)

    X = np.asarray(X)
    
    if method_name=='FgSegNet_M':
        # Image Pyramid
        scale2 = []
        scale3 = []
        for i in range(0, X.shape[0]):
           pyramid = tuple(pyramid_gaussian(X[i]/255., max_layer=2, downscale=2))
           scale2.append(pyramid[1]*255.) # 2nd scale
           scale3.append(pyramid[2]*255.) # 3rd scale
           del pyramid
           
        scale2 = np.asarray(scale2)
        scale3 = np.asarray(scale3)
    
    if method_name=='FgSegNet_M':
        return [X, scale2, scale3,]
    else:
        return [X,]
    
def pred(results, scene, mdl_path, vgg_weights_path, method_name, head):
    assert method_name in ['FgSegNet_M', 'FgSegNet_S'], 'method_name is incorrect'
    
    img_shape = results[0][0].shape # (height, width, channel)
    model = FgSegNet_M_S_module(lr, reg, img_shape, scene, vgg_weights_path)
    
    if method_name=='FgSegNet_M':
        model = model.initModel_M('CDnet')
    else:
        model = model.initModel_S('CDnet')
    model.load_weights(r"F:\download_code\FgSegNet_S\CDnet\models50\baseline\mdl_office.h5")
    # # make sure that training input shape equals to model output
    # input_shape = (img_shape[0], img_shape[1])
    # output_shape = (model.output._keras_shape[1], model.output._keras_shape[2])
    # assert input_shape==output_shape, 'Given input shape:' + str(input_shape) + ', but your model outputs shape:' + str(output_shape) 

    chk = keras.callbacks.ModelCheckpoint(mdl_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=num_patience, verbose=1, mode='auto')
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')


    # Predict and save
    from PIL import Image
    
    id=1
    if method_name=='FgSegNet_M':
        y=model.predict([results[0], results[1], results[2]])
    else:
        # maybe we can use early stopping instead for FgSegNet_S, and also set max epochs to 100 or 110 +50
        a, b, c, d = results[0].shape
        yy=np.zeros((b,c,3))
        data=results[0]
        for n in range(a):
            x=data[n]
            x=np.expand_dims(x, axis=0)
            y=model.predict(x)
            y=y[0]>=0.3
            y=np.squeeze(y, axis=2)
            y=y*255
            yy[:,:,0]=y
            path = os.path.join(head, "in%06d" % id + ".jpg",)
            Image.fromarray(yy.astype('uint8')).save(path)
            id+=1
    del model, results, chk, redu, early

def post_process(path):
    print('start post process...')
    id=1
    org = glob.glob(os.path.join(path, 'input','*.jpg'))
    prd = glob.glob(os.path.join(dataset_dir, 'pred','*.jpg'))
    for p1, p2 in zip(org, prd):
        org=cv2.imread(p1)
        prd=cv2.imread(p2)
        combine=cv2.addWeighted(org, 0.8, prd, 0.2, 0)
        cv2.imwrite(os.path.join(path,"cmb","%06d"%id + '.jpg'), combine)
        id+=1

# =============================================================================
# Main func
# =============================================================================
if __name__=="__main__":
    dataset = {
                'mydata':['mytable',],
    }

    # =============================================================================

    method_name = 'FgSegNet_S' # either <FgSegNet_M> or <FgSegNet_S>, default FgSegNet_M

    num_frames = 50 # either 50 or 200 frames, default 50 frames

    reduce_factor = 0.1
    num_patience = 6
    lr = 1e-4
    reg=5e-4
    max_epochs = 60 if num_frames==50 else 50 # 50f->60epochs, 200f->50epochs
    val_split = 0.2
    batch_size = 1
    # =============================================================================

    # Example: (free to modify)

    # FgSegNet/FgSegNet/FgSegNet_M_S_CDnet.py
    # FgSegNet/FgSegNet/FgSegNet_M_S_SBI.py
    # FgSegNet/FgSegNet/FgSegNet_M_S_UCSD.py
    # FgSegNet/FgSegNet/FgSegNet_M_S_module.py

    # FgSegNet/FgSegNet_dataset2014/...
    # FgSegNet/CDnet2014_dataset/...


    assert num_frames in [50,200], 'Incorrect number of frames'
    main_dir = os.path.join('..', method_name)
    main_mdl_dir = os.path.join(main_dir, 'CDnet', 'models' + str(num_frames))
    vgg_weights_path = r'F:\download_code\FgSegNet\FgSegNet\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not os.path.exists(vgg_weights_path):
        # keras func
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                    file_hash='6d6bbae143d832006294945121d1f1fc')

    print('*** Current method >>> ' + method_name + '\n')
    # for category, scene_list in dataset.items():
    #     print(category, scene_list)
    #     mdl_dir = os.path.join(main_mdl_dir, category)
    #     if not os.path.exists(mdl_dir):
    #         os.makedirs(mdl_dir)
            
    #     for scene in scene_list:
    #         print ('Training ->>> ' + category + ' / ' + scene)
            
    #         # training frame path and dataset2014 path
    #         train_dir = os.path.join('.', 'FgSegNet_dataset2014', category, scene + str(num_frames))
    #         dataset_dir = os.path.join('.', 'CDnet2014_dataset', category, scene)
    #         results = generateData(train_dir, dataset_dir, scene, method_name)
            
    #         mdl_path = os.path.join(mdl_dir, 'mdl_' + scene + '.h5')
    #         train(results, scene, mdl_path, vgg_weights_path, method_name)
    #         del results


    category, scene_list = 'mydata', ['mytable']

    mdl_dir = os.path.join(main_mdl_dir, category)
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)
        
    for scene in scene_list:
        print ('Predicting ->>> ' + category + ' / ' + scene)
        
        # training frame path and dataset2014 path
        # train_dir = os.path.join('.', 'FgSegNet_dataset2014', category, scene + str(num_frames))
        dataset_dir = os.path.join('.', 'CDnet2014_dataset', category, scene)
        results = generateData(dataset_dir, scene, method_name)
        
        mdl_path = os.path.join(mdl_dir, 'mdl_' + scene + '.h5')
        pred(results, scene, mdl_path, vgg_weights_path, method_name, head = r'F:\download_code\FgSegNet\CDnet2014_dataset\mydata\mytable\pred')
        del results

    post_process(r'F:\download_code\FgSegNet\CDnet2014_dataset\mydata\mytable')

