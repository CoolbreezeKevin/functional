<details open>
<summery>Idea Description</summery>
Attempting segmentation networks to tackle foreground segmentation problems cannot get a beautiful result, because they are sensitive to patterns. 

My Idea originates from a common and useful structure in many instance segmentation networks and target detection networks, saying "region proposal". 

Detection network trained on datasets, like coco2017, can accurately get the location and size of items in images. However, what will happen when provied classes are not included in the training set?

After some experiments, I guess network itself is sensitive to almost all kinds of items. The whole process may like geting a box, predicting a confidence for each certain class, and then giving an output for users. 

We assume that our foreground item in provided pictures is in the center of whole image and always the largest compared with other. Accordingly, I low down the confidence threshold to get more boxes. Relying on the bounding boxes provided by those networks, I define a scoring function, which is base on our prior experience, to select only one rectangle for postprocessing. Then the box is used to generate new boxes in the surroundings. Next, the boxes would be thrown into function cv2.grabcut() respectively and get the same number of masks.  Finally, a simple voting system is used to get the final answer. 
</details>

<details open>
<summery>Install</summery>
Visit https://github.com/ultralytics/yolov5 for more installation details. 
Note: I have already tried to install different versions of torch and torchvision. Althoughthe official said PtTorch>=3.7 and Python>=3.6.0 is required, last version python and pytorch somehow will suffer from different problems. 
So I propose my environments in the package. 
Runing the network itself is not the most time consuming progress, so using cpu or gpu will not affect the speed of whole process. 
```bash
conda create -n yolo python=3.7
conda activate yolo
git clone git.woa.com/laconicli/foregroundsegmentation_with_yolov5
cd yolov5
pip install -r requirements.txt
```
</details>

<details open>
<summery>Run Code</summery>
``` python
python testbox.py
```
For defferent performance needs, there are some parameters that you can adjust. 
Please search followings in a python IDE. 
--source # your targeted image set folder. 
--conf_thres # the lowest confidence for geting the box from networks
--Voter_Num # a smaller number for higher recall, optional 1, 2, 3. 
--RATE # larger RATE will get smaller generated boxes. 
</details>
