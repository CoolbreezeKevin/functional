# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:50:48 2021

@author: laconicli
"""

import os
import cv2
 
# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
length = 30
 
# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_movie = cv2.VideoWriter('mytable.mp4', fourcc, length, (360,240))
 
frame_number = 0
 
file_path = r'F:\download_code\FgSegNet\CDnet2014_dataset\mydata\mytable\cmb'
 
for i in range(1,144):
    # Grab a single frame of video
    frame = cv2.imread(os.path.join(file_path, "%06d"%i+".jpg"))
    frame_number += 1
 
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
output_movie.release()
# All done!
cv2.destroyAllWindows()
