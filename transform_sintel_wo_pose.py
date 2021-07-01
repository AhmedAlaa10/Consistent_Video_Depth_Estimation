#!/usr/bin/env python3
import os
import re
import sys
from posix import listdir
from shutil import copyfile
from pathlib import Path
import importlib.util
#from pyquaternion import Quaternion
import numpy as np
from math import sqrt
import cv2 #TODO: pip install opencv-python
import numpy as np
import os
from os.path import isfile, join
from PIL import Image

name = "alley_1" #TODO
img_type = "clean" #TODO "clean"/ "final" / "albedo"
data_type = "training" #TODO "training"/ "test" 
method="FN_wo_pose"


#dest_path = "/cluster_HDD/char/practicum_project_b/results" 

sintel_complete_path =  "../MPI-Sintel-complete" #TODO
#sintel_complete_path = "/cluster_HDD/char/practicum_project_b/MPI-Sintel-complete"
sintel_depth_path =  "../MPI-Sintel-depth-training-20150305" #TODO
#sintel_depth_path = "/cluster_HDD/char/practicum_project_b/MPI-Sintel-depth"
fps = 15 #TODO
fps_input_vid = fps

if len(sys.argv) > 1:
    name = str(sys.argv[1])
if len(sys.argv) > 2:
    method = str(sys.argv[2])

dest_path = "./data/"+method+"/" #TODO

#sintel_io = importlib.import_module(os.path.join(sintel_depth_path,"sdk/python/sintel_io.py"))
#spec = importlib.util.spec_from_file_location("sintel_io", os.path.join(sintel_depth_path,"sdk/python/sintel_io.py"))
#sintel_io = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(sintel_io)
#sintel_io.cam_read("path")

Path(dest_path).mkdir(parents=True, exist_ok=True)
dest_path = os.path.join(dest_path, name)
Path(dest_path).mkdir(parents=True, exist_ok=True)
dest_path = os.path.join(dest_path, img_type)
Path(dest_path).mkdir(parents=True, exist_ok=True)


src_img_path = os.path.join(sintel_complete_path, data_type, img_type, name)


def video_from_frames(): #Adapted from: https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
    pathIn= src_img_path
    pathOut = os.path.join(dest_path, "video.mp4")
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    files.sort()
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    for i in range(len(files)):
        filename=os.path.join(pathIn, files[i])
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps_input_vid, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()   


video_from_frames()

    


    
            









