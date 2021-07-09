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
import csv

name = "teddy" #TODO
number=3 #1/2/3 2,3 don't work because of distortion
method="FN_wo_pose"


#dest_path = "/cluster_HDD/char/practicum_project_b/results" 

rgbd_path =  "../RGBD" #TODO
fps = 15 #TODO
fps_input_vid = fps

if len(sys.argv) > 1:
    name = str(sys.argv[1])
if len(sys.argv) > 2:
    method = str(sys.argv[2])
if len(sys.argv) > 3:
    number = str(sys.argv[3])

dest_path = "./data/"+method+"/" #TODO
full_name="fr"+str(number)+"_"+name

#sintel_io = importlib.import_module(os.path.join(sintel_depth_path,"sdk/python/sintel_io.py"))
#spec = importlib.util.spec_from_file_location("sintel_io", os.path.join(sintel_depth_path,"sdk/python/sintel_io.py"))
#sintel_io = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(sintel_io)
#sintel_io.cam_read("path")
Path(dest_path).mkdir(parents=True, exist_ok=True)
dest_path = os.path.join(dest_path, full_name)
Path(dest_path).mkdir(parents=True, exist_ok=True)
dest_path = os.path.join(dest_path, "clean")
Path(dest_path).mkdir(parents=True, exist_ok=True)


folder_name="rgbd_dataset_freiburg"+str(number)+"_"+name
src_img_path = os.path.join(rgbd_path,folder_name ,"rgb")
src_path = os.path.join(rgbd_path,folder_name)


def video_from_frames(files): #Adapted from: https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
    frame_array = []
    for i in range(len(files)):
        filename=os.path.join(src_img_path, files[i]+".png")
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(os.path.join(dest_path, "video.mp4"),cv2.VideoWriter_fourcc(*'DIVX'), fps_input_vid, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()   

def parse_frames(path=os.path.join(src_path,"frames_for_cvd.txt")):
    frames=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            frames.append(row[0])
    return frames

frames=parse_frames()
video_from_frames(frames)

    


    
            









