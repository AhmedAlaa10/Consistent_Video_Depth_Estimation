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
import os
from os.path import isfile, join
from PIL import Image
import csv

name = "teddy" #TODO
number=3 #1/2/3 2,3 don't work because of distortion
method="FN"
num_frames=50
#local:
rgbd_path =  "../RGBD" #TODO


#server:
#dest_path = "/cluster_HDD/char/practicum_project_b/results" 
#sintel_complete_path = "/cluster_HDD/char/practicum_project_b/MPI-Sintel-complete"
#sintel_depth_path = "/cluster_HDD/char/practicum_project_b/MPI-Sintel-depth"

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



Path(dest_path).mkdir(parents=True, exist_ok=True)
dest_path = os.path.join(dest_path, full_name)
Path(dest_path).mkdir(parents=True, exist_ok=True)
dest_path = os.path.join(dest_path, "clean")
Path(dest_path).mkdir(parents=True, exist_ok=True)

img_path = os.path.join(dest_path, "color_full")
cam_path = os.path.join(dest_path, "colmap_dense")
Path(img_path).mkdir(parents=True, exist_ok=True)
Path(cam_path).mkdir(parents=True, exist_ok=True)
cam_path = os.path.join(cam_path, "pose_init")
Path(cam_path).mkdir(parents=True, exist_ok=True)

folder_name="rgbd_dataset_freiburg"+str(number)+"_"+name
src_img_path = os.path.join(rgbd_path,folder_name ,"rgb")
src_path = os.path.join(rgbd_path,folder_name)

def video_from_frames(): #Adapted from: https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
    pathIn= img_path
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

def parse_frames(path=os.path.join(src_path,"frames_for_cvd.txt")):
    frames=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            frames.append(row[0])
    if len(frames)!=num_frames:
        print("WARNING: number_of_frames doesn't match number of frames in:"+path)
    return frames

#Reads extrinsics from RGBD dataset and returns array of [tx ty tz qx qy qz qw] for index-corresponding name in frames:
def parse_extrinsics(frames, path=os.path.join(src_path,"groundtruth.txt")):
    extrinsics=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        i=0
        for row in csv_reader:
            if i>=num_frames:
                break
            if row[0]!="#" and float(row[0])>float(frames[i])+0.1:
                print("WARNING!: Camera extrinsics later than frame by "+str(float(row[0])-float(frames[i]))+" sec at index "+str(i)+"!")
            #print(row[0][:-2])
            #print(frames[i][:-4])
            if row[0]!="#" and float(row[0]) >= float(frames[i]):
                print(frames[i]+":"+row[0])
                extrinsics.append([row[1],row[2],row[3],row[4],row[5],row[6],row[7]])
                i+=1
            
    if len(extrinsics)!=num_frames:
        print("WARNING: number_of_frames doesn't match number of frames in:"+path+"("+str(len(extrinsics))+")")
    #print(extrinsics)
    return extrinsics
#os.chdir(path)
#print(os.listdir("."))
width, height = (None,None)
# Copy and rename images:
frames = parse_frames()
if len(listdir(img_path))==0:    
    for i, file in enumerate(frames):
        index =str(i).zfill(6)
        file_new = str("frame_"+index+".png") 
        #print(file_new)
        #copyfile(os.path.join(src_img_path, file), os.path.join(img_path, file_new))
        im = Image.open(os.path.join(src_img_path, file+".png"))
        im.save(os.path.join(img_path, file_new), "PNG")
        if width is None:
            width, height = im.size




# Copy and transform cam data:
if len(listdir(cam_path))==0:
    cameras_file = open(os.path.join(cam_path, "cameras.txt"),"w")
    images_file = open(os.path.join(cam_path, "images.txt"),"w")
    points3D_file = open(os.path.join(cam_path, "points3D.txt"),"w")
    points3D_file.close()
    frames_file = open(os.path.join(dest_path, "frames.txt"),"w")

    #TODO: use ROS default instead? https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    #Save camera intrinsics
    if number==1: 
        print("WARNING: Invalid results, because colmap can't model distortion! Use Freiburg3 only")
        line = str(1)+" "+"PINHOLE"+" "+str(width)+" "+str(height)+" "+str(517.3)+" "+str(516.5)+" "+str(318.6)+" "+str(255.3)+"\n" 
    elif number==2:
        print("WARNING: Invalid results, because colmap can't model distortion! Use Freiburg3 only")
        line = str(1)+" "+"PINHOLE"+" "+str(width)+" "+str(height)+" "+str(520.9)+" "+str(521.0)+" "+str(325.1)+" "+str(249.7)+"\n"
    elif number==3: 
        line = str(1)+" "+"PINHOLE"+" "+str(width)+" "+str(height)+" "+str(535.4)+" "+str(539.2)+" "+str(320.1)+" "+str(247.6)+"\n"
    else:
        print("Only number=3 allowed")
    
    cameras_file.write(line)
    cameras_file.close()
    print("OK")
    frames_ex = parse_extrinsics(frames)      
    for i , frame_ex in enumerate(frames_ex):
         
        cam_id=1       

        index =str(i).zfill(6)
        file_new = str("frame_"+index+".png") 
        print(file_new)

        line = str(i+1)+" "+str(frame_ex[3])+" "+str(frame_ex[4])+" "+str(frame_ex[5])+" "+str(frame_ex[6])+" "+str(frame_ex[0])+" "+str(frame_ex[1])+" "+str(frame_ex[2])+" "+str(cam_id)+" "+file_new+"\n" +"\n" #TODO: remove one  \n?
        images_file.write(line)

    images_file.close()        

    line = str(num_frames)+"\n"
    frames_file.write(line)
    line = str(width)+"\n"
    frames_file.write(line)
    line = str(height)+"\n"
    frames_file.write(line)

    step_size=(float(num_frames)/float(fps))/float(num_frames)
    time= 0.
    for i in range(num_frames):
        line = str(time)+"\n"        
        frames_file.write(line)
        time+=step_size
        
    frames_file.close()

    #video_from_frames()

    


    
            









