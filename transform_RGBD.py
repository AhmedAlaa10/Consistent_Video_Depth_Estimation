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

name = "xyz" #TODO
number=1 #1/2/3
method="FN"

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
    number = str(sys.argv[2])
if len(sys.argv) > 3:
    method = str(sys.argv[3])


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

folder_name="rgbd_datset_freiburg"+str(number)+"_"+name
src_img_path = os.path.join(rgbd_path,folder_name ,"rgb")
src_cam_path = os.path.join(rgbd_path,folder_name ,"groundtruth.txt")

def cam_read(filename):  #Copied from sintel_io.py from http://sintel.is.tue.mpg.de/depth
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def trace_method(matrix): #Copied from https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
            """
            This code uses a modification of the algorithm described in:
            https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
            which is itself based on the method described here:
            http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
            Altered to work with the column vector convention instead of row vectors
            """
            m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
            if m[2, 2] < 0:
                if m[0, 0] > m[1, 1]:
                    t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                    q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
                else:
                    t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                    q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
            else:
                if m[0, 0] < -m[1, 1]:
                    t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                    q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
                else:
                    t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                    q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

            q = np.array(q).astype('float64')
            q *= 0.5 / sqrt(t)
            return q

def quaternion_from_matrix(matrix):
    return trace_method(matrix)

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
    
#os.chdir(path)
#print(os.listdir("."))
width, height = (None,None)
# Copy and rename images:
if len(listdir(img_path))==0:
    files = listdir(src_img_path)
    for file in files:
        split1 = file.split("_")
        split2 = split1[1].split(".")
        index = int(split2[0])-1
        index =str(index).zfill(6)
        file_new = str(split1[0]+"_"+index+".png") 
        #print(file_new)
        #copyfile(os.path.join(src_img_path, file), os.path.join(img_path, file_new))
        im = Image.open(os.path.join(src_img_path, file))
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

    frame_cams = listdir(src_cam_path)
    frame_cams.sort()
    cams =[]
    for i , frame_cam in enumerate(frame_cams):
        I,E = cam_read(os.path.join(src_cam_path, frame_cam))
        print(frame_cam)

        #if I not in cams:
        #    cams.append(I)
        #cam_id = cams.index(I)+1
        cam_id=-1
        new = True
        for j, cam in enumerate(cams):
            if (I==cam).all():
                new =False
                cam_id =j +1
                break
        if new:
            cam_id = len(cams) +1
            cams.append(I)
        


        split1 = frame_cam.split("_")
        split2 = split1[1].split(".")
        index = int(split2[0])-1
        index =str(index).zfill(6)
        frame_name = str(split1[0]+"_"+index+".png") 

        R = np.delete(E, np.s_[3], axis=1)
        q = quaternion_from_matrix(matrix=R)
        line = str(i+1)+" "+str(q[0])+" "+str(q[1])+" "+str(q[2])+" "+str(q[3])+" "+str(E[0,3])+" "+str(E[1,3])+" "+str(E[2,3])+" "+str(cam_id)+" "+frame_name+"\n" +"\n" #TODO: remove one  \n?
        images_file.write(line)

    images_file.close()

    for i , I in enumerate(cams):
        line = str(i+1)+" "+"PINHOLE"+" "+str(width)+" "+str(height)+" "+str(I[0,0])+" "+str(I[1,1])+" "+str(I[0,2])+" "+str(I[1,2])+"\n" 
        cameras_file.write(line)

    cameras_file.close()

    number_of_frames=len(frame_cams)
    line = str(number_of_frames)+"\n"
    frames_file.write(line)
    line = str(width)+"\n"
    frames_file.write(line)
    line = str(height)+"\n"
    frames_file.write(line)

    step_size=(float(number_of_frames)/float(fps))/float(number_of_frames)
    time= 0.
    for i in range(number_of_frames):
        line = str(time)+"\n"        
        frames_file.write(line)
        time+=step_size
        
    frames_file.close()

    #video_from_frames()

    


    
            









