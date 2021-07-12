import os
from pickle import FALSE
import re
import sys
from posix import listdir
from shutil import copyfile
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.transform import resize
import utils.image_io
import copy
import csv
import open3d as o3d #pip install open3d
import open3d.visualization.rendering as rendering
import cv2 #pip install cv2
from scipy.spatial.transform import Rotation as R
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

name="alley_2"
batch_size=[1,2,3,4]
type= "FN"  #FN / GMA / custom / ...
preview=False
preview_mask=True
preview_cvd=True
preview_result=True

if not preview:
    preview_mask=False
    preview_cvd=False
    preview_result=False

if len(sys.argv) > 1:
    name = str(sys.argv[1])

output_dir= os.path.join("./data/CVD_DP",name)
os.makedirs(output_dir, exist_ok=True)
final_depth_dir = os.path.join(output_dir, "exact_depth")
os.makedirs(final_depth_dir, exist_ok=True)

Data_dir_mask = './data/human_depth/'+name+'_mask' #TODO: adjust mannually
Data_dir_depth = './data/human_depth/'+name+'_depth'

src_path="./data/"+type+"/"+name+"/clean/"
scales_path=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")
for bs in batch_size: 
        depth_path=os.path.join(src_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(bs)+"_Oadam/exact_depth/")
        if os.path.isfile(depth_path+"/frame_0001.dpt"):
            break

def depth_write(filename, depth): #Copied from sintel_io.py from http://sintel.is.tue.mpg.de/depth
    """ Write depth to file. """
    height,width = depth.shape[:2]
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR.encode('utf-8'))
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
        
    depth.astype(np.float32).tofile(f)
    f.close()

def parse_scales(path):
    global use_scales
    scales=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i=0
        for row in csv_reader:
            while i!=int(float(row[0])):
                scales.append(float(1.))
                i+=1
            
            scales.append(float(row[1]))            
            i+=1
    if len(scales)==0:
        print("WARNING no/invalid file at "+path)
        print("SCALES DISABLED!")
        use_scales=False
    #print(scales)
    return scales

def depth_read(filename): #Copied from sintel_io.py from http://sintel.is.tue.mpg.de/depth
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

scales=parse_scales(scales_path)

depth_frames=os.listdir(depth_path)
depth_frames.sort()
for i, frame in enumerate(depth_frames):
    depth_cvd = depth_read(os.path.join(depth_path, frame))
    depth_cvd*=scales[i]
    #print(depth_cvd.shape)

    path_mask = Data_dir_mask + '/' + 'frame_' +'%04d' %(i+1) + '_mask.png'
    path_depth = Data_dir_depth + '/' + 'frame_' +'%04d' %(i+1) +'_depth_resize.txt'
    if os.path.isfile(path_mask) and os.path.isfile(path_depth):
        mask=cv2.imread(path_mask,-1)                       
        mask = cv2.resize(mask, (depth_cvd.shape[1], depth_cvd.shape[0]), interpolation=cv2.INTER_AREA)
        depth_human=np.loadtxt(path_depth)
        depth_human = cv2.resize(depth_human, (depth_cvd.shape[1], depth_cvd.shape[0]),interpolation=cv2.INTER_AREA)                      
        #print(mask.shape)                   
        [rows, columns] = depth_cvd.shape
        depth = depth_cvd.copy()
        for row in range(rows):
            for column in range(columns):
                #print(mask_rgb_np[row,column,0])
                if(mask[row,column]>0):
                    depth[row,column] = depth_human[row,column]
    else:
        print("No Densepose data available for"+frame)
        depth=depth_cvd
    if preview_cvd:
        #depth_cv = cv2.cvtColor(np.array(depth), cv2.COLOR_RGBA2BGRA)
        #print(depth)
        depth_vis=depth_cvd.copy()
        depth_vis+=1
        depth_vis.squeeze()
        inv_depth =  np.divide(1., depth_vis, out=np.zeros_like(depth_vis), where=depth_vis!=0)
        #depth_cv = cv2.cvtColor(np.array(inv_depth), cv2.COLOR_RGBA2BGRA)
        cv2.imshow("Preview window", inv_depth)
        cv2.waitKey()

    if preview_mask:
        if os.path.isfile(path_mask) and os.path.isfile(path_depth):
            #depth_cv = cv2.cvtColor(np.array(depth), cv2.COLOR_RGBA2BGRA)
            #print(depth)
            depth_vis=mask.copy()
            #depth_cv = cv2.cvtColor(np.array(inv_depth), cv2.COLOR_RGBA2BGRA)
            cv2.imshow("Preview window", depth_vis)
            cv2.waitKey()

    if preview_result:
        if os.path.isfile(path_mask) and os.path.isfile(path_depth):
            #depth_cv = cv2.cvtColor(np.array(depth), cv2.COLOR_RGBA2BGRA)
            #print(depth)
            depth_vis=depth.copy()
            depth_vis+=1
            depth_vis.squeeze()
            inv_depth =  np.divide(1., depth_vis, out=np.zeros_like(depth_vis), where=depth_vis!=0)
            #depth_cv = cv2.cvtColor(np.array(inv_depth), cv2.COLOR_RGBA2BGRA)
            cv2.imshow("Preview window", inv_depth)
            cv2.waitKey()

    final_depth_path = os.path.join(final_depth_dir, frame)
    depth_write(final_depth_path,depth)
    print(frame+" done!")

