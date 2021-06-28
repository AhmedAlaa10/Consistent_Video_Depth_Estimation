import os
import re
import sys
from posix import listdir
from shutil import copyfile
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.transform import resize
import image_io
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


name="shaman_3"

if len(sys.argv) > 1:
    name = str(sys.argv[1])


batch_size=1 #TODO
#file_path="/home/umbra/Documents/MPI-Sintel-depth-training-20150305/training/depth/bamboo_2/frame_0001.dpt"
src_path="./data/FN/"+name+"/clean/"
depth_path=os.path.join(src_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(batch_size)+"_Oadam/exact_depth/")
initial_path=os.path.join(src_path,"depth_mc/exact_depth/")
truth_path="../MPI-Sintel-depth-training-20150305/training/depth/"+name+"/"

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

#depth = depth_read(depth_path)
#print((depth).shape)
#truth = depth_read(truth_path)
#print((truth).shape)
#shape=tuple((depth).shape)
#shape=(436, 1024)
#depth = resize(depth, truth.shape)
#print((depth).shape)

#print(np.nanmin(depth))
#print(np.nanmean(depth))
#print(np.nanmax(depth))
#print(np.nanmin(truth))
#print(np.nanmean(truth))
#print(np.nanmax(truth))

#distance = (truth - depth)
#distance.flatten()
depth_truth_dir= os.path.join(depth_path,"depth_truth")
depth_truth_fmt = os.path.join(depth_truth_dir, "frame_{:06d}")
depth_error_dir= os.path.join(depth_path,"depth_error")
depth_error_fmt = os.path.join(depth_error_dir, "frame_{:06d}")



#Calculate statistical scale factor:
scale_factor=[]
scale_factor_initial=[]
files = os.listdir(truth_path)
files.sort()
for file in files: #["frame_0001.dpt"]:
    truth = depth_read(os.path.join(truth_path, file))
    depth = depth_read(os.path.join(depth_path, file))
    #depth = resize(depth, truth.shape)
    depth_initial = depth_read(os.path.join(initial_path, file))
    #depth_initial = resize(depth_initial, truth.shape)
    truth[truth == 100000000000.0] = np.nan
    truth=resize(truth, depth.shape)

    scale_factor.append(np.nanmean(truth)/np.nanmean(depth)) 
    scale_factor_initial.append(np.nanmean(truth)/np.nanmean(depth_initial)) 
    print("ScaleFactor for "+file+": "+str(np.nanmean(truth)/np.nanmean(depth)))
scale_factor= np.nanmean(scale_factor)
scale_factor_initial= np.nanmean(scale_factor_initial)



#Compute distance:
ml1 =[]
mse =[]
ml1_norm=[]
mse_norm=[]
ml1_initial =[]
mse_initial =[]
ml1_norm_initial=[]
mse_norm_initial=[]

for file in files: #["frame_0001.dpt"]:
    truth = depth_read(os.path.join(truth_path, file))
    depth = depth_read(os.path.join(depth_path, file))
    #depth = resize(depth, truth.shape)
    depth_initial = depth_read(os.path.join(initial_path, file))
    #depth_initial = resize(depth_initial, truth.shape)
    truth[truth == 100000000000.0] = np.nan
    truth=resize(truth, depth.shape)



    
    depth_norm = depth * scale_factor 
    depth_norm_initial = depth_initial * scale_factor_initial

    distance = (truth - depth)
    distance_norm = (truth - depth_norm)
    ml1.append((np.abs(distance)).mean(axis=None))
    mse.append((np.square(distance)).mean(axis=None))
    ml1_norm.append((np.abs(distance_norm)).mean(axis=None))
    mse_norm.append((np.square(distance_norm)).mean(axis=None))

    distance_initial = (truth - depth_initial)
    distance_norm_initial = (truth - depth_norm_initial)
    ml1_initial.append((np.abs(distance_initial)).mean(axis=None))
    mse_initial.append((np.square(distance_initial)).mean(axis=None))
    ml1_norm_initial.append((np.abs(distance_norm_initial)).mean(axis=None))
    mse_norm_initial.append((np.square(distance_norm_initial)).mean(axis=None))
    print("\n"+file+":")
    print("min(t,d,n,i):")
    print(np.nanmin(truth))
    print(np.nanmin(depth))
    print(np.nanmin(depth_norm))
    print(np.nanmin(depth_initial))

    print("max(t,d,n,i):")
    print(np.nanmax(truth))
    print(np.nanmax(depth))
    print(np.nanmax(depth_norm))
    print(np.nanmax(depth_initial))

    print("Avg(t,d,n,i):")
    print(np.nanmean(truth))
    print(np.nanmean(depth))
    print(np.nanmean(depth_norm))
    print(np.nanmean(depth_initial))
#print(mse)
print("ScaleFactor: "+str(scale_factor))
print("Results (Fine-Tuned):")
aml1=np.nanmean(ml1)
amse=np.nanmean(mse)
print("Ml1: "+str(aml1))
print("MSE: "+str(amse))
aml1_norm=np.nanmean(ml1_norm)
amse_norm=np.nanmean(mse_norm)
print("Ml1(norm): "+str(aml1_norm))
print("MSE(norm): "+str(amse_norm))

print("\nResults (Initial):")
aml1_initial=np.nanmean(ml1_initial)
amse_initial=np.nanmean(mse_initial)
print("Ml1: "+str(aml1_initial))
print("MSE: "+str(amse_initial))
aml1_norm_initial=np.nanmean(ml1_norm_initial)
amse_norm_initial=np.nanmean(mse_norm_initial)
print("Ml1(norm): "+str(aml1_norm_initial))
print("MSE(norm): "+str(amse_norm_initial))