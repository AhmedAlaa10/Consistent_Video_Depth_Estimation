import os
import re
import sys
import csv
from posix import listdir
from shutil import copyfile
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.transform import resize
import utils.image_io
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'




name="shaman_3"
batch_size=[1,2,3,4] #TODO
gma=True
pose=True
dp=True
dp_gma=True
per_frame=True
use_scales=False

if len(sys.argv) > 1:
    name = str(sys.argv[1])

#file_path="/home/umbra/Documents/MPI-Sintel-depth-training-20150305/training/depth/bamboo_2/frame_0001.dpt"
depth_dataset_path="../MPI-Sintel-depth-training-20150305/"
if pose:
    src_path="./data/FN/"+name+"/clean/"
    scales_path=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")
else:
    src_path="./data/FN_wo_pose/"+name+"/clean/"
    scales_path=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")
if gma:
    gma_path="./data/GMA/"+name+"/clean/"
    scales_path_gma=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")
if dp:
    dp_path="./data/FN_DP/"+name+"/clean/"
    scales_path_dp=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")
if dp_gma:
    gma_dp_path="./data/GMA_DP/"+name+"/clean/"
    scales_path_gma_dp=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")


output_path=os.path.join(src_path,"evaluation")
os.makedirs(output_path, exist_ok=True)

#Dont change after here
#----------------------------------------------------------------------------------------------------------------------------------------------------
if gma and not os.path.isdir(gma_path):
    print("GMA depth folder ("+gma_path+") empty")
    gma=False
if dp and not os.path.isdir(dp_path):
    print("DensePose depth folder ("+dp_path+") empty")
    dp=False
if dp_gma and not os.path.isdir(gma_dp_path):
    print("GMA DensePose depth folder ("+gma_dp_path+") empty")
    dp_gma=False

norm_error_vis_path=os.path.join(output_path,"error_visualization_norm")
os.makedirs(norm_error_vis_path, exist_ok=True)

error_vis_path=os.path.join(output_path,"error_visualization")
os.makedirs(error_vis_path, exist_ok=True)

norm_initial_error_vis_path=os.path.join(output_path,"error_visualization_initial_norm")
os.makedirs(norm_initial_error_vis_path, exist_ok=True)

initial_error_vis_path=os.path.join(output_path,"error_visualization_initial")
os.makedirs(initial_error_vis_path, exist_ok=True)

norm_gma_error_vis_path=os.path.join(output_path,"error_visualization_gma_norm")
os.makedirs(norm_gma_error_vis_path, exist_ok=True)

gma_error_vis_path=os.path.join(output_path,"error_visualization_gma")
os.makedirs(gma_error_vis_path, exist_ok=True)

norm_dp_error_vis_path=os.path.join(output_path,"error_visualization_dp_norm")
os.makedirs(norm_dp_error_vis_path, exist_ok=True)

dp_error_vis_path=os.path.join(output_path,"error_visualization_dp")
os.makedirs(dp_error_vis_path, exist_ok=True)

norm_gma_dp_error_vis_path=os.path.join(output_path,"error_visualization_gma_dp_norm")
os.makedirs(norm_gma_dp_error_vis_path, exist_ok=True)

gma_dp_error_vis_path=os.path.join(output_path,"error_visualization_gma_dp")
os.makedirs(gma_dp_error_vis_path, exist_ok=True)




for bs in batch_size: 
    depth_path=os.path.join(src_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(bs)+"_Oadam/exact_depth/")
    if os.path.isfile(depth_path+"/frame_0001.dpt"):
        break
initial_path=os.path.join(src_path,"depth_mc/exact_depth/")
for bs in batch_size:
    depth_gma_path=os.path.join(gma_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(bs)+"_Oadam/exact_depth/")
    if os.path.isfile(depth_gma_path+"/frame_0001.dpt"):
        break
for bs in batch_size:
    depth_dp_path=os.path.join(dp_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(bs)+"_Oadam/exact_depth/")
    if os.path.isfile(depth_dp_path+"/frame_0001.dpt"):
        break
for bs in batch_size:
    depth_gma_dp_path=os.path.join(gma_dp_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(bs)+"_Oadam/exact_depth/")
    if os.path.isfile(depth_gma_dp_path+"/frame_0001.dpt"):
        break
truth_path=os.path.join(depth_dataset_path,"training/depth/"+name+"/")


#Path for colored truth depth visualization:
truth_viz_path=os.path.join(depth_dataset_path,"training/dept_viz_col/") 
os.makedirs(truth_viz_path, exist_ok=True)
truth_viz_path=os.path.join(truth_viz_path,name)
os.makedirs(truth_viz_path, exist_ok=True)

def parse_scales(path):
    global use_scales
    scales=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            scales.append(float(row[1]))
    if len(scales)!=50:
        print("WARNING no/invalid file at "+path)
        print("SCALES DISABLED!")
        use_scales=False

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

def save_image(file_name, image): # Copied from utils/image_io.py from https://github.com/facebookresearch/consistent_depth
    
    image = 255.0 * image
    image = Image.fromarray(image.astype("uint8"))
    image.save(file_name)

#depth = depth_read(depth_path)
#print((depth).shape)
#truth = depth_read(truth_path)
#print((truth).shape)
#shape=tuple((depth).shape)
#shape=(436, 1024)
#depth = resize(depth, truth.shape)
#print((depth).shape)

#print(np.min(depth))
#print(np.nanmean(depth))
#print(np.max(depth))
#print(np.min(truth))
#print(np.nanmean(truth))
#print(np.max(truth))

#distance = (truth - depth)
#distance.flatten()
depth_truth_dir= os.path.join(depth_path,"depth_truth")
depth_truth_fmt = os.path.join(depth_truth_dir, "frame_{:06d}")
depth_error_dir= os.path.join(depth_path,"depth_error")
depth_error_fmt = os.path.join(depth_error_dir, "frame_{:06d}")

#Read scales:
scales=parse_scales(scales_path)
if gma:
    scales_gma=parse_scales(scales_path_gma)
if dp:
    scales_dp=parse_scales(scales_path_dp)
if dp_gma:
    scales_gma_dp=parse_scales(scales_path_gma_dp)


#Calculate statistical scale factor:
scale_factor=[]
scale_factor_initial=[]
scale_factor_gma=[]
scale_factor_dp=[]
scale_factor_gma_dp=[]
files = os.listdir(truth_path)
files.sort()
for i, file in enumerate(files): #["frame_0001.dpt"]:
    truth = depth_read(os.path.join(truth_path, file))
    depth = depth_read(os.path.join(depth_path, file))
    if use_scales:
        depth*=scales[i]
    truth[truth == 100000000000.0] = np.nan
    truth = resize(truth, depth.shape)
    if gma:
        depth_gma = depth_read(os.path.join(depth_gma_path, file))
        if use_scales:
            depth_gma*=scales_gma[i]
    if dp:
        depth_dp = depth_read(os.path.join(depth_dp_path, file))
        if use_scales:
            depth_dp*=scales_dp[i]
    if dp_gma:
        depth_gma_dp = depth_read(os.path.join(depth_gma_dp_path, file))
        if use_scales:
            depth_gma_dp*=scales_gma_dp[i]

    #depth = resize(depth, truth.shape)
    depth_initial = depth_read(os.path.join(initial_path, file))
    if use_scales:
        depth_initial*=scales[i]


    #depth_initial = resize(depth_initial, truth.shape)
    

    scale_factor.append(np.nanmean(truth)/np.nanmean(depth)) 
    scale_factor_initial.append(np.nanmean(truth)/np.nanmean(depth_initial)) 
    if gma:
        scale_factor_gma.append(np.nanmean(truth)/np.nanmean(depth_gma))
    if dp:
        scale_factor_dp.append(np.nanmean(truth)/np.nanmean(depth_dp))
    if dp_gma:
        scale_factor_gma_dp.append(np.nanmean(truth)/np.nanmean(depth_gma_dp))
scale_factor= np.nanmean(scale_factor)
scale_factor_initial= np.nanmean(scale_factor_initial)
if gma:
    scale_factor_gma= np.nanmean(scale_factor_gma)
if dp:
    scale_factor_dp= np.nanmean(scale_factor_dp)
if dp_gma:
    scale_factor_gma_dp= np.nanmean(scale_factor_gma_dp)



#Compute distance:
ml1 =[]
mse =[]
ml1_norm=[]
mse_norm=[]
ml1_initial =[]
mse_initial =[]
ml1_norm_initial=[]
mse_norm_initial=[]
ml1_gma =[]
mse_gma =[]
ml1_norm_gma=[]
mse_norm_gma=[]
ml1_dp =[]
mse_dp =[]
ml1_norm_dp=[]
mse_norm_dp=[]
ml1_gma_dp =[]
mse_gma_dp =[]
ml1_norm_gma_dp=[]
mse_norm_gma_dp=[]
for i, file in enumerate(files): #["frame_0001.dpt"]:
    truth = depth_read(os.path.join(truth_path, file))
    depth = depth_read(os.path.join(depth_path, file))
    if use_scales:
            depth*=scales[i]
    #depth = resize(depth, truth.shape)
    depth_initial = depth_read(os.path.join(initial_path, file))
    if use_scales:
            depth_initial*=scales[i]
    truth[truth == 100000000000.0] = np.nan
    truth = resize(truth, depth.shape)
    

    #depth_initial = resize(depth_initial, truth.shape)


    if gma:
        depth_gma = depth_read(os.path.join(depth_gma_path, file))
        if use_scales:
            depth_gma*=scales_gma[i]
    if dp:
        depth_dp = depth_read(os.path.join(depth_dp_path, file))
        if use_scales:
            depth_dp*=scales_dp[i]
    if dp_gma:
        depth_gma_dp = depth_read(os.path.join(depth_gma_dp_path, file))
        if use_scales:
            depth_gma_dp*=scales_gma_dp[i]



    
    depth_norm = depth * scale_factor 
    depth_norm_initial = depth_initial * scale_factor_initial

    if gma:
        depth_norm_gma = depth_gma * scale_factor_gma
    if dp:
        depth_norm_dp = depth_dp * scale_factor_dp
    if dp_gma:
        depth_norm_gma_dp = depth_gma_dp * scale_factor_gma_dp
    

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

    if gma:
        distance_gma = (truth - depth_gma)
        distance_norm_gma = (truth - depth_norm_gma)
        ml1_gma.append((np.abs(distance_gma)).mean(axis=None))
        mse_gma.append((np.square(distance_gma)).mean(axis=None))
        ml1_norm_gma.append((np.abs(distance_norm_gma)).mean(axis=None))
        mse_norm_gma.append((np.square(distance_norm_gma)).mean(axis=None))
    
    if dp:
        distance_dp = (truth - depth_dp)
        distance_norm_dp = (truth - depth_norm_dp)
        ml1_dp.append((np.abs(distance_dp)).mean(axis=None))
        mse_dp.append((np.square(distance_dp)).mean(axis=None))
        ml1_norm_dp.append((np.abs(distance_norm_dp)).mean(axis=None))
        mse_norm_dp.append((np.square(distance_norm_dp)).mean(axis=None))

    if dp_gma:
        distance_gma_dp = (truth - depth_gma_dp)
        distance_norm_gma_dp = (truth - depth_norm_gma_dp)
        ml1_gma_dp.append((np.abs(distance_gma_dp)).mean(axis=None))
        mse_gma_dp.append((np.square(distance_gma_dp)).mean(axis=None))
        ml1_norm_gma_dp.append((np.abs(distance_norm_gma_dp)).mean(axis=None))
        mse_norm_gma_dp.append((np.square(distance_norm_gma_dp)).mean(axis=None))

    #Error(norm) vis:
    viz_path = os.path.join(norm_error_vis_path, file.split(".")[0] + ".png") 
    distance_norm+=1
    distance_norm.squeeze()
    inv_depth =  np.divide(1., distance_norm, out=np.zeros_like(distance_norm), where=distance_norm!=0)
    save_image(viz_path, inv_depth)

    #Error vis:
    viz_path = os.path.join(error_vis_path, file.split(".")[0] + ".png") 
    distance+=1
    distance.squeeze()
    inv_depth =  np.divide(1., distance, out=np.zeros_like(distance), where=distance!=0)
    save_image(viz_path, inv_depth)

    #Error(initial, norm) vis:
    viz_path = os.path.join(norm_initial_error_vis_path, file.split(".")[0] + ".png") 
    distance_norm_initial+=1
    distance_norm_initial.squeeze()
    inv_depth =  np.divide(1., distance_norm_initial, out=np.zeros_like(distance_norm_initial), where=distance_norm_initial!=0)
    save_image(viz_path, inv_depth)

    #Error(initial) vis:
    viz_path = os.path.join(initial_error_vis_path, file.split(".")[0] + ".png") 
    distance_initial+=1
    distance_initial.squeeze()
    inv_depth =  np.divide(1., distance_initial, out=np.zeros_like(distance_initial), where=distance_initial!=0)
    save_image(viz_path, inv_depth)

    if gma:
        #Error(GMA) vis:
        viz_path = os.path.join(gma_error_vis_path, file.split(".")[0] + ".png") 
        distance_gma+=1
        distance_gma.squeeze()
        inv_depth =  np.divide(1., distance_gma, out=np.zeros_like(distance_gma), where=distance_gma!=0)
        save_image(viz_path, inv_depth)

        #Error(GMA,norm) vis:
        viz_path = os.path.join(norm_gma_error_vis_path, file.split(".")[0] + ".png") 
        distance_norm_gma+=1
        distance_norm_gma.squeeze()
        inv_depth =  np.divide(1., distance_norm_gma, out=np.zeros_like(distance_norm_gma), where=distance_norm_gma!=0)
        save_image(viz_path, inv_depth)

    if dp:
        #Error(dp) vis:
        viz_path = os.path.join(dp_error_vis_path, file.split(".")[0] + ".png") 
        distance_dp+=1
        distance_dp.squeeze()
        inv_depth =  np.divide(1., distance_dp, out=np.zeros_like(distance_dp), where=distance_dp!=0)
        save_image(viz_path, inv_depth)

        #Error(dp,norm) vis:
        viz_path = os.path.join(norm_dp_error_vis_path, file.split(".")[0] + ".png") 
        distance_norm_dp+=1
        distance_norm_dp.squeeze()
        inv_depth =  np.divide(1., distance_norm_dp, out=np.zeros_like(distance_norm_dp), where=distance_norm_dp!=0)
        save_image(viz_path, inv_depth)

    if dp_gma:
        #Error(gma_dp) vis:
        viz_path = os.path.join(gma_dp_error_vis_path, file.split(".")[0] + ".png") 
        distance_gma_dp+=1
        distance_gma_dp.squeeze()
        inv_depth =  np.divide(1., distance_gma_dp, out=np.zeros_like(distance_gma_dp), where=distance_gma_dp!=0)
        save_image(viz_path, inv_depth)

        #Error(gma_dp,norm) vis:
        viz_path = os.path.join(norm_gma_dp_error_vis_path, file.split(".")[0] + ".png") 
        distance_norm_gma_dp+=1
        distance_norm_gma_dp.squeeze()
        inv_depth =  np.divide(1., distance_norm_gma_dp, out=np.zeros_like(distance_norm_gma_dp), where=distance_norm_gma_dp!=0)
        save_image(viz_path, inv_depth)

    #Color vis for truth:
    #viz_path = os.path.join(truth_viz_path, file.split(".")[0] + ".png") 
    #truth.squeeze()
    #inv_depth = np.divide(1., truth, out=np.zeros_like(truth), where=truth!=0)
    #save_image(viz_path, inv_depth)

#print(mse)
print("Evaluation for scenario "+str(name))
if use_scales:
    print("Scales read")
print("Results (Fine-Tuned):")
aml1=np.nanmean(ml1)
amse=np.nanmean(mse)
print("Ml1: "+str(aml1))
print("MSE: "+str(amse))
if per_frame:
    print(mse)
aml1_norm=np.nanmean(ml1_norm)
amse_norm=np.nanmean(mse_norm)
print("Ml1(norm): "+str(aml1_norm))
print("MSE(norm): "+str(amse_norm))

if gma:
    print("\nResults (GMA):")
    aml1_gma=np.nanmean(ml1_gma)
    amse_gma=np.nanmean(mse_gma)
    print("Ml1: "+str(aml1_gma))
    print("MSE: "+str(amse_gma))
    aml1_norm_gma=np.nanmean(ml1_norm_gma)
    amse_norm_gma=np.nanmean(mse_norm_gma)
    print("Ml1(norm): "+str(aml1_norm_gma))
    print("MSE(norm): "+str(amse_norm_gma))

if dp:
    print("\nResults (dp):")
    aml1_dp=np.nanmean(ml1_dp)
    amse_dp=np.nanmean(mse_dp)
    print("Ml1: "+str(aml1_dp))
    print("MSE: "+str(amse_dp))
    if per_frame:
        print(mse_dp)
    aml1_norm_dp=np.nanmean(ml1_norm_dp)
    amse_norm_dp=np.nanmean(mse_norm_dp)
    print("Ml1(norm): "+str(aml1_norm_dp))
    print("MSE(norm): "+str(amse_norm_dp))

if dp_gma:
    print("\nResults (gma_dp):")
    aml1_gma_dp=np.nanmean(ml1_gma_dp)
    amse_gma_dp=np.nanmean(mse_gma_dp)
    print("Ml1: "+str(aml1_gma_dp))
    print("MSE: "+str(amse_gma_dp))
    aml1_norm_gma_dp=np.nanmean(ml1_norm_gma_dp)
    amse_norm_gma_dp=np.nanmean(mse_norm_gma_dp)
    print("Ml1(norm): "+str(aml1_norm_gma_dp))
    print("MSE(norm): "+str(amse_norm_gma_dp))

print("\nResults (Initial):")
aml1_initial=np.nanmean(ml1_initial)
amse_initial=np.nanmean(mse_initial)
print("Ml1: "+str(aml1_initial))
print("MSE: "+str(amse_initial))
aml1_norm_initial=np.nanmean(ml1_norm_initial)
amse_norm_initial=np.nanmean(mse_norm_initial)
print("Ml1(norm): "+str(aml1_norm_initial))
print("MSE(norm): "+str(amse_norm_initial))

