import os
import re
import sys
from posix import listdir
from shutil import copyfile
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.transform import resize
import utils.image_io
import csv
import open3d as o3d #pip install open3d

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'




name="shaman_3"
batch_size=[1,2,3,4] #TODO
is_pose=True
viz=True

gtruth=True
norm=False
not_norm=True
init=False
standart=True 
gma=False
dp=False
dp_gma=False
cvd_dp=False
accumulate=False
rainbow=False
interactive=True
use_scales=True
scale=True
scale_f=0.1

start_index=0 #default=0

if len(sys.argv) > 1:
    name = str(sys.argv[1])

sintel_depth_path =  "../MPI-Sintel-depth-training-20150305" #TODO

#file_path="/home/umbra/Documents/MPI-Sintel-depth-training-20150305/training/depth/bamboo_2/frame_0001.dpt"
depth_dataset_path="../MPI-Sintel-depth-training-20150305/"
if is_pose:
    src_path="./data/FN/"+name+"/clean/"
else:
    src_path="./data/FN_wo_pose/"+name+"/clean/"
gma_path="./data/GMA/"+name+"/clean/"
dp_path="./data/FN_DP/"+name+"/clean/"
gma_dp_path="./data/GMA_DP/"+name+"/clean/"

if is_pose:
    src_path="./data/FN/"+name+"/clean/"
    scales_path=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")
else:
    src_path="./data/FN_wo_pose/"+name+"/clean/"
    scales_path=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")

gma_path="./data/GMA/"+name+"/clean/"
scales_path_gma=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")

dp_path="./data/FN_DP/"+name+"/clean/"
scales_path_dp=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")

gma_dp_path="./data/GMA_DP/"+name+"/clean/"
scales_path_gma_dp=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")

#Dont change after here
#----------------------------------------------------------------------------------------------------------------------------------------------------
src_cam_path = os.path.join(sintel_depth_path, "training", "camdata_left", name)

if gma and not os.path.isdir(gma_path):
    print("GMA depth folder ("+gma_path+") empty")
    gma=False
if dp and not os.path.isdir(dp_path):
    print("DensePose depth folder ("+dp_path+") empty")
    dp=False
if dp_gma and not os.path.isdir(gma_dp_path):
    print("GMA DensePose depth folder ("+gma_dp_path+") empty")
    dp_gma=False

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
depth_cvd_dp_path=os.path.join("./data/CVD_DP/",name,"exact_depth")
truth_path=os.path.join(depth_dataset_path,"training/depth/"+name+"/")


#Path for colored truth depth visualization:
if gtruth and viz:
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

def cam_read(filename):  #Adapted from sintel_io.py from http://sintel.is.tue.mpg.de/depth
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
    N = np.append(N, [[0,0,0,1]], axis=0)

    return M,N

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

scales_gma=parse_scales(scales_path_gma)

scales_dp=parse_scales(scales_path_dp)

scales_gma_dp=parse_scales(scales_path_gma_dp)

if gtruth:
    files = os.listdir(truth_path)
elif standart:
    files = os.listdir(depth_path)
else:
    files = os.listdir(truth_path)
files.sort()
if (gtruth or norm) and standart:
    #Calculate statistical scale factor:
    scale_factor=[]
    scale_factor_initial=[]
    scale_factor_gma=[]
    scale_factor_dp=[]
    scale_factor_gma_dp=[]

    files.sort()
    for i, file in enumerate(files): #["frame_0001.dpt"]:
        truth = depth_read(os.path.join(truth_path, file))
        if scale:
            truth*=scale_f
        depth = depth_read(os.path.join(depth_path, file))
        if scale:
            depth*=scale_f
        if use_scales:
            depth*=scales[i]
        truth[truth == 100000000000.0] = np.nan
        truth = resize(truth, depth.shape)
        if gma:
            depth_gma = depth_read(os.path.join(depth_gma_path, file))
            if use_scales:
                depth_gma*=scales_gma[i]
            if scale:
                depth_gma*=scale_f
        if dp:
            depth_dp = depth_read(os.path.join(depth_dp_path, file))
            if use_scales:
                depth_dp*=scales_dp[i]
            if scale:
                depth_dp*=scale_f
        if dp_gma:
            depth_gma_dp = depth_read(os.path.join(depth_gma_dp_path, file))
            if use_scales:
                depth_gma_dp*=scales_gma_dp[i]
            if scale:
                depth_dp*=scale_f
                
        if cvd_dp:
            depth_cvd_dp = depth_read(os.path.join(depth_cvd_dp_path, file))
            if scale:
                depth_cvd_dp*=scale_f

        #depth = resize(depth, truth.shape)
        depth_initial = depth_read(os.path.join(initial_path, file))
        if use_scales:
            depth*=scales[i]
        if scale:
            depth_initial*=scale_f


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
pcs_acc=[]
ix=1
for i, file in enumerate(files): #["frame_0001.dpt"]:

    if ix < start_index:
        ix+=1
        continue

    #Get camera data:
    if len(os.listdir(src_cam_path))>0:
        frame_cam = file.split(".")[0]+".cam"
        I,E = cam_read(os.path.join(src_cam_path, frame_cam))
        if init or standart or gma or dp or cvd_dp:            
            intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 160, 1120.0*(384/1024), 1120.0*(160/436), 511.5*(384/1024) , 217.5*(160/436) )
        else:
            intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
        extrinsic=E
    else:
        print("Extrinsics not available")
        if init or standart or gma or dp or cvd_dp:            
            intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 160, 1120.0*(384/1024), 1120.0*(160/436), 511.5*(384/1024) , 217.5*(160/436) )
        else:
            intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
        extrinsic=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


        
    if standart:
        depth = depth_read(os.path.join(depth_path, file))
        if use_scales:
            depth*=scales[i]
        if scale:
            depth*=scale_f
    #depth = resize(depth, truth.shape)
    if init:
        depth_initial = depth_read(os.path.join(initial_path, file))
        if use_scales:
            depth_initial*=scales[i]
        if scale:
            depth_initial*=scale_f
    
    if gma:
        depth_gma = depth_read(os.path.join(depth_gma_path, file))
        if use_scales:
            depth_gma*=scales_gma[i]
        if scale:
            depth_gma*=scale_f
        
    if dp:
        depth_dp = depth_read(os.path.join(depth_dp_path, file))
        if use_scales:
            depth_dp*=scales_dp[i]
        if scale:
            depth_dp*=scale_f
    if dp_gma:
        depth_gma_dp = depth_read(os.path.join(depth_gma_dp_path, file))
        if use_scales:
            depth_gma_dp*=scales_gma_dp[i]
        if scale:
            depth_gma_dp*=scale_f

    if cvd_dp:
        depth_cvd_dp = depth_read(os.path.join(depth_cvd_dp_path, file))
        if scale:
            depth_cvd_dp*=scale_f

    if gtruth:
        truth = depth_read(os.path.join(truth_path, file))
        if scale:
            truth*=scale_f
        truth[truth == 100000000000.0] = np.nan
        if standart:
            truth = resize(truth, depth.shape)
        elif init:
            truth = resize(truth, depth_initial.shape)
        elif gma:
            truth = resize(truth, depth_gma.shape)
        elif dp:
            truth = resize(truth, depth_dp.shape)
        elif cvd_dp:
            truth = resize(truth, depth_cvd_dp.shape)

    if (gtruth or norm) and standart:   
        
        depth_norm = depth * scale_factor 
        depth_norm_initial = depth_initial * scale_factor_initial

        if gma:
            depth_norm_gma = depth_gma * scale_factor_gma
        if dp:
            depth_norm_dp = depth_dp * scale_factor_dp
        if dp_gma:
            depth_norm_gma_dp = depth_gma_dp * scale_factor_gma_dp
        if cvd_dp:
            #depth_norm_cvd_dp = depth_cvd_dp * scale_factor_cvd_dp
            pass #TODO
        

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

    # colmapintrinsic:      w  h   fx     fy      cx   cy   #Scale c with
    #                    1024 436 1120.0 1120.0 511.5 217.5
    #                     384 160
    if gtruth:
        depth_img_t=o3d.geometry.Image(truth)
        pc_t=o3d.geometry.PointCloud.create_from_depth_image(depth_img_t, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

    if standart:
        if not_norm:
            depth_img=o3d.geometry.Image(depth)
            pc=o3d.geometry.PointCloud.create_from_depth_image(depth_img, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

        if norm:
            depth_img_n=o3d.geometry.Image(depth_norm)
            pc_n=o3d.geometry.PointCloud.create_from_depth_image(depth_img_n, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

    if init:
        if norm:
            depth_img_i_n=o3d.geometry.Image(depth_norm_initial)
            pc_i_n=o3d.geometry.PointCloud.create_from_depth_image(depth_img_i_n, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

        if not_norm:
            depth_img_i=o3d.geometry.Image(depth_initial)
            pc_i=o3d.geometry.PointCloud.create_from_depth_image(depth_img_i, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

    if gma:
        if norm:
            depth_img_g_n=o3d.geometry.Image(depth_norm_gma)
            pc_g_n=o3d.geometry.PointCloud.create_from_depth_image(depth_img_g_n, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

        if not_norm:
            depth_img_g=o3d.geometry.Image(depth_gma)
            pc_g=o3d.geometry.PointCloud.create_from_depth_image(depth_img_g, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)
    if dp:
        if norm:
            depth_img_d_n=o3d.geometry.Image(depth_norm_dp)
            pc_d_n=o3d.geometry.PointCloud.create_from_depth_image(depth_img_d_n, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

        if not_norm:
            depth_img_d=o3d.geometry.Image(depth_dp)
            pc_d=o3d.geometry.PointCloud.create_from_depth_image(depth_img_d, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

    if dp_gma:
        if norm:
            depth_img_g_d_n=o3d.geometry.Image(depth_norm_gma_dp)
            pc_g_d_n=o3d.geometry.PointCloud.create_from_depth_image(depth_img_g_d_n, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

        if not_norm:
            depth_img_g_d=o3d.geometry.Image(depth_gma_dp)
            pc_g_d=o3d.geometry.PointCloud.create_from_depth_image(depth_img_g_d, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

    if cvd_dp:
        if norm:
            depth_img_cvd_dp__n=o3d.geometry.Image(depth_norm_cvd_dp)
            pc_cvd_dp_n=o3d.geometry.PointCloud.create_from_depth_image(depth_img_cvd_dp_n, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

        if not_norm:
            depth_img_cvd_dp=o3d.geometry.Image(depth_cvd_dp)
            pc_cvd_dp=o3d.geometry.PointCloud.create_from_depth_image(depth_img_cvd_dp, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

    #intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5, 217.5)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=20, create_uv_map=False)
   


    pcs=[]
    if gtruth:
        pcs.append(pc_t)
    if standart:
        if not_norm:
            pcs.append(pc)
        if norm:
            pcs.append(pc_n)
    if init:
        if not_norm:
            pcs.append(pc_i)
        if norm:
            pcs.append(pc_i_n)
    if gma:
        if not_norm:
            pcs.append(pc_g)
        if norm:
            pcs.append(pc_g_n)
    if dp:
        if not_norm:
            pcs.append(pc_d)
        if norm:
            pcs.append(pc_d_n)
    if dp_gma:
        if not_norm:
            pcs.append(pc_g_d)
        if norm:
            pcs.append(pc_g_d_n)
    if cvd_dp:
        if not_norm:
            pcs.append(pc_cvd_dp)
        if norm:
            pcs.append(pc_cvd_dp_n)
    if accumulate:
        pcs+=pcs_acc
    print("\n"+str(file)+":")
    if gtruth and not rainbow:
        print("ground truth = green")
        pc_t.paint_uniform_color([0.5, 0.706, 0.5]) #green
    #o3d.visualization.draw_geometries([pc_t])
    if standart:
        if not_norm and not rainbow:
            print("depth = light red")
            pc.paint_uniform_color([1, 0, 0]) #red
        if norm and not rainbow:
            print("depth(norm) = dark red")
            pc_n.paint_uniform_color([0.55, 0, 0]) #dark red
    if init:
        if not_norm and not rainbow:
            print("depth initial = light blue")
            pc_i.paint_uniform_color([0, 0.651, 0.929]) #light blue
        if norm and not rainbow:
            print("depth initial(norm) = dark blue")
            pc_i_n.paint_uniform_color([0, 0, 1]) #dark blue
    if gma:
        if not_norm and not rainbow:
            print("depth gma = pink")
            pc_g.paint_uniform_color([0.9, 0.2, 0.84]) #pink
        if norm and not rainbow:
            print("depth gma(norm) = purple")
            pc_g_n.paint_uniform_color([0.5, 0.195, 0.66]) #purple
    if dp:
        if not_norm and not rainbow:
            print("depth dp = gray")
            pc_d.paint_uniform_color([0.31, 0.31, 0.31]) #gray
        if norm and not rainbow:
            print("depth dp(norm) = black")
            pc_d_n.paint_uniform_color([0., 0., 0.]) #black
    if dp_gma:
        if not_norm and not rainbow:
            print("depth gma dp = yellow")
            pc_g_d.paint_uniform_color([1, 0.706, 0]) #yellow
        if norm and not rainbow:
            print("depth gma dp(norm) = orange")
            pc_g_d_n.paint_uniform_color([1, 0.58, 0.]) #orange
    if cvd_dp:
        if not_norm and not rainbow:
            print("depth cvd_dp = turquoise")
            pc_cvd_dp.paint_uniform_color([0, 1, 0.98]) #turquoise
        if norm and not rainbow:
            print("depth cvd_dp(norm) = dark turquoise")
            pc_cvd_dp_n.paint_uniform_color([0, 0.5, 0.49]) #dark turquoise   
    if interactive:
        o3d.visualization.gui.Application.instance.initialize()
        w = o3d.visualization.O3DVisualizer("03DVisualizer",1024, 436)
        w.reset_camera_to_default()
        w.setup_camera(intrinsic,extrinsic)
        #w.scene.set_background(np.array([1., 1., 1., 1.])) #Black
        o3d.visualization.gui.Application.instance.add_window(w)
        w.show_axes = True
        w.show_settings = True
        w.point_size=7
        w.size_to_fit() #Full screen
        #w.size_to_fit() #Full screen
        mat = o3d.visualization.rendering.Material()
        #mat.base_color = [1.0, 1.0, 1.0, 1.0]
        mat.shader = 'defaultUnlit'
        for pc in pcs: 
                print("added pc")
                w.add_geometry(str(ix),pc,mat)
                ix+=1
        o3d.visualization.gui.Application.instance.run()
    else:
        print(pcs)
        o3d.visualization.draw_geometries(pcs)
        
    
 
    if accumulate:
        pcs_acc+=pcs
    ix+=1







