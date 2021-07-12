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


dataset="sintel" #"sintel"/"RGBD"
number=3
fps = 7.5 #TODO
name="shaman_3"
batch_size=[1,2,3,4] 
render_obj=True
use_scales=True
scale=True
scale_f=0.5

preview=False
interactive=True
pp=False
rgbd=True
vis_depth=False
vis_obj=False
vis_mask=False


use_gtruth=False #DEFAULT: False
#if use_gtruth: #TODO: fix
#    rgbd=False
use_initial=False  #DEFAULT: False
use_cvd_dp=True  #DEFAULT: False
if use_cvd_dp:
    use_scales=False
type= "FN"  #FN / GMA / custom /  ...
if use_gtruth:
    custom_intrinsic=o3d.camera.PinholeCameraIntrinsic(1080, 1920, 1671.770118, 1671.770118, 540 , 960)
else:
    custom_intrinsic=o3d.camera.PinholeCameraIntrinsic(224, 384, 1671.770118*(224/1080), 1671.770118*(384/1920), 540*(224/1080) , 960*(384/1920))
norm=False
obj_path=None #frames for obj  DEFAULT:None

accumulate=False


start_index=0 #default=0



if len(sys.argv) > 1:
    name = str(sys.argv[1])
    
if len(sys.argv) > 2:
    type = str(sys.argv[2])

if len(sys.argv) > 3:
    dataset = str(sys.argv[3])

if dataset == "RGBD":
    depth_dataset_path="../RGBD/"
elif dataset == "sintel":
    depth_dataset_path="../MPI-Sintel-depth-training-20150305/"
else:
    print('Only "sintel"/"RGBD" allowed for dataset')
#file_path="/home/umbra/Documents/MPI-Sintel-depth-training-20150305/training/depth/bamboo_2/frame_0001.dpt"
sintel_path="../MPI-Sintel-complete/"

src_path="./data/"+type+"/"+name+"/clean/"

img_path=os.path.join(src_path,"color_down_png")

output_path=os.path.join(src_path,"render_frames")
os.makedirs(output_path, exist_ok=True)

mask_path=os.path.join(src_path,"render_masks")
os.makedirs(mask_path, exist_ok=True)

metadata_path=os.path.join(src_path,"R_hierarchical2_mc/metadata_scaled.npz")
scales_path=os.path.join(src_path,"R_hierarchical2_mc/scales.csv")

metadata = np.load(metadata_path) #TODO: unscale?


#Dont change after here
#----------------------------------------------------------------------------------------------------------------------------------------------------
if dataset == "RGBD":
    folder_name="rgbd_dataset_freiburg"+str(number)+"_"+name
    src_cam_path = os.path.join(depth_dataset_path, folder_name, "groundtruth.txt")
elif dataset == "sintel":
    src_cam_path = os.path.join(depth_dataset_path, "training", "camdata_left", name)
else:
    pass

if not os.path.isdir(src_path):
    print("depth path ("+ src_path +") empty")
    exit()


if use_gtruth:
    depth_path=os.path.join(depth_dataset_path,"training/depth/"+name+"/")
elif use_initial:
    depth_path=os.path.join(src_path,"depth_mc/exact_depth/")
elif use_cvd_dp:
    depth_path=os.path.join("./data","CVD_DP",name,"exact_depth")
else:
    for bs in batch_size: 
        depth_path=os.path.join(src_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(bs)+"_Oadam/exact_depth/")
        if os.path.isfile(depth_path+"/frame_0001.dpt"):
            break
if dataset == "RGBD":
    split = name.split("_")
    folder_name="rgbd_dataset_freiburg"+str(split[0][2:])
    i=1
    while i < len(split):
        folder_name+="_"+split[i]
        i+=1        
    rgbd_path=os.path.join(depth_dataset_path,folder_name)
    truth_path=os.path.join(depth_dataset_path,folder_name,"depth/")
elif dataset == "sintel":
    truth_path=os.path.join(depth_dataset_path,"training/depth/"+name+"/")
    rgbd_path=""
else:
    pass

def parse_scales(path):
    global use_scales
    scales=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            scales.append(float(row[1]))
    #print(len(scales))
    #print(len(img_path))
    if len(scales)==0:
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

def video_from_frames(pathIn,pathOut): #Adapted from: https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    files.sort()
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
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
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()   

def get_depth_frames(frames, path=truth_path ):
    depth_frames=[]    
    i=0
    depth_files=os.listdir(path)
    depth_files.sort()
    for idx, file in enumerate(depth_files):
        if i>=len(frames):
            break
        #if float(row[0])>float(frames[i])+0.1:
        #    print("WARNING!: Camera extrinsics later than frame by "+str(float(row[0])-float(frames[i]))+" sec at index "+str(i)+"!")
        #print(row[0][:-2])
        #prnt(frames[i][:-4])
        ts=file.split(".")[0]+"."+file.split(".")[1]
        ts_b=depth_files[idx-1].split(".")[0]+"."+depth_files[idx-1].split(".")[1]
        if float(ts) >= float(frames[i]):
            if abs(float(ts) >= float(frames[i])) < abs(float(ts_b) >= float(frames[i])):
                depth_frames.append(ts)
            else:
                depth_frames.append(ts_b)           
            
            i+=1
    return depth_frames

def parse_frames(path=os.path.join(rgbd_path,"frames_for_cvd.txt")):
    frames=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            frames.append(row[0])
    return frames

#Reads extrinsics from RGBD dataset and returns array of [tx ty tz qx qy qz qw] for index-corresponding name in frames:
def parse_extrinsics(frames, path=os.path.join(rgbd_path,"groundtruth.txt")):
    extrinsics=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        i=0
        for row in csv_reader:
            if i>=len(frames):
                break
            if row[0]!="#" and float(row[0])>float(frames[i])+0.1:
                print("WARNING!: Camera extrinsics later than frame by "+str(float(row[0])-float(frames[i]))+" sec at index "+str(i)+"!")
            #print(row[0][:-2])
            #print(frames[i][:-4])
            if row[0]!="#" and float(row[0]) >= float(frames[i]):
                print(frames[i]+":"+row[0])
                extrinsics.append([row[1],row[2],row[3],row[4],row[5],row[6],row[7]])
                i+=1
            
    if len(extrinsics)!=len(frames):
        print("WARNING: number_of_frames doesn't match number of frames in:"+path+"("+str(len(extrinsics))+")")
    #print(extrinsics)
    return extrinsics

scales=parse_scales(scales_path)

files = os.listdir(depth_path)
files.sort()

if norm:
    #Calculate statistical scale factor:
    scale_factor_n=[]
    if dataset == "RGBD":
        frames = parse_frames()
        depth_frames= get_depth_frames(frames)
        files = ["frame_"+str(i+1).zfill(4)+".dpt" for i in range(50)]
    elif dataset == "sintel":
        files = os.listdir(truth_path)
        files.sort()
    else:
        pass

    for i, file in enumerate(files): #["frame_0001.dpt"]:
        if dataset == "RGBD":
            frame = depth_frames[i]
            #print(os.path.join(truth_path, frame+".png"))
            truth = np.array(cv2.imread(os.path.join(truth_path, frame+".png"), cv2.IMREAD_UNCHANGED)).astype(float)
            #print(np.mean(truth))
            truth/=5000.
        elif dataset == "sintel":
            truth = depth_read(os.path.join(truth_path, file))
        else:
            pass
        depth = depth_read(os.path.join(depth_path, file))
        if use_scales:
            depth*=scales[i]
        truth[truth == 100000000000.0] = np.nan
        truth = resize(truth, depth.shape)   

        scale_factor_n.append(np.nanmean(truth)/np.nanmean(depth))  

    scale_factor_n= np.nanmean(scale_factor_n)

if scale:
    #Calculate statistical scale factor:
    scale_factor=[]

    for i, file in enumerate(files): #["frame_0001.dpt"]:
        depth = depth_read(os.path.join(depth_path, file))
        if use_scales:
            depth*=scales[i]
        if norm:           
            depth= depth * scale_factor_n 
        scale_factor.append(np.nanmean(depth))  

    scale_factor= np.nanmean(scale_factor)
    scale_factor=(scale_f/scale_factor)
 


#Compute distance:
if dataset == "RGBD":
    frames_gt=parse_frames()
    extrinsics=parse_extrinsics(frames_gt)

pcs_acc=[]
ml1 =[]
mse =[]
ml1_norm=[]
mse_norm=[]
ix=1
for i, file in enumerate(files): #["frame_0001.dpt"]:

    if ix < start_index:
        ix+=1
        continue

    #Get camera data:
    """
    if type=="custom" or dataset=="custom":
        extrinsic=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        intrinsic= custom_intrinsic
        if i==start_index:
            cam_ex=extrinsic

    elif dataset == "RGBD":

        if number==1: 
            print("WARNING: Invalid results, because colmap can't model distortion! Use Freiburg3 only")
                 
        elif number==2:
            print("WARNING: Invalid results, because colmap can't model distortion! Use Freiburg3 only")
                
        elif number==3:
            if use_gtruth:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(640, 480, 535.4, 539.2, 320.1 , 247.6)
            else:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 288, 535.4*(384/640), 539.2*(288/480), 320.1 *(384/640), 247.6*(288/480))
                print("OK")
        else:
            print("Only number=3 allowed")

                        
        ex=extrinsics[i]
        E=np.array(R.from_quat([float(ex[3]),float(ex[4]),float(ex[5]),float(ex[6])]).as_matrix())

        E = np.append(E, [[float(ex[0])],[float(ex[1])],[float(ex[2])]], axis=1) 
        E = np.append(E, [[0,0,0,1]], axis=0)
        #print(E)
        extrinsic=E
        if i==start_index:
            cam_ex=E

    elif dataset == "sintel":
        #print(src_cam_path)
        if os.path.isdir(src_cam_path) and len(os.listdir(src_cam_path))>0:
            frame_cam = file.split(".")[0]+".cam"
            I,E = cam_read(os.path.join(src_cam_path, frame_cam))
                # colmapintrinsic:      w  h   fx     fy      cx   cy   #Scale c with
                #                    1024 436 1120.0 1120.0 511.5 217.5
                #                     384 160
            if use_gtruth:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
            else:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 160, 1120.0*(384/1024), 1120.0*(160/436), 511.5*(384/1024) , 217.5*(160/436) )
            extrinsic=E
            if i==start_index:
                cam_ex=E
        else:
            print("Extrinsics not available")
            if use_gtruth:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
            else:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 160, 1120.0*(384/1024), 1120.0*(160/436), 511.5*(384/1024) , 217.5*(160/436) )
            extrinsic=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
            if i==start_index:
                #print("OK")
                cam_ex=extrinsic
    else:
        pass
    """
        
    if type=="custom" or dataset=="custom":
        extrinsic=metadata['extrinsics'][i]
        extrinsic = np.append(extrinsic, [[0,0,0,1]], axis=0)
        intrinsic= custom_intrinsic

    elif dataset == "RGBD":

        if number==1: 
            print("WARNING: Invalid results, because colmap can't model distortion! Use Freiburg3 only")
                 
        elif number==2:
            print("WARNING: Invalid results, because colmap can't model distortion! Use Freiburg3 only")
                
        elif number==3:
            if use_gtruth:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(640, 480, 535.4, 539.2, 320.1 , 247.6)
            else:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 288, 535.4*(384/640), 539.2*(288/480), 320.1 *(384/640), 247.6*(288/480))
                print("OK")
        else:
            print("Only number=3 allowed")

        if use_gtruth:
            ex=extrinsics[i]
            E=np.array(R.from_quat([float(ex[3]),float(ex[4]),float(ex[5]),float(ex[6])]).as_matrix())

            E = np.append(E, [[float(ex[0])],[float(ex[1])],[float(ex[2])]], axis=1) 
            E = np.append(E, [[0,0,0,1]], axis=0)
            #print(E)
            extrinsic=E                
        else:
            extrinsic=metadata['extrinsics'][i]
            extrinsic = np.append(extrinsic, [[0,0,0,1]], axis=0)

     

    elif dataset == "sintel":
        #print(src_cam_path)
        if os.path.isdir(src_cam_path) and len(os.listdir(src_cam_path))>0:
                # colmapintrinsic:      w  h   fx     fy      cx   cy   #Scale c with
                #                    1024 436 1120.0 1120.0 511.5 217.5
                #                     384 160
            if use_gtruth:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
                frame_cam = file.split(".")[0]+".cam"
                I,E = cam_read(os.path.join(src_cam_path, frame_cam))
                extrinsic=E
                print(extrinsic)
                print(metadata['extrinsics'][i])
            else:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 160, 1120.0*(384/1024), 1120.0*(160/436), 511.5*(384/1024) , 217.5*(160/436) )
                extrinsic=metadata['extrinsics'][i]
                extrinsic = np.append(extrinsic, [[0,0,0,1]], axis=0)

        else:
            print("Extrinsics not available")
            if use_gtruth:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
                extrinsic=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
            else:
                intrinsic=o3d.camera.PinholeCameraIntrinsic(384, 160, 1120.0*(384/1024), 1120.0*(160/436), 511.5*(384/1024) , 217.5*(160/436) )
                extrinsic=metadata['extrinsics'][i]
                extrinsic = np.append(extrinsic, [[0,0,0,1]], axis=0)
            

    else:
        pass   
    depth = depth_read(os.path.join(depth_path, file))
    if use_scales:
        depth*=scales[i]
        
    if scale:
        depth*=scale_factor 

   
    
    if norm:           
        depth= depth * scale_factor_n 
        distance = (truth - depth)
        ml1.append((np.abs(distance)).mean(axis=None))
        mse.append((np.square(distance)).mean(axis=None))
        
    if rgbd:
        if use_gtruth:
            frame_img = file.split(".")[0]+".png"
            frame_path= os.path.join(sintel_path,"training","clean",name,frame_img)
            #print(frame_path)
            rgb_image=cv2.imread(frame_path)
            #print(rgb_image)
            rgb_image=o3d.geometry.Image(rgb_image)
        else:
            split1 = file.split("_")
            split2 = split1[1].split(".")
            index = int(split2[0])-1
            index =str(index).zfill(6)
            file_new = str(split1[0]+"_"+index+".png") 
            frame_path= os.path.join(img_path,file_new)
            rgb_image=cv2.imread(frame_path)          
            rgb_image=o3d.geometry.Image(rgb_image)
    print(np.min(depth))
    print(np.mean(depth))
    depth_img_d=o3d.geometry.Image(depth)
    if rgbd:
        rgbd_img_d=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_img_d, depth_scale=1, depth_trunc=1000.0, convert_rgb_to_intensity=False)
        disable_extrinsics=False
        if disable_extrinsics:
            pc_d=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img_d, intrinsic, extrinsic=cam_ex) #Keeping first ex -> closer result, and drift in other direction -> ex overcompensating?
        else:
            pc_d=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img_d, intrinsic, extrinsic=extrinsic) #Keeping first ex -> closer result, and drift in other direction -> ex overcompensating?
        #extrinsic*=0.01
        #print(extrinsic)
    else:
        pc_d=o3d.geometry.PointCloud.create_from_depth_image(depth_img_d, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #depth_scale doesn't work is like 1

    #intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5, 217.5)
    #TODO: create objs:
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    obj1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=20, create_uv_map=False)
    #obj2= copy.deepcopy(obj1).translate((0, -40, 3))
    #ambush_5:
    #obj2= copy.deepcopy(obj1).translate((4.6*scale_f, 100.0e+02*scale_f, 26.*scale_f))
    #obj2= copy.deepcopy(obj1).translate((-6.8, -37.3, 1.3)) #red,green,blue? 
    #t=np.array([2.5,-2,2.3]) #-x(l->r),-z(c->f),-y(t->b) #ambush_5 1 scales=True radius=0.1:[2.5,-2.6,2.2]

    #Wave w/o extrinsics: 
    #xl=-0.19
    #xr=0.19
    #x=xl+(xr-xl)/len(files)*i
    #t=np.array([x,-0.16,0.6]) #x(l->r),y(t->b),z(c->f) -center at 0,0,z #wave 1 scales=True:[-0.085,-0.16,0.6]

    #Wave 1 w extrinsics:
    #xl=0.33
    #xr=-0.07
    #x=xl-np.abs(xr-xl)/len(files)*i#i
    #zl=-0.6
    #zr=-0.40
    #z=zl+np.abs(zr-zl)/len(files)*i#i
    #t=np.array([x,0.15,z]) #-x(l->r)?,y(t->b)?,-z(c->f), -center at 0,0,z #wave 1 radius=0.01 scales=True:[x,0.16,-0.6]

    #Wave 1 w broken extrinsics:
    xl=0.33
    xr=-1.35
    x=xl-np.abs(xr-xl)/len(files)*i#i
    yl=0.17
    yr=0.17
    y=yl+np.abs(yr-yl)/len(files)*i#i
    zl=-0.6
    zr=-0.7
    z=zl-np.abs(zr-zl)/len(files)*i#i
    t=np.array([x,y,z]) #-x(l->r)?,y(t->b)?,-z(c->f), -center at 0,0,z #wave 1 radius=0.01 scales=True:[x,0.16,-0.6]
    if i>60:
        render_obj=False
    print(i)
    """
    [0. 0. 0.]
    [ 0.396352   -0.05877081  0.14071229] #red,green,blue? 

    [1. 0. 0.]
    [ 1.39064884 -0.05222342  0.03426548]

    [0. 1. 0.]
    [0.39101239 0.94114733 0.15233972]

    [0. 0. 1.]
    [ 0.50286623 -0.06976354  1.1349627 ]
    """
    use_intr=False
    if use_intr: #TODO: fix
        if scale:
            t[2]*=scale_f
        print(t)
        I=np.array(intrinsic.intrinsic_matrix)
        print(I)
        t=np.matmul(I,t)
        print(t)
        #t[0]/=t[2]
        #t[1]/=t[2]
        print(t)
        t[0]-=I[0][2]*t[2]
        t[1]-=I[1][2]*t[2]
    else:
        if scale:
            t*=scale_f 
    print(t)
    t=np.append(t, [1], axis=0)
    cam_ex=metadata['extrinsics'][0]
    cam_ex=np.append(cam_ex, [[0,0,0,1]], axis=0)
    t=np.matmul(np.array(cam_ex[:-1]),t) #TODO: use cam_ex or extrinsic
    #print(cam_ex)
    print(t)
    #t=t[:-1]
    if type=="custom" or dataset=="custom":
        obj2= copy.deepcopy(obj1).translate((-t[0],-t[1],-t[2])) #red,green,blue? 
    elif dataset=="sintel":
        obj2= copy.deepcopy(obj1).translate((-t[0],-t[2],-t[1])) #red,green,blue?  
    else:
        pass
    objs=[obj2]
    """
    t=np.array([1,0,0]) #x,y,z?
    t=np.append(t, [1], axis=0)
    t=np.matmul(np.array(cam_ex[:-1]),t)
    print(cam_ex)
    print(t)
    #t=t[:-1]
    obj3= copy.deepcopy(obj1).translate((-t[0],-t[2],-t[1])) #red,green,blue?  

    t=np.array([0,-10,0]) #x,y,z?
    t=np.append(t, [1], axis=0)
    t=np.matmul(np.array(cam_ex[:-1]),t)
    print(cam_ex)
    print(t)
    #t=t[:-1]
    obj4= copy.deepcopy(obj1).translate((-t[0],-t[2],-t[1])) #red,green,blue? 

    t=np.array([0,0,1]) #x,y,z?
    t=np.append(t, [1], axis=0)
    t=np.matmul(np.array(cam_ex[:-1]),t)
    print(cam_ex)
    print(t)
    #t=t[:-1]
    obj5= copy.deepcopy(obj1).translate((-t[0],-t[2],-t[1])) #red,green,blue?  
    objs=[obj2,obj3,obj4,obj5]#cf]#,obj1,obj2]
    """    
 
    

    pcs=[pc_d]   
    if accumulate:
        pcs+=pcs_acc
    if render_obj:
        pcs+=objs
    print("\n"+str(file)+":")

    
    

    if interactive and not pp:
        o3d.visualization.gui.Application.instance.initialize()
        w = o3d.visualization.O3DVisualizer("03DVisualizer",1024, 436)
        w.reset_camera_to_default()
        w.setup_camera(intrinsic,extrinsic) 

        #obj2.paint_uniform_color([0, 0, 0]) #Black #TODO: set obj colors
        #obj3.paint_uniform_color([1, 0, 0]) #Red #TODO: set obj colors
        #obj4.paint_uniform_color([0.5, 0.706, 0.5]) #Green #TODO: set obj colors
        #obj5.paint_uniform_color([0, 0.651, 0.929]) #Blue #TODO: set obj colors
        obj2.paint_uniform_color([1, 0, 0]) #Red #TODO: set obj colors
        if not rgbd:
            pc_d.paint_uniform_color([0.5, 0.706, 0.5]) #green
        w.scene.set_background(np.array([1.,1.,1.,1.])) #white

        #w.scene.set_background(np.array([1., 1., 1., 1.])) #Black
        o3d.visualization.gui.Application.instance.add_window(w)
        w.show_axes = True
        w.show_ground = True
        w.show_settings = True
        w.point_size=7
        w.size_to_fit() #Full screen
        mat = o3d.visualization.rendering.Material()
        #mat.base_color = [1.0, 1.0, 1.0, 1.0]
        mat.shader = 'defaultUnlit'
        for pc in pcs: 
                print("added pc")
                w.add_geometry(str(ix),pc,mat)
                ix+=1
        o3d.visualization.gui.Application.instance.run()
        print(np.asarray(o3d.visualization.PickedPoint.coord))
        #o3d.visualization.gui.Application.instance.quit()
        #w.export_current_image("test.png")
    elif interactive and pp:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        for pc in pcs:
            vis.add_geometry(pc)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
    else:
        if dataset=="custom" or type=="custom":
            if use_gtruth:
                renderer= o3d.visualization.rendering.OffscreenRenderer(1080, 1920,headless=False)
            else:
                renderer= o3d.visualization.rendering.OffscreenRenderer(224, 384,headless=False)
        elif dataset=="sintel":
            if use_gtruth:
                renderer= o3d.visualization.rendering.OffscreenRenderer(1024, 436,headless=False)
            else:
                renderer= o3d.visualization.rendering.OffscreenRenderer(384, 160,headless=False)
        else:
            pass
        renderer.setup_camera(intrinsic,extrinsic)        
        mat = o3d.visualization.rendering.Material()
        #mat.base_color = [1.0, 1.0, 1.0, 1.0]
        mat.shader = 'defaultUnlit'
        frame = file.split(".")[0]+".png"


        if obj_path is None: #Render obj
            obj_path=os.path.join(src_path,"render_objs")
            os.makedirs(obj_path, exist_ok=True)

        obj2.paint_uniform_color([1, 0, 0]) #Red #TODO: set obj colors
        pc_d.paint_uniform_color([1.,1.,1.]) #white
        renderer.scene.set_background(np.array([1.,1.,1.,1.])) #white
           
        #scene=o3d.visualization.rendering.Open3DScene(renderer)

        for pc in pcs: 
            #print("added pc")
            renderer.scene.add_geometry(str(ix),pc,mat)
            ix+=1
        img_obj = renderer.render_to_image()    
        img_obj_np = np.array(img_obj)
        img_obj_cv = cv2.cvtColor(img_obj_np, cv2.COLOR_RGBA2BGRA)
        if vis_depth:
            img_d = renderer.render_to_depth_image()
            img_d_cv = cv2.cvtColor(np.array(img_d), cv2.COLOR_GRAY2BGR)


        if vis_obj:
            if vis_depth:
                cv2.imshow("Preview window", img_d_cv)
                cv2.waitKey()
            cv2.imshow("Preview window", img_obj_cv)
            cv2.waitKey()

        save_path = os.path.join(obj_path,frame)
        o3d.io.write_image(save_path, img_obj)

            
        
        #Render Mask:
        for obj in objs:
            obj.paint_uniform_color([1., 1., 1.]) #White
        pc_d.paint_uniform_color([0., 0., 0.]) #black        
        renderer.scene.set_background(np.array([0., 0., 0., 0.])) #Black

        for pc in pcs: 
            #print("added pc")
            renderer.scene.add_geometry(str(ix),pc,mat)
            ix+=1
        mask_rgb = renderer.render_to_image()    
        mask_rgb_cv = cv2.cvtColor(np.array(mask_rgb), cv2.COLOR_RGBA2BGRA)
        mask_rgb_np = np.array(mask_rgb)

        # Get boolean mask from rgb image:
        #print(mask_rgb_np.shape)
        [rows, columns, channels] = mask_rgb_np.shape
        mask = np.zeros((rows,columns))
        for row in range(rows):
            for column in range(columns):
                #print(mask_rgb_np[row,column,0])
                if(mask_rgb_np[row,column,0]>230): #TODO: adjust + increase light source for mask; alt: swap black/white
                    mask[row,column] = 1
                else:
                    mask[row,column] = 0    

        #print(mask)
        #print(mask.shape)
        #print(np.min(mask))
        #print(np.mean(np.array(mask)))
        #print(np.max(mask))
        #mask_cv = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)     
        save_path = os.path.join(mask_path,frame)
        #o3d.io.write_image(save_path, mask_rgb_cv)
        if vis_mask:
            cv2.imshow("Preview window", mask)
            cv2.waitKey()
        cv2.imwrite(save_path, mask)


        #Create final image:
        if use_gtruth:
            frame_img = file.split(".")[0]+".png"
            frame_path= os.path.join(sintel_path,"training","clean",name,frame_img)
            #print(frame_path)
            img=cv2.imread(frame_path)
        else:
            split1 = file.split("_")
            split2 = split1[1].split(".")
            index = int(split2[0])-1
            index =str(index).zfill(6)
            file_new = str(split1[0]+"_"+index+".png") 
            frame_path= os.path.join(img_path,file_new)
            img=cv2.imread(frame_path) 
        print(img.shape)
        print(mask.shape)
        [rows, columns, channels] = img.shape
        result = img.copy()
        for row in range(rows):
            for column in range(columns):
                #print(mask_rgb_np[row,column,0])
                if(mask[row,column]==1):
                    result[row,column] = img_obj_np[row,column][::-1]
                #else:
                    #result[row,column] = img[row,column]  
        #print(result)
        
        if preview:
            #result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGRA)
            cv2.imshow("Preview window", result)
            cv2.waitKey()

        save_path = os.path.join(output_path,frame)
        cv2.imwrite(save_path, result)
        print("Frame rendered")


    if accumulate and i==0:
        pcs_acc+=pcs
    ix+=1

    #break #TODO: delete

#TODO: Create video
video_path=os.path.join(src_path, "rendered_video.mp4")
video_from_frames(output_path,video_path)
print("Video with "+str(fps)+" fps created at "+video_path)






