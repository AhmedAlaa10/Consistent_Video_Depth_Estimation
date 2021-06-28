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
import copy

import open3d as o3d #pip install open3d
import open3d.visualization.rendering as rendering
import cv2 #pip install cv2

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'




name="shaman_3"
batch_size=1 #TODO
batch_size_gma=4
is_pose=True
viz=True
render_obj=True

gtruth=True
norm=False
not_norm=True
init=False
standart=False 
gma=False
accumulate=False

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



output_path=os.path.join(src_path,"evaluation")
os.makedirs(output_path, exist_ok=True)

#Dont change after here
#----------------------------------------------------------------------------------------------------------------------------------------------------
src_cam_path = os.path.join(sintel_depth_path, "training", "camdata_left", name)

if not os.path.isdir(gma_path):
    gma=False


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

depth_path=os.path.join(src_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(batch_size)+"_Oadam/exact_depth/")
initial_path=os.path.join(src_path,"depth_mc/exact_depth/")
depth_gma_path=os.path.join(gma_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(batch_size_gma)+"_Oadam/exact_depth/")
truth_path=os.path.join(depth_dataset_path,"training/depth/"+name+"/")


#Path for colored truth depth visualization:
if gtruth and viz:
    truth_viz_path=os.path.join(depth_dataset_path,"training/dept_viz_col/") 
    os.makedirs(truth_viz_path, exist_ok=True)
    truth_viz_path=os.path.join(truth_viz_path,name)
    os.makedirs(truth_viz_path, exist_ok=True)

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

    files.sort()
    for file in files: #["frame_0001.dpt"]:
        truth = depth_read(os.path.join(truth_path, file))
        depth = depth_read(os.path.join(depth_path, file))
        truth[truth == 100000000000.0] = np.nan
        truth = resize(truth, depth.shape)
        if gma:
            depth_gma = depth_read(os.path.join(depth_gma_path, file))

        #depth = resize(depth, truth.shape)
        depth_initial = depth_read(os.path.join(initial_path, file))


        #depth_initial = resize(depth_initial, truth.shape)
        

        scale_factor.append(np.nanmean(truth)/np.nanmean(depth)) 
        scale_factor_initial.append(np.nanmean(truth)/np.nanmean(depth_initial)) 
        if gma:
            scale_factor_gma.append(np.nanmean(truth)/np.nanmean(depth_gma))
    scale_factor= np.nanmean(scale_factor)
    scale_factor_initial= np.nanmean(scale_factor_initial)
    if gma:
        scale_factor_gma= np.nanmean(scale_factor_gma)


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
pcs_acc=[]
i=1
for file in files: #["frame_0001.dpt"]:

    if i < start_index:
        i+=1
        continue

    #Get camera data:
    if len(os.listdir(src_cam_path))>0:
        frame_cam = file.split(".")[0]+".cam"
        I,E = cam_read(os.path.join(src_cam_path, frame_cam))
        intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
        extrinsic=E
    else:
        print("Extrinsics not available")
        intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5 , 217.5 )
        extrinsic=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


        
    if standart:
        depth = depth_read(os.path.join(depth_path, file))
    #depth = resize(depth, truth.shape)
    if init:
        depth_initial = depth_read(os.path.join(initial_path, file))
    if gtruth:
        truth = depth_read(os.path.join(truth_path, file))
        truth[truth == 100000000000.0] = np.nan
        if standart:
            truth = resize(truth, depth.shape)
        elif init:
            truth = resize(truth, depth_initial.shape)
        elif gma:
            truth = resize(truth, depth_gma.shape)

        

    #depth_initial = resize(depth_initial, truth.shape)


    if gma:
        dept_gma = depth_read(os.path.join(depth_gma_path, file))


    if (gtruth or norm) and standart:   
        
        depth_norm = depth * scale_factor 
        depth_norm_initial = depth_initial * scale_factor_initial

        if gma:
            depth_norm_gma = depth_gma * scale_factor_gma
        

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

    # colmapintrinsic:      w  h   fx     fy      cx   cy   #Scale c with
    #                    1024 436 1120.0 1120.0 511.5 217.5
    #                     384 160
    if gtruth:
        depth_img_t=o3d.geometry.Image(truth)
        pc_t=o3d.geometry.PointCloud.create_from_depth_image(depth_img_t, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

    if standart:
        if not_norm:
            depth_img_d=o3d.geometry.Image(depth)
            pc_d=o3d.geometry.PointCloud.create_from_depth_image(depth_img_d, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

        if norm:
            depth_img_d_n=o3d.geometry.Image(depth_norm)
            pc_d_n=o3d.geometry.PointCloud.create_from_depth_image(depth_img_d_n, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1) #TODO: import extrinsic,intrinsic

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

    #intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5, 217.5)
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    obj1 = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=20, create_uv_map=False)
    obj2= copy.deepcopy(obj1).translate((0, 40, -3))
    obj2.paint_uniform_color([1, 0, 0])
    objs=[obj2]#cf]#,obj1,obj2]

    pcs=[]
    if gtruth:
        pcs.append(pc_t)
    if standart:
        if not_norm:
            pcs.append(pc_d)
        if norm:
            pcs.append(pc_d_n)
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
    
    # Flip it, otherwise the pointcloud will be upside down
    for pc in pcs:
        pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if accumulate:
        pcs+=pcs_acc
    if render_obj:
        pcs+=objs
    print("\n"+str(file)+":")
    if gtruth:
        print("ground truth = green")
        pc_t.paint_uniform_color([0.5, 0.706, 0.5]) #green
    #o3d.visualization.draw_geometries([pc_t])
    if standart:
        if not_norm:
            print("depth = yellow")
            pc_d.paint_uniform_color([1, 0.706, 0]) #orange/yellow
        if norm:
            print("depth(norm) = red")
            pc_d_n.paint_uniform_color([1, 0, 0]) #red
    if init:
        if not_norm:
            print("depth initial = light blue")
            pc_i.paint_uniform_color([0, 0.651, 0.929]) #light blue
        if norm:
            print("depth initial(norm) = dark blue")
            pc_i_n.paint_uniform_color([0, 0, 1]) #dark blue
    if gma:
        if not_norm:
            print("depth gma = pink")
            pc_g.paint_uniform_color([0.9, 0.2, 0.84]) #pink
        if norm:
            print("depth gma(norm) = purple")
            pc_g_n.paint_uniform_color([0.5, 0.195, 0.66]) #purple
      
    
    #render = rendering.OffscreenRenderer(640, 480)

    #white = rendering.Material()
    #white.base_color = [1.0, 1.0, 1.0, 1.0]
    #white.shader = "defaultLit"

    #green = rendering.Material()
    #green.base_color = [0.0, 0.5, 0.0, 1.0]
    #green.shader = "defaultLit"

    #o3d.visualization.draw_geometries(pcs)
    #o3d.visualization.draw_geometries_with_animation_callback()
    #o3d.visualization.draw_geometries_with_custom_animation()
    #TODO:open3d.org/docs/release/tutorial/visualization/interactive_visualization.html
    
    #vis = o3d.visualization.VisualizerWithEditing()
    #vis.create_window(visible=True)
    #for pc in pcs:
    #    vis.add_geometry(pc)
    #    vis.update_geometry(pc)
    #vis.poll_events()
    #vis.update_renderer()
    #vis.capture_screen_image("test.jpg")
    #vis.destroy_window()

    renderer= o3d.visualization.rendering.OffscreenRenderer(1024, 436,headless=True)
    #extrinsic=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    renderer.scene.set_background([0.1, 0.2, 0.3, 1.0]) 
    renderer.setup_camera(intrinsic,extrinsic)
    #scene=o3d.visualization.rendering.Open3DScene(renderer)
    mat = o3d.visualization.rendering.Material()
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    mat.shader = 'defaultUnlit'
    for pc in pcs: #TODO pcs
        renderer.scene.add_geometry(str(i),pc,mat)
        i+=1
    img = renderer.render_to_image()
    print(img)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Preview window", img_cv)
    cv2.waitKey()

    o3d.io.write_image("test.jpg", img)

    if accumulate:
        pcs_acc+=pcs
    i+=1







