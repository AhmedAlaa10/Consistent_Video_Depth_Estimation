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
import copy

import open3d as o3d #pip install open3d
import open3d.visualization.rendering as rendering
import cv2 #pip install cv2

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'



fps = 7.5 #TODO
name="shaman_3"
batch_size=1 #TODO
render_obj=True

preview=False
interactive=False
vis_depth=False
vis_obj=False
vis_mask=False


use_gtruth=False #DEFAULT: False
use_initial=False
type= "FN"  #FN / GMA / ...
norm=False
obj_path=None #frames for obj  DEFAULT:None

accumulate=False


start_index=0 #default=0

if len(sys.argv) > 1:
    name = str(sys.argv[1])

sintel_depth_path =  "../MPI-Sintel-depth-training-20150305" #TODO

#file_path="/home/umbra/Documents/MPI-Sintel-depth-training-20150305/training/depth/bamboo_2/frame_0001.dpt"
depth_dataset_path="../MPI-Sintel-depth-training-20150305/"

src_path="./data/"+type+"/"+name+"/clean/"

img_path=os.path.join(src_path,"color_full")

output_path=os.path.join(src_path,"render_frames")
os.makedirs(output_path, exist_ok=True)

mask_path=os.path.join(src_path,"render_masks")
os.makedirs(mask_path, exist_ok=True)


#Dont change after here
#----------------------------------------------------------------------------------------------------------------------------------------------------
src_cam_path = os.path.join(sintel_depth_path, "training", "camdata_left", name)

if not os.path.isdir(src_path):
    print("depth path ("+ src_path +") empty")
    exit()


if use_gtruth:
    depth_path=os.path.join(depth_dataset_path,"training/depth/"+name+"/")
elif use_initial:
    depth_path=os.path.join(src_path,"depth_mc/exact_depth/")
else:
    depth_path=os.path.join(src_path,"R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS"+str(batch_size)+"_Oadam/exact_depth/")
truth_path=os.path.join(depth_dataset_path,"training/depth/"+name+"/")

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



files = os.listdir(depth_path)
files.sort()

if norm:
    #Calculate statistical scale factor:
    scale_factor=[]

    for file in files: #["frame_0001.dpt"]:
        truth = depth_read(os.path.join(truth_path, file))
        depth = depth_read(os.path.join(depth_path, file))
        truth[truth == 100000000000.0] = np.nan
        truth = resize(truth, depth.shape)   

        scale_factor.append(np.nanmean(truth)/np.nanmean(depth))  

    scale_factor= np.nanmean(scale_factor)
 


#Compute distance:
pcs_acc=[]
ml1 =[]
mse =[]
ml1_norm=[]
mse_norm=[]
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
        
   
    depth = depth_read(os.path.join(depth_path, file))
    



    if norm:           
        depth= depth * scale_factor 
        distance = (truth - depth)
        ml1.append((np.abs(distance)).mean(axis=None))
        mse.append((np.square(distance)).mean(axis=None))
        

    # colmapintrinsic:      w  h   fx     fy      cx   cy   #Scale c with
    #                    1024 436 1120.0 1120.0 511.5 217.5
    #                     384 160
    depth_img_d=o3d.geometry.Image(depth)
    pc_d=o3d.geometry.PointCloud.create_from_depth_image(depth_img_d, intrinsic, extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

    #intrinsic=o3d.camera.PinholeCameraIntrinsic(1024, 436, 1120.0, 1120.0, 511.5, 217.5)
    #TODO: create objs:
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    obj1 = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=20, create_uv_map=False)
    obj2= copy.deepcopy(obj1).translate((0, -40, 3))
    
    objs=[obj2]#cf]#,obj1,obj2]

    pcs=[pc_d]   
    if accumulate:
        pcs+=pcs_acc
    if render_obj:
        pcs+=objs
    print("\n"+str(file)+":")

    
    

    if interactive:
        o3d.visualization.gui.Application.instance.initialize()
        w = o3d.visualization.O3DVisualizer("03DVisualizer",1024, 436)
        w.reset_camera_to_default()
        w.setup_camera(intrinsic,extrinsic)

        obj2.paint_uniform_color([1, 0, 0]) #Red #TODO: set obj colors
        pc_d.paint_uniform_color([0.5, 0.706, 0.5]) #green
        w.scene.set_background(np.array([1.,1.,1.,1.])) #white

        #w.scene.set_background(np.array([1., 1., 1., 1.])) #Black
        o3d.visualization.gui.Application.instance.add_window(w)
        w.show_axes = True
        w.size_to_fit() #Full screen
        mat = o3d.visualization.rendering.Material()
        #mat.base_color = [1.0, 1.0, 1.0, 1.0]
        mat.shader = 'defaultUnlit'
        for pc in pcs: 
                print("added pc")
                w.add_geometry(str(i),pc,mat)
                i+=1
        o3d.visualization.gui.Application.instance.run()
        #o3d.visualization.gui.Application.instance.quit()
        #w.export_current_image("test.png")
    else:
        renderer= o3d.visualization.rendering.OffscreenRenderer(1024, 436,headless=False)
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
            renderer.scene.add_geometry(str(i),pc,mat)
            i+=1
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
            renderer.scene.add_geometry(str(i),pc,mat)
            i+=1
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
        
        split1 = file.split("_")
        split2 = split1[1].split(".")
        index = int(split2[0])-1
        index =str(index).zfill(6)
        file_new = str(split1[0]+"_"+index+".png") 
        frame_path= os.path.join(img_path,file_new)
        img=cv2.imread(frame_path) 
        #print(img.shape)
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


    if accumulate:
        pcs_acc+=pcs
    i+=1

    #break #TODO: delete

#TODO: Create video
video_path=os.path.join(src_path, "rendered_video.mp4")
video_from_frames(output_path,video_path)
print("Video with "+fps+" fps created at "+video_path)







