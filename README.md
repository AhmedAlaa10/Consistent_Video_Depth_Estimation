# Consistent Video Depth Estimation
Code adpated from: https://github.com/facebookresearch/consistent_depth 

Setup:

Follow the instructions on: https://github.com/facebookresearch/consistent_depth 
pip install open3d
pip install cv2


Run on Sintel dataset:

w camera info:

Type=clean && Name=<scenario_name> && Method=<method>   <method>:FN/GMA/FN_DP/GMA_DP
python transform_sintel.py $Name $Method

flownet: python main.py --path ./data/$Method/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint FlowNet2
gma: python main.py --path ./data/$Method/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint GMA
flownet+DensePose: python main.py --path ./data/$Method/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint FlowNet2 --DensePose
gma+DensePose: python main.py --path ./data/$Method/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint GMA --DensePose

additional options: --batch_size 1 



w/o camera info:

Type=clean && Name=<scenario_name> && Method=<method>   <method>:FN_wo_pose/GMA_wo_pose/FN_DP_wo_pose/GMA_DP_wo_pose/custom
python transform_sintel_wo_pose.py $Name $Method

flownet: python main.py --video_file ./data/$Method/$Name/$Type/video.mp4 --path ./data/$Method/$Name/$Type/ --make_video --flow_checkpoint FlowNet2
gma: python main.py --video_file ./data/$Method/$Name/$Type/video.mp4 --path ./data/$Method/$Name/$Type/ --make_video --flow_checkpoint GMA
flownet+DensePose: python main.py --video_file ./data/$Method/$Name/$Type/video.mp4  --path ./data/$Method/$Name/$Type/ --make_video --flow_checkpoint FlowNet2 --DensePose
gma+DensePose: python main.py --video_file ./data/$Method/$Name/$Type/video.mp4 --path ./data/$Method/$Name/$Type/ --make_video --flow_checkpoint GMA --DensePose

additional options: --batch_size 2 | --camera_params "1120.0, 511.5, 217.5" --camera_model "SIMPLE_PINHOLE"




default file structure:

MPI-Sintel-complete
MPI-Sintel-depth-training-20150305
consistent_video_depth_estimation
-data
--FN
--GMA
--...



