# Consistent Video Depth Estimation

Code adpated from: https://github.com/facebookresearch/consistent_depth 

Run on Sintel dataset:

w camera info:
Type=clean && Name=<scenario_name>
python transform_sintel.py $Name

flownet: python main.py --path ./data/FN/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint FlowNet2
gma: python main.py --path ./data/GMA/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint GMA
flownet+DensePose: python main.py --path ./data/FN_DP/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint FlowNet2 --DensePose
gma+DensePose: python main.py --path ./data/GMA_DP/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint GMA --DensePose

additional options: --batch_size 1 

w/o camera info:
Type=clean && Name=<scenario_name>
python transform_sintel_wo_pose.py $Name

flownet: python main.py --video_file ./data/FN_wo_pose/$Name/$Type/video.mp4 --path ./data/FN_wo_pose/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint FlowNet2
gma: python main.py --video_file ./data/GMA_wo_pose/$Name/$Type/video.mp4 --path ./data/GMA_wo_pose/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint GMA
flownet+DensePose: python main.py --video_file ./data/FN_DP_wo_pose/$Name/$Type/video.mp4  --path ./data/FN_DP_wo_pose/$Name/$Type/--initialize_pose --make_video --flow_checkpoint FlowNet2 --DensePose
gma+DensePose: python main.py --video_file ./data/GMA_DP_wo_pose/$Name/$Type/video.mp4 --path ./data/GMA_DP_wo_pose/$Name/$Type/ --initialize_pose --make_video --flow_checkpoint GMA --DensePose

additional options: --batch_size 1 




default file structure:

MPI-Sintel-complete
MPI-Sintel-depth-training-20150305
consistent_video_depth_estimation
-data
--FN
--GMA
--...



