
#################64 x 64 uncondition###########################################################
MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
--cross_attention_shift True --dropout 0.1 
--video_attention_resolutions 2,4,8
--audio_attention_resolutions -1
--num_heads 4
--num_heads_upsample -1
--video_size 20,3,64,64 --audio_size 2,80000 --learn_sigma False --num_channels 128 --random_flip False
--num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True 
--use_scale_shift_norm True --num_workers 2 --video_fps 4 --audio_fps 16000"

# Modify --devices to your own GPU ID
TRAIN_FLAGS="--lr 5e-5 --batch_size 2 
--devices 0,1 --log_interval 100 --save_interval 5000 --use_db False --resume_checkpoint /data/stan_2024/20_2024_sony/svg-2024-spatial-track-starter-kit-master/models/pretrained_models/model330011.pt" 

#--schedule_sampler loss-second-moment
#--resume_checkpoint /data/stan_2024/20_2024_sony/svg-2024-spatial-track-starter-kit-master/models/pretrained_models/model330011.pt

DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --save_type mp4 --sample_fn ddpm"  #p_sample_loop  dpm_solver++

# Modify the following pathes to your own paths
DATA_DIR="/data/stan_2024/20_2024_sony/SVG2024-spatial-track-v0/video_dev"
OUTPUT_DIR="/data/stan_2024/20_2024_sony/SVG2024-spatial-track-v0/debug"
NUM_GPUS=2

# python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS 

mpiexec -n $NUM_GPUS  python3 py_scripts/multimodal_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $VIDEO_FLAGS $DIFFUSION_FLAGS 
