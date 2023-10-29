export CUDA_VISIBLE_DEVICES=5
python demo_video.py -c ./configs/refer_youtube_vos.yaml -rm test --backbone "video-swin-b" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_base_patch244_window877_kinetics400_22k.pth" \
-ckpt "/mnt/data_16TB/lzy23/SOC/base_joint/new_joint_base.tar" \
--video_dir "/mnt/data_16TB/lzy23/rvosdata/a2d_sentences/Release/clips320H/0gZz8hESBEs.mp4"