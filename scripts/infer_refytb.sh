python3 infer_refytb.py -c ./configs/refer_youtube_vos.yaml -rm test --version "joint_base_test" -ng 1 --backbone "video-swin-b" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_base_patch244_window877_kinetics400_1k.pth" \
-ckpt "/mnt/data_16TB/lzy23/SOC/base_joint/new_joint_base.tar"
