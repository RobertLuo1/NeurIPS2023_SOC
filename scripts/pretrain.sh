python3 main_pretrain.py -c ./configs/coco.yaml -rm train --version "coco_pretrain_base" -ng 8 --epochs 30 \
--lr_drop 15 20 -bs 8 --backbone "video-swin-b" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_base_patch244_window877_kinetics400_22k.pth"