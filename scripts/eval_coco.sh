python3 main_pretrain.py -c ./configs/coco.yaml -rm test --version "coco_4" -ng 1 --backbone "video-swin-s" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_small_patch244_window877_kinetics400_1k.pth"
## when eval the coco_version can be ignored but necessary to be gievn