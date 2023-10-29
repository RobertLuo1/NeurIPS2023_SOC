python3 main.py -c ./configs/refer_youtube_vos.yaml -rm train -ng 8 -epochs 20 \
-pw "/mnt/data_16TB/lzy23/pretrained/pretrained_coco/coco_1/best_pth.tar" --version "finetuneytb_base" \
--lr_drop 10 -bs 1 -ws 8 --backbone "video-swin-t" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"

### ng means num_gpu -pw means the path of the pretrained weights lr_drop means the scheduler