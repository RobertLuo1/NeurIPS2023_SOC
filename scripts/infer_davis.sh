python infer_davis.py -c ./configs/davis.yaml -rm test --version "davis_base_joint" -ng 8 --backbone "video-swin-b" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_base_patch244_window877_kinetics400_22k.pth" \
-ckpt "/mnt/data_16TB/lzy23/SOC/base_joint/new_joint_base.tar"

sleep 30s
python eval_davis.py --results_path "/mnt/data_16TB/lzy23/SOC/runs/davis/davis_base_joint/anno_0"
python eval_davis.py --results_path "/mnt/data_16TB/lzy23/SOC/runs/davis/davis_base_joint/anno_1"
python eval_davis.py --results_path "/mnt/data_16TB/lzy23/SOC/runs/davis/davis_base_joint/anno_2"
python eval_davis.py --results_path "/mnt/data_16TB/lzy23/SOC/runs/davis/davis_base_joint/anno_3"