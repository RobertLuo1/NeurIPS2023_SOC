import ruamel.yaml
import torchvision.transforms as T
# from tools.colormap import colormap
from datasets.transforms import RandomResize
from models import build_model
import os
from PIL import Image
import torch
import misc as utils
import numpy as np
import torch.nn.functional as F
import random
import argparse
from torchvision.io import read_video
import torchvision.transforms.functional as Func
import shutil

size_transform = RandomResize(sizes=[360], max_size=640)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
color = np.array([0, 0, 255]).astype('uint8')
def vis_add_mask(img, mask, color):
    source_img = np.asarray(img).copy()
    origin_img = np.asarray(img).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    source_img = Image.fromarray(source_img)
    mask = Image.fromarray(mask)
    return origin_img, source_img, mask


def main(config):
    print(config.backbone_pretrained)
    model, _, _ = build_model(config) 
    device = config.device
    model.to(device)

    if config.checkpoint_path is not None:
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        state_dict = checkpoint["model_state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        print("pleas specify the checkpoint")

    model.eval()
    video_dir = config.video_dir
    ver_dir = video_dir.split("/")[-1]
    ver_dir = ver_dir.split(".")[0]
    save_dir = os.path.join('/mnt/data_16TB/lzy23/SOC/video_demo', ver_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    output_dir = os.path.join(save_dir, "SOC", "visual")
    source_dir = os.path.join(save_dir, "SOC", "source")
    mask_dir = os.path.join(save_dir, "SOC", "mask")
    
    exp = "a man falls down suddenly"
    with open(os.path.join(save_dir, "expression.txt"), 'w') as f:
        f.write(exp + "\n")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)


    video_frames, _, _ = read_video(video_dir, pts_unit='sec')  # (T, H, W, C)
    source_frames= []
    imgs = []
    print("length",len(video_frames))
    num_frame = [12,62,92,132,142,152]
    for i in range(0,len(video_frames),5):
        source_frame = Func.to_pil_image(video_frames[i].permute(2, 0, 1))
        source_frames.append(source_frame) #(C H W)
    for frame in source_frames:
        origin_w, origin_h = frame.size
        img, _ = size_transform(frame)
        imgs.append(transform(img)) # list[img]
    
    frame_length = len(imgs)

    imgs = torch.stack(imgs, dim=0) # [video_len, 3, H, W]
    samples = utils.nested_tensor_from_videos_list([imgs]).to(config.device)
    img_h, img_w = imgs.shape[-2:]
    size = torch.as_tensor([int(img_h), int(img_w)]).to(config.device)
    targets = [[{"size": size}] for _ in range(frame_length)]
    valid_indices = None


    with torch.no_grad():
        outputs = model(samples, valid_indices, [exp], targets)
    
    pred_logits = outputs["pred_cls"][:, 0, ...] # [t, q, k]
    pred_masks = outputs["pred_masks"][:, 0, ...]   # [t, q, h, w] 

    pred_scores = pred_logits.sigmoid() # [t, q, k]
    pred_scores = pred_scores.mean(0)   # [q, k]
    max_scores, _ = pred_scores.max(-1) # [q,]
    _, max_ind = max_scores.max(-1)     # [1,]

    max_inds = max_ind.repeat(frame_length)
    pred_masks = pred_masks[range(frame_length), max_inds, ...] # [t, h, w]
    pred_masks = pred_masks.unsqueeze(0)

    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
    pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).detach().cpu().numpy() 

    color = [255, 144, 30]

    for t, img in enumerate(source_frames):
        origin_img, source_img, mask = vis_add_mask(img, pred_masks[t], color)
        # save_postfix = img_path.replace(".jpg", ".png")
        origin_img.save(os.path.join(output_dir, f'{t}.png'))
        source_img.save(os.path.join(source_dir, f'{t}.png'))
        mask.save(os.path.join(mask_dir, f'{t}.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DEMO script')
    parser.add_argument('--config_path', '-c',
                        default='./configs/refer_youtube_vos.yaml',                        help='path to configuration file')
    parser.add_argument('--running_mode', '-rm', choices=['train', 'test', 'pred', 'resume_train'],
                        default='test',
                        help="mode to run, either 'train' or 'eval'")
    parser.add_argument("--backbone", type=str, required=False,
                        help="the backbone name")
    parser.add_argument("--backbone_pretrained_path", "-bpp", type=str, required=False,
                        help="the backbone_pretrained_path")
    parser.add_argument('--checkpoint_path', '-ckpt', type=str, default='',
                            help='the finetune refytbs checkpoint_path')
    parser.add_argument("--video_dir", type=str, required=False)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    with open(args.config_path) as f:
        config = ruamel.yaml.safe_load(f)
    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args)}
    config = argparse.Namespace(**config)
    
    main(config)