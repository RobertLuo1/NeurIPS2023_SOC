import torch
from tqdm import tqdm
import os
from utils import flatten_temporal_batch_dims
import torch.nn.functional as F
import torchvision.transforms as transforms 
import numpy as np
from PIL import Image
import cv2 as cv2

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
colors = [[212, 255, 127], [193,182,255], [106,106,255], [255, 206, 135]]

def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample

@torch.no_grad()
def predict(model, data_loader_val, device, postprocessor, output_dir):
    model.eval()
    for batch_dict in tqdm(data_loader_val):
        predictions = []
        samples = batch_dict['samples'].to(device)
        targets = to_device(batch_dict['targets'], device)
        text_queries = batch_dict['text_queries']

        # keep only the valid targets (targets of frames which are annotated):
        valid_indices = torch.tensor([i for i, t in enumerate(targets) if None not in t]).to(device)
        targets = [targets[i] for i in valid_indices.tolist()]
        outputs = model(samples, valid_indices, text_queries)
        outputs.pop('aux_outputs', None)

        outputs, targets = flatten_temporal_batch_dims(outputs, targets)
        processed_outputs = postprocessor(outputs, resized_padded_sample_size=samples.tensors.shape[-2:],
                                                resized_sample_sizes=[t['size'] for t in targets],
                                                orig_sample_sizes=[t['orig_size'] for t in targets])
        image_ids = [t['image_id'] for t in targets]
        folder = image_ids[0]
        folder = folder.split('_')[1]
        if os.path.exists(os.path.join(output_dir, folder)):
            save_folder = os.path.join(output_dir, folder)
        else:
            os.mkdir(os.path.join(output_dir, folder))
            save_folder = os.path.join(output_dir, folder)
        for p, image_id in zip(processed_outputs, image_ids):
            value, index = p['scores'].max(dim=0)
            masks = p['masks'][index]
            predictions.append(
                {
                    'image_id': image_id,
                    'segmentation': masks,
                    'score': value.item()
                }
            )
        
        images = torch.index_select(samples.tensors, 0, valid_indices)[0] #[b, C, H, W]
        images = F.interpolate(images, size=targets[0]['orig_size']) #[b c H, W]
        
        #from tensor to image
        image = images[0]
        image = image.permute(1, 2, 0) #[H W C]
        image = (image*torch.tensor(std, device=device)+torch.tensor(mean, device=device))
        image = image.permute(2, 0, 1) #[C H W]
        image = transforms.ToPILImage()(image)
        # image.save(os.path.join(save_folder, 'ori.jpg'))
        for idx, pred in enumerate(predictions):
            output_name = pred['image_id'] + '.jpg'
            output_path = os.path.join(save_folder, output_name)
            # color = torch.tensor(colors[idx], device=device).float()
            mask = pred['segmentation'][0].cpu().numpy() #[H, W]
            color = np.full(shape=(mask.shape[0], mask.shape[1], 3), fill_value=0, dtype=np.uint8)
            img = np.array(image)
            # img_mask = image.copy()
            fill_color = colors[idx]
            color[mask.astype(bool), :] = np.array(fill_color, dtype=np.uint8)
            img = cv2.addWeighted(img, 1, color, 0.5, 0)
            # image[pred['segmentation'][0].bool(), :] = image[pred['segmentation'][0].bool(), :] * (1 - opacity) + color * opacity
            # image = image.int()
        # image = image.permute(2, 0, 1) #[C H W]
        # image = transforms.ToPILImage()(image)
        img = Image.fromarray(img).convert('RGB')
        # draw = ImageDraw.Draw(img)
        # draw.text((0,0), text_queries, fill=(255, 0, 0), font=ImageFont)
        img.save(output_path)
            
            
            


        

    