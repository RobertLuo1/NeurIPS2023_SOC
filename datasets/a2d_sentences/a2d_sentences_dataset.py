import json
import torch
import numpy as np
from torchvision.io import read_video
import h5py
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision.transforms.functional as F
import pandas
from os import path
from glob import glob
from tqdm import tqdm
import datasets.transforms as T
# import datasets.refer_youtube_vos.transforms.transform_video as T 
from pycocotools.mask import encode, area
from misc import nested_tensor_from_videos_list
import random
from datasets.a2d_sentences.create_gt_in_coco_format import create_a2d_sentences_ground_truth_test_annotations


def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id


class A2DSentencesDataset(Dataset):
    """
    A Torch dataset for A2D-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """
    def __init__(self, subset_type: str = 'train', dataset_path: str = '/mnt/data_16TB/lzy23/rvosdata/a2d_sentences', window_size=8,
                 dataset_coco_gt_format_path=None, distributed=False, **kwargs):
        super(A2DSentencesDataset, self).__init__()
        assert subset_type in ['train', 'test'], 'error, unsupported dataset subset type. supported: train, test'
        self.subset_type = subset_type
        self.mask_annotations_dir = path.join(dataset_path, 'text_annotations/a2d_annotation_with_instances')
        self.videos_dir = path.join(dataset_path, 'Release/clips320H')
        # if subset_type == 'pred':
        #     self.get_text_annotations(dataset_path, 'test', distributed)
        # else:
        self.text_annotations = self.get_text_annotations(dataset_path, subset_type, distributed)
        self.window_size = window_size
        self.transforms = A2dSentencesTransforms(subset_type, **kwargs)
        self.collator = Collator()
        # create ground-truth test annotations for the evaluation process if necessary:
        if subset_type == 'test' and not path.exists(dataset_coco_gt_format_path):
            if (distributed and dist.get_rank() == 0) or not distributed:
                create_a2d_sentences_ground_truth_test_annotations()
            if distributed:
                dist.barrier()

    @staticmethod
    def get_text_annotations(root_path, subset, distributed):
        saved_annotations_file_path = f'/mnt/data_16TB/lzy23/rvosdata/a2d_sentences/a2d_sentences_single_frame_{subset}_annotations.json'
        if path.exists(saved_annotations_file_path):
            with open(saved_annotations_file_path, 'r') as f:
                text_annotations_by_frame = [tuple(a) for a in json.load(f)]
                return text_annotations_by_frame
        elif (distributed and dist.get_rank() == 0) or not distributed: #let zero card does this
            print(f'building a2d sentences {subset} text annotations...')
            # without 'header == None' pandas will ignore the first sample...
            a2d_data_info = pandas.read_csv(path.join(root_path, 'Release/videoset.csv'), header=None)
            assert len(a2d_data_info) == 3782, f'error: a2d videoset.csv file is missing one or more samples.'
            # 'vid', 'label', 'start_time', 'end_time', 'height', 'width', 'total_frames', 'annotated_frames', 'subset'
            a2d_data_info.columns = ['vid', '', '', '', '', '', '', '', 'subset']
            with open(path.join(root_path, 'text_annotations/a2d_missed_videos.txt'), 'r') as f:
                unused_videos = f.read().splitlines()
            subsets = {'train': 0, 'test': 1} #0 for training and 1 for testing
            # filter unused videos and videos which do not belong to our train/test subset:
            used_videos = a2d_data_info[
                ~a2d_data_info.vid.isin(unused_videos) & (a2d_data_info.subset == subsets[subset])]
            used_videos_ids = list(used_videos['vid']) #the used videos
            text_annotations = pandas.read_csv(path.join(root_path, 'text_annotations/a2d_annotation.txt'))
            assert len(text_annotations) == 6655, 'error: a2d_annotations.txt is missing one or more samples.'
            # filter the text annotations based on the used videos:
            used_text_annotations = text_annotations[text_annotations.video_id.isin(used_videos_ids)]
            # remove a single dataset annotation mistake in video: T6bNPuKV-wY
            used_text_annotations = used_text_annotations[used_text_annotations['instance_id'] != '1 (copy)'] #wrong annotations
            # convert data-frame to list of tuples:
            used_text_annotations = list(used_text_annotations.to_records(index=False))
            text_annotations_by_frame = []
            mask_annotations_dir = path.join(root_path, 'text_annotations/a2d_annotation_with_instances')
            for video_id, instance_id, text_query in tqdm(used_text_annotations):
                frame_annot_paths = sorted(glob(path.join(mask_annotations_dir, video_id, '*.h5')))
                instance_id = int(instance_id)
                for p in frame_annot_paths:
                    f = h5py.File(p)
                    instances = list(f['instance'])
                    if instance_id in instances:
                        # in case this instance does not appear in this frame it has no ground-truth mask, and thus this
                        # frame-instance pair is ignored in evaluation, same as SOTA method: CMPC-V. check out:
                        # https://github.com/spyflying/CMPC-Refseg/blob/094639b8bf00cc169ea7b49cdf9c87fdfc70d963/CMPC_video/build_A2D_batches.py#L98
                        frame_idx = int(p.split('/')[-1].split('.')[0])
                        text_query = text_query.lower()  # lower the text query prior to augmentation & tokenization
                        text_annotations_by_frame.append((text_query, video_id, frame_idx, instance_id))
            with open(saved_annotations_file_path, 'w') as f:
                json.dump(text_annotations_by_frame, f, indent=2)
        if distributed:
            dist.barrier()
            with open(saved_annotations_file_path, 'r') as f:
                text_annotations_by_frame = [tuple(a) for a in json.load(f)]
        return text_annotations_by_frame

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
    
    def __getitem__(self, idx):
        text_query, video_id, frame_idx, instance_id = self.text_annotations[idx]

        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        video_frames, _, _ = read_video(path.join(self.videos_dir, f'{video_id}.mp4'), pts_unit='sec')  # (T, H, W, C)
        # get a window of window_size frames with frame frame_idx in the middle.
        # note that the original a2d dataset is 1 indexed, so we have to subtract 1 from frame_idx
        start_idx, end_idx = frame_idx - 1 - self.window_size // 2, frame_idx - 1 + (self.window_size + 1) // 2

        # # read the source window frames:
        # video_frames, _, _ = read_video(path.join(self.videos_dir, f'{video_id}.mp4'), pts_unit='sec')  # (T, H, W, C)
        # # get a window of window_size frames with frame frame_idx in the middle.
        # # note that the original a2d dataset is 1 indexed, so we have to subtract 1 from frame_idx
        # # start_idx, end_idx = frame_idx - 1 - self.window_size // 2, frame_idx - 1 + (self.window_size + 1) // 2
        # vid_len = len(video_frames)
        # frame_id = frame_idx - 1

        # if self.subset_type == 'train':
        #     # get a window of window_size frames with frame frame_id in the middle.
        #     window_size = self.window_size
        #     # random sparse sample
        #     sample_indx = [frame_id]
        #     # local sample
        #     sample_id_before = random.randint(1, 3)
        #     sample_id_after = random.randint(1, 3)
        #     local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
        #     sample_indx.extend(local_indx)

        #     # global sampling
        #     if window_size > 3:
        #         all_inds = list(range(vid_len))
        #         global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
        #         global_n = window_size - len(sample_indx)
        #         if len(global_inds) > global_n:
        #             select_id = random.sample(range(len(global_inds)), global_n)
        #             for s_id in select_id:
        #                 sample_indx.append(global_inds[s_id])
        #         elif vid_len >=global_n:  # sample long range global frames
        #             select_id = random.sample(range(vid_len), global_n)
        #             for s_id in select_id:
        #                 sample_indx.append(all_inds[s_id])
        #         else:
        #             select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
        #             for s_id in select_id:                                                                   
        #                 sample_indx.append(all_inds[s_id])
        #     sample_indx.sort()
        #     # find the valid frame index in sampled frame list, there is only one valid frame
        #     valid_indices = sample_indx.index(frame_id)

        # elif self.subset_type == 'test':
        #     start_idx, end_idx = frame_id - self.window_size // 2, frame_id + (self.window_size + 1) // 2
        #     sample_indx = []
        #     for i in range(start_idx, end_idx):
        #         i = min(max(i, 0), len(video_frames)-1)  # pad out of range indices with edge frames
        #         sample_indx.append(i)
        #     sample_indx.sort()
        #     # find the valid frame index in sampled frame list, there is only one valid frame
        #     valid_indices = sample_indx.index(frame_id)
        

        # extract the window source frames:
        source_frames, boxes = [], []
        for i in range(start_idx, end_idx):
            i = min(max(i, 0), len(video_frames)-1)  # pad out of range indices with edge frames
            source_frames.append(F.to_pil_image(video_frames[i].permute(2, 0, 1))) #(C H W)

        # read the instance mask:
        frame_annot_path = path.join(self.mask_annotations_dir, video_id, f'{frame_idx:05d}.h5')
        f = h5py.File(frame_annot_path, 'r')
        instances = list(f['instance'])
        instance_idx = instances.index(instance_id)  # existence was already validated during init

        instance_masks = np.array(f['reMask'])
        if len(instances) == 1:
            instance_masks = instance_masks[np.newaxis, ...] #[1, H, W]
        instance_masks = torch.tensor(instance_masks).transpose(1, 2) #[H W]
        mask_rles = [encode(mask) for mask in instance_masks.numpy()]
        mask_areas = area(mask_rles).astype(np.float)
        f.close()

        for mask in instance_masks:
            mask = mask.numpy()
            if (mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
            else:
                box = torch.tensor([0, 0, 0, 0]).to(torch.float)
            boxes.append(box)
        label = torch.tensor(0, dtype=torch.long)

        h, w = instance_masks.shape[-2:]
        boxes = torch.stack(boxes, dim=0) #[o 4]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h) 
        # create the target dict for the center frame:
        target = {'masks': instance_masks[instance_idx].unsqueeze(0),
                  "boxes": boxes[instance_idx].unsqueeze(0),
                  'orig_size': instance_masks.shape[-2:],  # original frame shape without any augmentations
                  # size with augmentations, will be changed inside transforms if necessary
                  'size': instance_masks.shape[-2:],
                  "is_ref_inst_visible": instance_masks[instance_idx].any(),
                  'referred_instance_idx': torch.tensor(0),  # idx in 'masks' of the text referred instance
                  'area': torch.tensor(mask_areas),
                  'iscrowd': torch.zeros(len(instance_masks)),  # for compatibility with DETR COCO transforms
                  'image_id': get_image_id(video_id, frame_idx, instance_id),
                  "caption": text_query,
                  "labels": torch.tensor(0, dtype=torch.long)
                }

        # create dummy targets for adjacent frames:
        targets = self.window_size * [None]
        center_frame_idx = self.window_size // 2
        targets[center_frame_idx] = target
        source_frames, targets, text_query = self.transforms(source_frames, targets, text_query)
        return source_frames, targets, text_query

    def __len__(self):
        return len(self.text_annotations)


class A2dSentencesTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 random_color, train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        self.random_color = subset_type == 'train' and random_color
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # no more scales for now due to GPU memory constraints. might be changed later
        self.photometricDistort = T.PhotometricDistort()
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            elif subset_type == 'test':
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets, text_query):
        valid_frame = [idx for idx, t in enumerate(targets) if t is not None][0]
        if self.h_flip_augmentation and torch.randn(1) > 0.5:
            h, w = targets[valid_frame]['size']
            source_frames = [F.hflip(f) for f in source_frames]
            targets[valid_frame]['masks'] = F.hflip(targets[valid_frame]['masks'])
            boxes = targets[valid_frame]['boxes'] 
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            targets[valid_frame]["boxes"] = boxes
            # Note - is it possible for both 'right' and 'left' to appear together in the same query. hence this fix:
            text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
        if self.random_color and torch.randn(1) > 0.5:
            source_frames, targets = self.photometricDistort(source_frames, targets)
        source_frames, targets = list(zip(*[self.size_transforms(f, t) for f, t in zip(source_frames, targets)]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets, text_query

class A2dSentencesTransforms_2:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        scales = [288, 320, 352, 392, 416, 448, 480, 512]
        #scales = [320]
        # max_size = 640
        max_size = 640
        if subset_type == "train":
            self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    # T.PhotometricDistort(),
                    #T.RandomResize(scales, max_size=max_size),
                    # T.RandomSelect(
                    # T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    # T.Check(),
                        # ]),
                        # T.Compose([
                        #     T.RandomResize([300, 400, 500]),
                        #     T.RandomSizeCrop(284, 400),
                        #     T.RandomResize(scales, max_size=max_size),
                        #     T.Check(),
                        # ])
                    # ),
                    normalize,
                ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([360], max_size=640),
                normalize,   
            ])
    
    def __call__(self, source_frames, targets, text_query):
        # targets = [targets[windows_size // 2]]
        valid_frame = [idx for idx, t in enumerate(targets) if t is not None][0]
        source_frames, targets = self.transforms(source_frames, targets)
        source_frames = torch.stack(source_frames)
        text_query = targets[valid_frame]['caption']
        return source_frames, targets, text_query 

class Collator:
    def __call__(self, batch):
        samples, targets, text_queries = list(zip(*batch))
        samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W] and mask mask for useful area
        # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
        targets = list(zip(*targets))
        batch_dict = {
            'samples': samples,
            'targets': targets,
            'text_queries': text_queries
        }
        return batch_dict

if __name__ == '__main__':
    dataset = A2DSentencesDataset()

