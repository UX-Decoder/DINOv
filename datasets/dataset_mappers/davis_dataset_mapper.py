# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import os

import cv2
import scipy.io
import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from detectron2.structures import BitMasks, Boxes, Instances

from ..shapes import build_shape_sampler
from detectron2.config import configurable


# __all__ = ["YTVOSDatasetMapper"]
__all__ = ["DAVISDatasetMapper"]


class VideoReader(object):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, image_dir, mask_dir, objects, min_size=None, max_size=None):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.use_all_mask = True
        self.vid_name = os.path.basename(image_dir)

        self.frames = sorted(os.listdir(self.image_dir))
        self.palette = Image.open(os.path.join(mask_dir, sorted(os.listdir(mask_dir))[0])).getpalette()
        self.first_gt_path = os.path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[0])

        self.object_ids = [int(x) for x in list(objects.keys())]
        self.object_id_to_start_frame = {key: os.path.join(mask_dir, "{}.png".format(objects[key]['frames'][0])) for key in objects.keys()}
        self.object_id_to_end_frame = {key: os.path.join(mask_dir, "{}.png".format(objects[key]['frames'][-1])) for key in objects.keys()}
        self.start_frames = self.object_id_to_start_frame.values()
        self.end_frames = self.object_id_to_end_frame.values()
        self.mappers = {idx:int(x) for idx,x in enumerate(self.object_ids)}

        t = []
        t.append(transforms.Resize(min_size, interpolation=Image.BICUBIC, max_size=max_size))
        self.transform = transforms.Compose(t)

    def __getitem__(self, idx):
        dataset_dict = {}
        frame = self.frames[idx]

        im_path = os.path.join(self.image_dir, frame)
        image = Image.open(im_path).convert('RGB')
        dataset_dict['width'] = image.size[0]
        dataset_dict['height'] = image.size[1]
        image = self.transform(image)
        image = torch.from_numpy(np.asarray(image).copy())
        image = image.permute(2,0,1)

        gt_path = os.path.join(self.mask_dir, '{}.png'.format(frame[:-4]))

        key_frames = torch.zeros(len(self.object_ids)).bool()
        end_frames = torch.zeros(len(self.object_ids)).bool()
        if os.path.exists(gt_path):
            mask = Image.open(gt_path).convert('P')
            mask = np.array(mask, dtype=np.uint8)

            object_masks = []
            for idx in self.object_ids:
                object_masks += [mask==idx]

            instances = Instances(image.shape[-2:])
            _,h,w = image.shape
            # sbd dataset only has one gt mask.
            masks = [cv2.resize(object_mask.astype(np.uint8), (w,h), interpolation=cv2.INTER_CUBIC) for object_mask in object_masks]
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks
            instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict['instances'] = instances
            dataset_dict['gt_masks_orisize'] = torch.stack([torch.from_numpy(object_mask) for object_mask in object_masks])

            if gt_path in self.start_frames:
                for index, obj_id in enumerate(self.object_ids):
                    if gt_path == self.object_id_to_start_frame[str(obj_id)]:
                        key_frames[index] = True

            if gt_path in self.end_frames:
                for index, obj_id in enumerate(self.object_ids):
                    if gt_path == self.object_id_to_end_frame[str(obj_id)]:
                        end_frames[index] = True


        dataset_dict['image'] = image
        dataset_dict['key_frame'] = key_frames
        dataset_dict['frame_id'] = frame.split('/')[-1].split('.')[0]
        dataset_dict['end_frame'] = end_frames
        return dataset_dict

    def __len__(self):
        return len(self.frames)

# This is specifically designed for the COCO dataset.
class DAVISDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        dataset_name='',
        min_size_test=800,
        max_size_test=1333,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = False
        self.dataset_name = 'davis'
        self.min_size_test = min_size_test
        self.max_size_test = 1333

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=''):
        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
            "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        return VideoReader(dataset_dict['file_name'], dataset_dict['mask_name'], dataset_dict['objects'], min_size=self.min_size_test, max_size=self.max_size_test)