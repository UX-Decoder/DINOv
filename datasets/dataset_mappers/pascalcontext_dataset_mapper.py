# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy

import scipy.io
import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import transforms as T

__all__ = ["PascalContextSegDatasetMapper"]


# This is specifically designed for the COCO dataset.
class PascalContextSegDatasetMapper_ori:
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
            min_size_test=None,
            max_size_test=None,
            mean=None,
            std=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.pixel_mean = torch.tensor(mean)[:, None, None]
        self.pixel_std = torch.tensor(std)[:, None, None]

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC))
        # HACK for ViT evaluation
        # t.append(transforms.Resize([self.min_size_test, self.min_size_test], interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = {
            "is_train": is_train,
            "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
            "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
            "mean": cfg['INPUT']['PIXEL_MEAN'],
            "std": cfg['INPUT']['PIXEL_STD'],
        }
        return ret

    def read_semseg(self, file_name):
        if '.png' in file_name:
            semseg = np.asarray(Image.open(file_name))
        elif '.mat' in file_name:
            semseg = scipy.io.loadmat(file_name)['LabelMap']
        return semseg

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict['file_name']
        semseg_name = dataset_dict['sem_seg_file_name']
        image = Image.open(file_name).convert('RGB')

        dataset_dict['width'] = image.size[0]
        dataset_dict['height'] = image.size[1]

        if self.is_train == False:
            image = self.transform(image)
            image = torch.from_numpy(np.asarray(image).copy())
            image = image.permute(2, 0, 1)

        semseg = self.read_semseg(semseg_name)
        semseg = torch.from_numpy(semseg.astype(np.int32))

        dataset_dict['image'] = image
        dataset_dict['semseg'] = semseg
        return dataset_dict

class PascalContextSegDatasetMapper:
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
            augmentations=None,
            min_size_test=None,
            max_size_test=None,
            mean=None,
            std=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.pixel_mean = torch.tensor(mean)[:, None, None]
        self.pixel_std = torch.tensor(std)[:, None, None]

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC))
        # HACK for ViT evaluation
        # t.append(transforms.Resize([self.min_size_test, self.min_size_test], interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)
        self.augmentations = T.AugmentationList(augmentations)
        self.ignore_label = 0

    @classmethod
    def from_config(cls, cfg, is_train=True):
        augs = utils.build_augmentation(cfg, is_train)
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
            "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
            "mean": cfg['INPUT']['PIXEL_MEAN'],
            "std": cfg['INPUT']['PIXEL_STD'],
        }
        return ret

    def read_semseg(self, file_name):
        if '.png' in file_name:
            semseg = np.asarray(Image.open(file_name))
        elif '.mat' in file_name:
            semseg = scipy.io.loadmat(file_name)['LabelMap']
        return semseg

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format='RGB')
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        semseg_name = None
        if "sem_seg_file_name" in dataset_dict:
            semseg_name = dataset_dict.pop("sem_seg_file_name")
            if semseg_name.split('.')[-1]!='mat':
                sem_seg_gt = utils.read_image(semseg_name, "L").squeeze(2)
            else:
                sem_seg_gt = self.read_semseg(semseg_name)
                sem_seg_gt = sem_seg_gt.astype(np.uint8)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        # if self.size_divisibility > 0:
        #     image_size = (image.shape[-2], image.shape[-1])
        #     padding_size = [
        #         0,
        #         self.size_divisibility - image_size[1],
        #         0,
        #         self.size_divisibility - image_size[0],
        #     ]
        #     image = F.pad(image, padding_size, value=128).contiguous()
        #     if sem_seg_gt is not None:
        #         sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            # sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            ###
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            # TODO: this is a hack for datasets with backgorund as 0, which is meaningless
            classes = classes[classes != self.ignore_label] - 1
            # print("semseg_name, classes ", semseg_name, classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"] = instances

        #######



        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # file_name = dataset_dict['file_name']
        # semseg_name = dataset_dict['sem_seg_file_name']
        # image = Image.open(file_name).convert('RGB')
        #
        # dataset_dict['width'] = image.size[0]
        # dataset_dict['height'] = image.size[1]
        #
        # if self.is_train == False:
        #     image = self.transform(image)
        #     image = torch.from_numpy(np.asarray(image).copy())
        #     image = image.permute(2, 0, 1)
        #
        # semseg = self.read_semseg(semseg_name)
        # semseg = torch.from_numpy(semseg.astype(np.int32))
        #
        # dataset_dict['image'] = image
        # dataset_dict['semseg'] = semseg
        return dataset_dict