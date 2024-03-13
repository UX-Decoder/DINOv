import sys
import random

import cv2
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib import distance_transform

from .scribble import Scribble
from dinov.utils import configurable


class SimpleClickSampler(nn.Module):
    @configurable
    def __init__(self, mask_mode='point', sample_negtive=False, is_train=True, dilation=None, dilation_kernel=None):
        super().__init__()
        self.mask_mode = mask_mode
        self.sample_negtive = sample_negtive
        self.is_train = is_train
        self.dilation = dilation
        self.register_buffer("dilation_kernel", dilation_kernel)

    @classmethod
    def from_config(cls, cfg, is_train=True, mode=None):
        mask_mode = mode
        sample_negtive = cfg['STROKE_SAMPLER']['EVAL']['NEGATIVE']

        dilation = cfg['STROKE_SAMPLER']['DILATION']
        dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.cuda.current_device())

        # Build augmentation
        return {
            "mask_mode": mask_mode,
            "sample_negtive": sample_negtive,
            "is_train": is_train,
            "dilation": dilation,
            "dilation_kernel": dilation_kernel,
        }

    def forward_scribble(self, instances, pred_masks=None, prev_masks=None):
        gt_masks_batch = instances.gt_masks
        _,h,w = gt_masks_batch.shape

        rand_shapes = []
        for i in range(len(gt_masks_batch)):
            gt_masks = gt_masks_batch[i:i+1]
            assert len(gt_masks) == 1 # it only supports a single image, with a single candidate mask.
            # pred_masks is after padding

            # We only consider positive points
            pred_masks = torch.zeros(gt_masks.shape).bool() if pred_masks is None else pred_masks[:,:h,:w]
            prev_masks = torch.zeros(gt_masks.shape).bool() if prev_masks is None else prev_masks

            fp = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks)
            next_mask = torch.zeros(gt_masks.shape).bool()

            mask_dt = torch.from_numpy(cv2.distanceTransform(fp[0].numpy().astype(np.uint8), cv2.DIST_L2, 0)[None,:])
            max_value = mask_dt.max()
            next_mask[(mask_dt==max_value).nonzero()[0:1].t().tolist()] = True

            points = next_mask[0].nonzero().flip(dims=[-1])
            next_mask = Scribble.draw_by_points(points, gt_masks, h, w)
            rand_shapes += [(prev_masks | next_mask)]

        types = ['scribble' for i in range(len(gt_masks_batch))]
        return {'gt_masks': instances.gt_masks, 'rand_shape': rand_shapes, 'types': types, 'sampler': self}

    def forward(self, instances, *args, **kwargs):
        if self.mask_mode == 'Point':
            return self.forward_point(instances, *args, **kwargs)
        elif self.mask_mode == 'Circle':
            assert False, "Circle not support best path."
        elif self.mask_mode == 'Scribble':
            assert False, "Scribble not support best path."
        elif self.mask_mode == 'Polygon':
            assert False, "Polygon not support best path."
