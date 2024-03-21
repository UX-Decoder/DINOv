# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
from utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image
from detectron2.data import MetadataCatalog
import os
import cv2

metadata = MetadataCatalog.get('coco_2017_train_panoptic')


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def task_openset(model,generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                   generic_vp5, generic_vp6, generic_vp7, generic_vp8, image_tgt=None, text_size=640,hole_scale=100,island_scale=100):
    in_context_examples = [generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                   generic_vp5, generic_vp6, generic_vp7, generic_vp8]
    in_context_examples = [x for x in in_context_examples if x is not None]
    t = []
    t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
    def prepare_image(image_ori):
        width = image_ori.size[0]
        height = image_ori.size[1]
        image_ori = np.asarray(image_ori)        
        images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
        return images, height, width
    transform1 = transforms.Compose(t)
    image_ori_tgt = transform1(image_tgt)
    images_tgt, height_tgt, width_tgt = prepare_image(image_ori_tgt)
    data_tgt = {"image": images_tgt, "height": height_tgt, "width": width_tgt}
    batched_inputs = []
    batched_inputs_tgt = [data_tgt]
    multi_scale_features2, mask_features2, _, _ = model.model.get_encoder_feature(batched_inputs_tgt)
    input_query_label_content_all = []
    point_coords = torch.ones(1, 4).cuda().float()
    point_coords[:, :2] = 0.
    input_query_bbox_content_init = inverse_sigmoid(point_coords[None])
    for image in in_context_examples:
        image_ori = transform1(image['image'])
        mask_ori = transform1(image['mask'])
        images, height, width = prepare_image(image_ori)
        
        data = {"image": images, "height": height, "width": width}
        data['seg_image'] = data_tgt

        mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
        mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)

        data['targets'] = [dict()]
        data['targets'][0]['rand_shape']=mask_ori
        data['targets'][0]['pb']=torch.tensor([1.])    # FIXME 0 or 1

        frame = data
        rand_shape = mask_ori
        frame['targets'][0]['rand_shape'] = rand_shape
        
        batched_inputs.append(frame)
        
        multi_scale_features, _, padded_h, padded_w = model.model.get_encoder_feature([frame])
        input_query_label_content, input_query_bbox_content, attn_mask_content = model.model. \
            get_visual_prompt_content_feature(multi_scale_features, frame['targets'][0]['rand_shape'], padded_h, padded_w)
        input_query_label_content_all.append(input_query_label_content)
        
    # prompt to tgt image
    input_query_label_content_current = torch.stack(input_query_label_content_all).mean(0)
    masks, ious, ori_masks, scores_per_image_openset = model.model.evaluate_demo_content_openset_multi_with_content_features(
        batched_inputs_tgt, mask_features2, multi_scale_features2, input_query_label_content_current,
        input_query_bbox_content_init, attn_mask_content, padded_h, padded_w)
    if len(ious.shape)>1:
        ious=ious[0]
    ids=torch.argsort(scores_per_image_openset,descending=True)
    areas=[]
    image_ori = image_ori_tgt
    new_pred_mask = []
    new_pred_class_score = []
    for i in ids:
        new_pred_class_score.append(scores_per_image_openset[i])
        new_pred_mask.append(masks[i])
    pred_masks_poses = new_pred_mask
    ious = new_pred_class_score
    visual = Visualizer(image_ori, metadata=metadata)
    for i,(pred_masks_pos,iou, _, _) in enumerate(zip(pred_masks_poses,ious, pred_masks_poses, pred_masks_poses)):
        iou=round(float(iou),2)
        texts=f'{iou}'
        mask=(pred_masks_pos>0.0).cpu().numpy()
        area=mask.sum()
        areas.append(area)
        # uncomment for additional postprocessing
        # mask,_=remove_small_regions(mask,int(hole_scale),mode="holes")
        # mask,_=remove_small_regions(mask,int(island_scale),mode="islands")
        mask=(mask).astype(np.float)
        color=[0.,0.,1.0]
        color=[0.502, 0.0, 0.502]
        demo = visual.draw_binary_mask(mask, text='', alpha=0.7, edge_color=color)
    res = demo.get_image()

    torch.cuda.empty_cache()

    return res

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True