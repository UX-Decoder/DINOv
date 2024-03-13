# ------------------------------------------------------------------------
# Copyright (c) IDEA, Inc. and its affiliates.
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils import box_ops
from ..utils.misc import sigmoid_focal_loss_mean, iou_score_loss
from ..utils.misc import sigmoid_focal_loss_list as sigmoid_focal_loss, \
    sigmoid_ce_loss_list_jit as sigmoid_ce_loss_jit, \
    dice_loss_list_jit as dice_loss_jit, \
    calculate_uncertainty


class SetCriterionReferOne(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, content_matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, dn="no", dn_losses=[], panoptic_on=False,
                 semantic_ce_loss=False, num_mask_tokens=3, iou_loss=True, nms_threshold=0.9):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_part = -1
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn = dn
        self.dn_losses = dn_losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss
        self.num_mask_tokens = num_mask_tokens
        self.index = None
        self.iou_loss = iou_loss
        self.prediction_switch = None
        self.index_switch = {'part': torch.arange(0, self.num_mask_tokens - 1).cuda(),
                             'whole': torch.arange(self.num_mask_tokens - 1, self.num_mask_tokens).cuda(),
                             'all': torch.arange(0, self.num_mask_tokens).cuda(), }
        print("iou_loss is ", iou_loss)

        # content matcher
        self.content_matcher = content_matcher
        self.nms_threshold = nms_threshold

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        if 'boxes' not in targets[0].keys() or 'masks' not in targets[0].keys():
            # FIXME only consider batchsize=1 case
            assert len(targets) == 1
            return {"loss_bbox_0": 0.0 * outputs['pred_boxes'].sum(),
                    "loss_giou_0": 0.0 * outputs['pred_boxes'].sum(), }
        assert self.index is not None
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        loss_bbox = loss_bbox.sum(1)
        losses["loss_bbox_0"] = torch.gather(loss_bbox.view(-1, self.num_mask_tokens), 1, self.index.unsqueeze(1)).sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_giou_0"] = torch.gather(loss_giou.view(-1, self.num_mask_tokens), 1, self.index.unsqueeze(1)).sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        if 'masks' not in targets[0].keys():
            # FIXME only consider batchsize=1 case
            assert len(targets) == 1
            return {"loss_mask_bce_0": 0.0 * outputs['pred_masks'].sum(),
                    "loss_mask_dice_0": 0.0 * outputs['pred_masks'].sum(),
                    "iou_score_loss_0": 0.0 * outputs['pred_ious'].sum(),
                    }

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }
        mask_loss = losses["loss_mask_bce_0"] + losses["loss_mask_dice_0"]
        mask_loss, index = mask_loss.view(-1, self.num_mask_tokens).min(1)
        bs = outputs["pred_masks"].shape[0]
        assert index is not None
        self.index = index
        losses["loss_mask_bce_0"] = torch.gather(losses["loss_mask_bce_0"].view(-1, self.num_mask_tokens), 1,
                                                 index.unsqueeze(1)).sum() / num_masks
        dice_loss = torch.gather(losses["loss_mask_dice_0"].view(-1, self.num_mask_tokens), 1, index.unsqueeze(1))
        losses["loss_mask_dice_0"] = dice_loss.sum() / num_masks

        target_iou = 1 - dice_loss
        src_ious = outputs["pred_ious"]
        iou_idx = ([src_idx[0].view(bs, -1)[:, :int(index.shape[0] / bs)].flatten(),
                    src_idx[1].view(bs, -1)[:, :int(index.shape[0] / bs)].flatten()])
        src_ious = src_ious[iou_idx].view(-1, self.num_mask_tokens)
        src_ious = torch.gather(src_ious, 1, index.unsqueeze(1))

        if self.iou_loss:
            losses['iou_score_loss_0'] = iou_score_loss(src_ious, target_iou).sum() / num_masks

        del src_masks
        del target_masks
        return losses

    def loss_match_score(self, outputs, targets, indices, num_masks):
        if 'masks' not in targets[0].keys():
            return {"loss_match_score_0": outputs['pred_boxes'].sum() * 0.0}
        assert outputs["pred_masks"].shape[0] == 1
        src_masks = outputs["pred_masks"][0]
        src_boxes = outputs["pred_boxes"][0]
        src_ious = outputs["pred_ious"][0].flatten(0, 1)
        item_indice = nms(box_ops.box_cxcywh_to_xyxy(src_boxes), src_ious, self.nms_threshold)  # FIXME iou threshold
        new_output = {}
        new_output["pred_masks"] = src_masks[item_indice][None]
        new_output["pred_boxes"] = src_boxes[item_indice][None]
        scores = src_ious[item_indice][None]
        new_targets = [{'masks': t['content_masks']} for t in targets]
        indices = self.content_matcher(new_output, new_targets, cost=['mask'])
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_classes = torch.stack([ind[0] for ind in indices]).cuda()
        src_logits = outputs["pred_match_score"]
        src_logits = torch.index_select(src_logits, -1, item_indice)
        src_logits = src_logits[tgt_idx][None]
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {"loss_match_score_0": loss_ce}
        return losses

    def loss_match_score_maskloss(self, outputs, targets, indices, num_masks):
        if 'masks' not in targets[0].keys():
            return {"loss_match_score_0": outputs['pred_boxes'].sum() * 0.0,
                    "loss_mask_bce_0": outputs['pred_boxes'].sum() * 0.0,
                    "loss_mask_dice_0": outputs['pred_boxes'].sum() * 0.0,
                    }
        assert outputs["pred_masks"].shape[0] == 1
        src_masks = outputs["pred_masks"][0]
        src_boxes = outputs["pred_boxes"][0]
        src_ious = outputs["pred_ious"][0].flatten(0, 1)
        item_indice = nms(box_ops.box_cxcywh_to_xyxy(src_boxes), src_ious, self.nms_threshold)  # FIXME iou threshold
        new_output = {}
        new_output["pred_masks"] = src_masks[item_indice][None]
        new_output["pred_boxes"] = src_boxes[item_indice][None]
        new_targets = [{'masks': t['content_masks']} for t in targets]
        indices = self.content_matcher(new_output, new_targets, cost=['mask'])

        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_classes = torch.stack([ind[0] for ind in indices]).cuda()
        src_logits = outputs["pred_match_score"]
        src_logits = torch.index_select(src_logits, -1, item_indice)
        src_logits = src_logits[tgt_idx][None]


        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        ############
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.type(self.empty_weight.dtype),
                lambda logits: calculate_uncertainty(logits.type(self.empty_weight.dtype)),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks).sum()/num_masks,
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks).sum()/num_masks,
            "loss_match_score_0": loss_ce
        }

        return losses

    def loss_match_score_sigmoid(self, outputs, targets, indices, num_masks, layer_id=None, extra=None):
        if 'masks' not in targets[0].keys():
            return {"loss_match_score_0": outputs['pred_boxes'].sum() * 0.0}
        assert outputs["pred_masks"].shape[0] == 1
        src_masks = outputs["pred_masks"][0]
        src_boxes = outputs["pred_boxes"][0]
        src_ious = outputs["pred_ious"][0].flatten(0, 1)
        item_indice = nms(box_ops.box_cxcywh_to_xyxy(src_boxes), src_ious, self.nms_threshold)  # FIXME iou threshold
        new_output = {}
        new_output["pred_boxes"] = box_ops.box_xyxy_to_cxcywh(src_boxes[item_indice])[None]
        new_output["pred_masks"] = src_masks[item_indice][None]
        scores = src_ious[item_indice][None]
        new_targets = [{'masks': t['content_masks']} for t in targets]
        indices = self.content_matcher(new_output, new_targets, cost=['mask'])
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_classes = torch.stack([ind[0] for ind in indices]).cuda()
        src_logits = outputs["pred_match_score"]
        src_logits = torch.index_select(src_logits, -1, item_indice)
        src_logits = src_logits[tgt_idx][None]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss_mean(src_logits, target_classes_onehot, num_masks, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]



        losses = {"loss_match_score_0": loss_ce}
        return losses

    def prep_for_dn(self, mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar, pad_size = mask_dict['scalar'], mask_dict['pad_size']
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes, num_tgt, single_pad, scalar

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
            'match_content': self.loss_match_score,
            'match_content_sigmoid': self.loss_match_score_sigmoid,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None, task='sam', extra={}):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        targets_ = targets['position']  # FIXME ensure position and content targets are the same
        if 'masks' in targets['content'][0].keys():
            for i in range(len(targets_)):
                targets_[i]['content_masks'] = targets['content'][i]['masks']
        targets = targets_
        assert len(targets)==1, "now only support one image training for interactive segmentation"
        prediction_switch = extra
        self.prediction_switch = prediction_switch
        if mask_dict is None:
            sudo_loss = sum([outputs[key].sum() for key in outputs.keys() if key != 'aux_outputs' and outputs[key] is not None]) * 0.0
            losses = dict()
            l_dict = dict()
            l_dict['loss_bbox_0'] = sudo_loss
            l_dict['loss_giou_0'] = sudo_loss
            if prediction_switch['whole']:
                l_dict['loss_mask_cls_0'] = sudo_loss
            if prediction_switch['part']:
                l_dict['loss_mask_part_cls_0'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_mask_bce_0'] = sudo_loss
            l_dict['loss_mask_dice_0'] = sudo_loss
            l_dict['iou_score_loss_0'] = sudo_loss
            losses.update(l_dict)
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    l_dict = dict()
                    l_dict[f'loss_bbox_{i + 1}'] = sudo_loss
                    l_dict[f'loss_giou_{i + 1}'] = sudo_loss
                    if prediction_switch['whole']:
                        l_dict[f'loss_mask_cls_{i + 1}'] = sudo_loss
                    if prediction_switch['part']:
                        l_dict[f'loss_mask_part_cls_{i + 1}'] = torch.as_tensor(0.).to('cuda')
                    l_dict[f'loss_mask_bce_{i + 1}'] = sudo_loss
                    l_dict[f'loss_mask_dice_{i + 1}'] = sudo_loss
                    l_dict[f'iou_score_loss_{i + 1}'] = sudo_loss
                    losses.update(l_dict)
            return losses
        exc_idx = []
        for i in range(len(targets)):
            if len(targets[i]['boxes']) > 0:
                tgt_idx = torch.arange(0, len(targets[i]['boxes'])).long().cuda().repeat_interleave(
                    self.num_mask_tokens)
                src_idx = torch.arange(0, outputs['pred_masks'].shape[1]).long().cuda()
            else:
                tgt_idx = src_idx = torch.tensor([]).long().cuda()
            exc_idx.append((src_idx, tgt_idx))
        indices = exc_idx
        num_masks = sum(len(t["boxes"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs['pred_masks'].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        self.index = None
        if 'masks' in self.losses:
            assert 'masks' in self.losses[0], "must calculate mask loss first for match"
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        self.index = None

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k.replace('_0', f"_{i + 1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)
                self.index = None
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)