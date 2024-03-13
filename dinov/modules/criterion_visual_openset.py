# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------

"""
DINOv criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from timm.loss import SoftTargetCrossEntropy
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, _max_by_axis
from ..utils import box_ops
from ..utils.misc import sigmoid_focal_loss, sigmoid_ce_loss_jit, dice_loss_jit, calculate_uncertainty


class SetCriterionVisualOpenSet(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, top_x_layers, losses,
                 num_points, oversample_ratio, importance_sample_ratio, grounding_weight, dn="no",dn_losses=[], panoptic_on=False, semantic_ce_loss=False):
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
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.top_x_layers = top_x_layers
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
        self.grounding_weight = grounding_weight

    def loss_labels_masked(self, outputs, targets, indices, num_boxes, log=True, layer_id=None, extra=None):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        if indices is None or len(targets) == 0:
            loss_ce = outputs['pred_logits'].sum() * 0.0
            losses = {"loss_mask_cls_0": loss_ce}
            return losses

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits[idx], target_classes_onehot[idx], num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_mask_cls_0': loss_ce}

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, layer_id=None, log=True, key='gt_whole_classes', extra=None):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if self.prediction_switch is None or 'whole' not in self.prediction_switch.keys():
            if 'labels' in targets[0].keys():
                key = 'labels'
        else:
            if not self.prediction_switch['whole']:
                return {"fake_no_loss_mask_cls_0": 0.0}
            elif key not in targets[0].keys() or targets[0].get('fake', False):
                # FIXME only consider batchsize=1 case
                # assert len(targets) == 1
                return {"loss_mask_cls_0": 0.0 * outputs['pred_logits'].sum()}
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[key][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {"loss_mask_cls_0": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_id=None, extra=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        if indices is None or len(targets) == 0 or targets[0].get('fake', False):
            loss = outputs['pred_boxes'].sum() * 0.0
            # loss = outputs['pred_boxes'].sum()
            losses = {"loss_bbox_0": loss, "loss_giou_0": loss}
            return losses

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_0'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_0'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, layer_id=None, extra=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        if indices is None or len(targets) == 0 or targets[0].get('fake', False):
            loss = outputs['pred_masks'].sum() * 0.0
            losses = {"loss_mask_bce_0": loss, "loss_mask_dice_0": loss}
            return losses

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
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def prep_for_dn(self,mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar,pad_size=mask_dict['scalar'],mask_dict['pad_size']
        assert pad_size % scalar==0
        single_pad=pad_size//scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes,num_tgt,single_pad,scalar

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

    def get_loss(self, loss, outputs, targets, indices, num_masks=None, layer_id=None, extra=None):
        loss_map = {
            'labels': self.loss_labels,
            'dn_labels': self.loss_labels_masked,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, layer_id=layer_id, extra=extra)

    def forward(self, outputs, targets_all, mask_dict=None, extra=None, task='seg'):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # TODO: use different matching and loss weight when only detection
        prediction_switch = extra
        self.prediction_switch = prediction_switch

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        match_cost = ["cls", "box", "mask"]
        if task == 'det' or task == 'seg_from_teacher':
            match_cost = ["cls", "box"]
        targets = targets_all['generic']
        if mask_dict is not None and 'num_class' in targets[0].keys():
            self.num_classes = targets[0]['num_class']
        # Retrieve the matching between the outputs of the last layer and the targets
        if self.dn is not "no" and mask_dict is not None and not targets[0].get('fake', False):
            output_known_lbs_bboxes,num_tgt,single_pad,scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                exc_idx.append((output_idx, tgt_idx))

        indices = self.matcher(outputs_without_aux, targets, match_cost, extra=extra) if not targets[0].get('fake', False) else None
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if task == 'det' and loss == 'masks':
                continue   # not compute mask loss for detection data only
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, layer_id=0, extra=extra))

        if self.dn != "no" and mask_dict is not None and not targets[0].get('fake', False):
            l_dict={}
            for loss in self.dn_losses:
                if task == 'det' and loss == 'masks':
                    continue  # not compute mask loss for detection data only
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx, num_masks*scalar, layer_id=0))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != "no":
            l_dict = dict()
            l_dict['loss_bbox_0_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_0_dn'] = torch.as_tensor(0.).to('cuda')
            if prediction_switch['whole'] and 'dn_labels' in self.dn_losses:
                l_dict['loss_mask_cls_0_dn'] = torch.as_tensor(0.).to('cuda')
            if prediction_switch['part']:
                l_dict['loss_mask_part_cls_0_dn'] = torch.as_tensor(0.).to('cuda')
            if task != 'det' and 'masks' in self.dn_losses:
                l_dict['loss_mask_bce_0_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_mask_dice_0_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, match_cost) if not targets[0].get('fake', False) else None
                for loss in self.losses:
                    if task == 'det' and loss == 'masks':
                        continue  # not compute mask loss for detection data only
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, layer_id=(i+1), extra=extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i>=start:
                    if self.dn != "no" and mask_dict is not None and not targets[0].get('fake', False):
                        out_=output_known_lbs_bboxes['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            if task == 'det' and loss == 'masks':
                                continue  # not compute mask loss for detection data only
                            l_dict.update(
                                self.get_loss(loss, out_, targets, exc_idx, num_masks * scalar, layer_id=(i+1), extra=extra))
                        l_dict = {k.replace('_0', f"_{i+1}_dn"): v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != "no":
                        l_dict = dict()
                        l_dict[f'loss_bbox_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_giou_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        if prediction_switch['whole'] and 'dn_labels' in self.dn_losses:
                            l_dict[f'loss_mask_cls_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        if prediction_switch['part']:
                            l_dict[f'loss_mask_part_cls_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        if self.dn == "seg" and task != 'det' and 'masks' in self.dn_losses:
                            l_dict[f'loss_mask_bce_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                            l_dict[f'loss_mask_dice_{i+1}_dn'] = torch.as_tensor(0.).to('cuda')
                        losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets, match_cost)
            full_set = ['labels', 'masks', 'boxes']
            for loss in list(set(self.losses) and set(full_set)):
                if task == 'det' and loss == 'masks':
                    continue  # not compute mask loss for detection data only
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_masks, layer_id=-1, extra=extra)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)
        # from pudb.remote import set_trace
        # set_trace(term_size=(80, 24))
        # print("losses ", losses, torch.as_tensor(0.).to('cuda').device)
        # print("losseslosses.keys() ", losses.keys(), torch.as_tensor(0.).to('cuda').device)

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
