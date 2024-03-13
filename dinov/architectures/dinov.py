# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops.boxes import nms

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
import random
from datasets.shapes.sampler import build_shape_sampler, ShapeSampler
import copy
from kornia.contrib import distance_transform
from .registry import register_model
from ..utils import configurable, box_ops, get_class_names, get_iou, box_postprocess, build_point_grid
from ..backbone import build_backbone, Backbone
from ..body import build_openseed_head
from ..body.decoder.utils.utils import from_divisablity, getIdx
from ..modules import sem_seg_postprocess, HungarianMatcher, HungarianMatcherMany2Many, \
    SetCriterionVisualOpenSet, SetCriterionReferOne, SetCriterionReferMany
from ..language import build_language_encoder
from utils.dist import get_world_size, all_gather


class DINOv(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion_switch: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        train_dataset_name: str,
        background: bool,
        coco_on=True,
        coco_mask_on=True,
        o365_on=True,
        coco_track=True,
        sam_on: bool = True,
        pascal_part_on=True,
        regenerate_point: bool = False,
        num_mask_tokens: int = 3,
        max_num_instance_content: int = 3,
        max_num_instance: int = 10,
        max_train_example: int = 4,
        cross_gpu_batch: bool = True,
        shape_sampler: ShapeSampler,
        use_shape_sampler: True,
        openset_use_shape_sampler: False,
        many2many: True,
        vis: False,
        refer_image: False,
        freeze_all: False,
        freeze_backbone_enc: False,
        freeze_backbone_enc_decoder: False,
        nms_thersh: float = 0.9,
        point_per_side: int = 20,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion_switch = criterion_switch
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        self.train_class_names = dict()
        self.train_dataset_name = train_dataset_name
        self.coco_mask_on = coco_mask_on
        self.task_switch = {'coco': coco_on, 'o365': o365_on, 'pascal_part': pascal_part_on, 'sam': sam_on, 'coco_track': coco_track}
        print("self.task_switch ", self.task_switch)
        # HACK for only two datasets for seg and det
        if coco_on:
            task = 'coco'
            if not coco_mask_on:
                task = 'det'
            self.train_class_names[task] = get_class_names('coco_2017_train_panoptic', background=background)
            self.train_class_names['ade'] = get_class_names('ade20k_panoptic_train', background=background)
            self.train_class_names[task] = [a.replace("-merged", "").replace("-other", "").replace("-stuff", "") for a
                                             in self.train_class_names[task]]
            train_class_names = []
            for name in self.train_class_names[task]:
                names = name.split('-')
                if len(names) > 1:
                    assert len(names) == 2
                    train_class_names.append(names[1] + ' ' + names[0])
                else:
                    train_class_names.append(name)
            self.train_class_names[task] = train_class_names

        if o365_on and len(train_dataset_name)>1:
            self.train_class_names['det'] = get_class_names(train_dataset_name[1], background=background)
            self.train_class_names['det'] = [a.lower().split('/') for a in self.train_class_names['det']]

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.max_num_instance = max_num_instance
        self.default_num_instance_openset = max_num_instance_content
        self.num_mask_tokens = num_mask_tokens
        self.regenerate_point = regenerate_point

        self.shape_sampler = shape_sampler
        self.use_shape_sampler = use_shape_sampler
        self.openset_use_shape_sampler = openset_use_shape_sampler
        self.temp = 0
        self.max_train_example = max_train_example
        self.cross_gpu_batch = cross_gpu_batch

        # for interactive
        self.max_num_instance_content = max_num_instance_content
        self.many2many = many2many
        self.vis = vis
        self.refer_image = refer_image
        self.nms_thersh = nms_thersh
        self.stability_score_offset = 1.0
        self.stability_score_thresh = 0.92
        self.point_per_side = point_per_side

        # freeze some parameters
        to_freeze_dict = ['label_enc', 'pb_embedding']
        if freeze_all:
            for (name, param) in self.named_parameters():
                param.requires_grad = False
            print("!!!!!!!!freeze_all!!!!!!!!, except ", to_freeze_dict)
            for (name, param) in self.named_parameters():
                for f_name in to_freeze_dict:
                    if f_name in name:
                        param.requires_grad = True
                        print(name)
                        break
                else:
                    pass
        # freeze backbone and enc parameters
        to_freeze_dict = ['sem_seg_head.predictor']
        # to_freeze_dict = ['sem_seg_head.predictor']
        if freeze_backbone_enc:
            for (name, param) in self.named_parameters():
                param.requires_grad = False
            print("!!!!!!!!freeze_backbone_enc!!!!!!!!, except ", to_freeze_dict)
            for (name, param) in self.named_parameters():
                for f_name in to_freeze_dict:
                    if f_name in name:
                        param.requires_grad = True
                        print(name)
                        break
                else:
                    pass
        # freeze backbone and enc parameters
        to_freeze_dict = ['sem_seg_head.predictor']
        not_freeze_dict = ['sem_seg_head.predictor.decoder']
        if freeze_backbone_enc_decoder:
            for (name, param) in self.named_parameters():
                param.requires_grad = False
            print("!!!!!!!!freeze_backbone_enc_decoder!!!!!!!!, except ", to_freeze_dict)
            for (name, param) in self.named_parameters():
                for f_name in to_freeze_dict:
                    if f_name in name and not_freeze_dict[0] not in name:
                        param.requires_grad = True
                        print(name)
                        break
                else:
                    pass
        for (name, param) in self.named_parameters():
            if param.requires_grad:
                print(name)
        print("use openset_use_shape_sampler ", openset_use_shape_sampler)

    @classmethod
    def from_config(cls, cfg):
        """
        :param cfg: input cfg from yaml file
        :return: model parameters for the __init__ function
        """

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        iou_weight = dec_cfg['IOU_WEIGHT']
        class_weight = dec_cfg['CLASS_WEIGHT']
        match_class_weight = dec_cfg.get('MATCH_CLASS_WEIGHT', class_weight)
        cost_class_weight = dec_cfg['COST_CLASS_WEIGHT']
        cost_dice_weight = dec_cfg['COST_DICE_WEIGHT']
        dice_weight = dec_cfg['DICE_WEIGHT']
        cost_mask_weight = dec_cfg['COST_MASK_WEIGHT']
        mask_weight = dec_cfg['MASK_WEIGHT']
        cost_box_weight = dec_cfg['COST_BOX_WEIGHT']
        box_weight = dec_cfg['BOX_WEIGHT']
        cost_giou_weight = dec_cfg['COST_GIOU_WEIGHT']
        giou_weight = dec_cfg['GIOU_WEIGHT']

        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
        )
        # many2many interactive matcher
        m2m_matcher = HungarianMatcherMany2Many(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3)
        )
        # content matcher for visual prompt
        content_matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
        )

        # losses and weight_dict
        weight_dict = {"loss_mask_cls_0": class_weight}
        weight_dict.update({"loss_match_score_0": match_class_weight})
        weight_dict.update({"loss_mask_part_cls_0": class_weight})
        weight_dict.update({"loss_mask_bce_0": mask_weight, "loss_mask_dice_0": dice_weight})
        weight_dict.update({"loss_bbox_0": box_weight, "loss_giou_0": giou_weight})
        weight_dict.update({"iou_score_loss_0": iou_weight})
        # two stage is the query selection scheme (from mask dino)
        if dec_cfg['TWO_STAGE']:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training (from mask dino)
        dn = dec_cfg['DN']
        # TODO hack for dn label loss
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["dn_labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["masks", "boxes"]  # FIXME
        else:
            dn_losses=[]

        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k.replace('_0', '_{}'.format(i+1)): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if dec_cfg['BOX']:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]
        loss_match = copy.deepcopy(losses)
        loss_match = loss_match[1:]
        if dec_cfg.get('softmax', True):
            loss_match.append('match_content')  # FIXME
        else:
            loss_match.append('match_content_sigmoid')  # FIXME

        # update task switch
        task_switch = {}
        task_switch.update({'bbox': dec_cfg.get('DETECTION', True), 'mask': dec_cfg.get('MASK', True)})
        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'box': dec_cfg.get('TOP_DETECTION_LAYERS', 10)}

        # building criterion
        criterion = SetCriterionVisualOpenSet(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=None,
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
        )

        criterion_m2m_content_match = SetCriterionReferMany(
            enc_cfg['NUM_CLASSES'],
            matcher=m2m_matcher,
            content_matcher=content_matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=loss_match,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
            num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3),
            nms_threshold=dec_cfg.get('NMS_THRESHOLD', 0.9),
        )

        criterion_1o1_content_match = SetCriterionReferOne(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            content_matcher=content_matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=loss_match,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
            num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3),
            nms_threshold=dec_cfg.get('NMS_THRESHOLD', 0.9),
        )

        criterion_switch = {'open_set': criterion, '1o1_cm': criterion_1o1_content_match,
                            'm2m_cm': criterion_m2m_content_match}

        # build model
        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)
        sem_seg_head = build_openseed_head(cfg, backbone.output_shape(), lang_encoder, extra=extra)
        shape_sampler = build_shape_sampler(cfg, )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['COCO']['TEST']['DETECTIONS_PER_IMAGE'],
            "data_loader": None,
            "focus_on_box": cfg['MODEL']['DECODER']['TEST']['TEST_FOUCUS_ON_BOX'],
            "transform_eval": cfg['MODEL']['DECODER']['TEST']['PANO_TRANSFORM_EVAL'],
            "pano_temp": cfg['MODEL']['DECODER']['TEST']['PANO_TEMPERATURE'],
            "semantic_ce_loss": cfg['MODEL']['DECODER']['TEST']['SEMANTIC_ON'] and cfg['MODEL']['DECODER']['SEMANTIC_CE_LOSS'] and not cfg['MODEL']['DECODER']['TEST']['PANOPTIC_ON'],
            "train_dataset_name": cfg['DATASETS']['TRAIN'], # HACK for only two training set
            "background": cfg['MODEL'].get('BACKGROUND', True),
            "coco_on": dec_cfg.get('COCO', True),
            "coco_mask_on": dec_cfg.get('COCO_MASK', True),
            "pascal_part_on": dec_cfg.get('PASCAL', True),
            "o365_on": dec_cfg.get('O365', True),
            "coco_track": dec_cfg.get('COCO_TRACK', False),
            "regenerate_point": dec_cfg.get('RE_POINT', False),
            "num_mask_tokens": dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3),
            "max_num_instance": dec_cfg.get('MAX_NUM_INSTANCE', 100),
            "max_num_instance_content": dec_cfg.get('MAX_NUM_INSTANCE_CONTENT', 10),
            'shape_sampler': shape_sampler,
            'use_shape_sampler': dec_cfg.get('USE_SHAPE_SAMPLER', True),
            'openset_use_shape_sampler': dec_cfg.get('OPENSET_USE_SHAPE_SAMPLER', False),
            'max_train_example': dec_cfg.get('MAX_TRAIN_EXAMPLE', 4),
            'cross_gpu_batch': dec_cfg.get('CROSS_GPU_BATCH', True),
            'criterion_switch': criterion_switch,
            'many2many': dec_cfg.get('MANY2MANY', True),
            'vis': dec_cfg.get('VIS', False),
            'nms_thersh': dec_cfg.get('NMS_THRESHOLD', False),
            'refer_image': dec_cfg.get('REFER_IMAGE', False),
            'point_per_side': dec_cfg.get('point_per_side', 20),
            "sam_on": dec_cfg.get('SAM', True),
            "freeze_all": dec_cfg.get('freeze_all', False),
            "freeze_backbone_enc": dec_cfg.get('freeze_backbone_enc', False),
            "freeze_backbone_enc_decoder": dec_cfg.get('freeze_backbone_enc_decoder', False),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, inference_task='seg', dataset_name=''):
        """
        :param batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
        :param inference_task: determine the inference task, including ['generic', 'get_content', 'inference_select']
        :param dataset_name:
        :return:
        """
        if self.training:
            losses = {}
            if self.task_switch['coco']:
                self.criterion = self.criterion_switch['open_set']
                prediction_switch = {'part': False, 'whole': True, 'seg': True, 'det': True}
                task = 'visual_openset'
                if not self.coco_mask_on:
                    task = 'det'
                data = batched_inputs if type(batched_inputs) == list else batched_inputs['coco']
                losses_coco = self.forward_seg(data, task=task, prediction_switch=prediction_switch)
                new_losses_coco = {}
                for key, value in losses_coco.items():
                    new_losses_coco['coco.'+str(key)] = losses_coco[key]
                losses.update(new_losses_coco)
            if self.task_switch['coco_track']:
                prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
                self.criterion = self.criterion_switch['1o1_cm']
                data = batched_inputs if type(batched_inputs) == list else batched_inputs['coco']
                losses_coco = self.forward_seg(data, task='visual_refer', prediction_switch=prediction_switch, coco_track=True)
                new_losses_coco = {}
                for key, value in losses_coco.items():
                    new_losses_coco['coco_track.'+str(key)] = losses_coco[key]
                losses.update(new_losses_coco)
            if self.task_switch['sam']:
                prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
                if self.many2many:
                    self.criterion = self.criterion_switch['m2m_cm']
                else:
                    self.criterion = self.criterion_switch['1o1_cm']
                self.criterion.num_classes = 1
                data = batched_inputs if type(batched_inputs) == list else batched_inputs['sam']
                losses_sam = self.forward_seg(data, task='visual_refer', prediction_switch=prediction_switch)
                new_losses_sam = {}
                for key, value in losses_sam.items():
                    new_losses_sam['sam.' + str(key)] = losses_sam[key]
                losses.update(new_losses_sam)

            return losses
        else:
            if 'generic' in inference_task or 'get_content' in inference_task:
                processed_results = self.forward_seg(batched_inputs, task=inference_task, dataset_name=dataset_name)
            elif inference_task == 'visual_openset':
                processed_results = self.evaluate_visual_openset(batched_inputs, task=inference_task, dataset_name=dataset_name)
            else:
                raise NotImplementedError
            return processed_results

    def forward_seg(self, batched_inputs, task='seg', prediction_switch={'part': True, 'whole': True, 'seg': True, 'det': True}, coco_track=False, dataset_name=''):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:
            # mask classification target prepare
            targets = {}
            if "instances" in batched_inputs[0]:
                if task == 'visual_refer':
                    # referring segmentation for one2one match, sam data does not have vocabulary
                    # therefore, each instance can only match to its original one
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    num_mask = gt_instances[0].gt_classes.shape[0]
                    index = torch.randperm(num_mask)
                    if num_mask == 0:
                        print("wrong empty image! argets_per_image.gt_classes.shape[0] ",
                              gt_instances[0].gt_classes.shape[0], "targets_per_image", gt_instances[0])
                    if not coco_track:
                        # fix the number of instances to be self.max_num_instance for efficient training
                        if self.max_num_instance > num_mask:
                            # repeat if it is less than self.max_num_instance
                            rep = 0 if num_mask == 0 else int(self.max_num_instance / num_mask) + 1
                            index = index.repeat(rep)
                        # trim to the self.max_num_instance
                        index = index[:self.max_num_instance]
                    else:
                        # for coco data, we use the original instance length as an image in coco contains
                        # no less than 80 instances.
                        max_num_instance_ori = self.max_num_instance
                        max_num_instance_content_ori = self.max_num_instance_content
                        self.max_num_instance = len(index)
                        if self.max_num_instance == 0:
                            self.max_num_instance = 2
                        if self.max_num_instance_content>self.max_num_instance:
                            self.max_num_instance_content = self.max_num_instance
                    # prepare the ground-truth target for position prompts
                    targets['position'] = self.prepare_targets_interactive(gt_instances, images,
                                                                   prediction_switch=prediction_switch, index=index)
                    index = index[:self.max_num_instance_content]
                    # prepare the ground-truth target for content visual prompts
                    targets['content'] = self.prepare_targets_visual_refer_seg(gt_instances, images,
                                                                      prediction_switch=prediction_switch, task='coco',
                                                                      index=index)
                    if coco_track:
                        # re-initialize the original param
                        self.max_num_instance = max_num_instance_ori
                        self.max_num_instance_content = max_num_instance_content_ori
                else:
                    # visual in-context training with coco data
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    gt_instances_copy = copy.deepcopy(gt_instances)
                    # gather all targets cross all gpus
                    targets['content'] = self.prepare_targets_visual_openset_batch_cross_gpu(gt_instances_copy, images, task=task, prediction_switch=prediction_switch, cross_gpu=True)
                    contain_fake = []  # indicator: if an image on a gpu contains no ground-truth instances
                    for t in targets['content']:
                        contain_fake.append(torch.tensor(t.get('fake', False)))
                    contain_fake = torch.stack(contain_fake).cuda()
                    contain_fake = contain_fake.sum()
                    contain_fake_ = all_gather(contain_fake).sum()
                    if contain_fake_ or not self.cross_gpu_batch:
                        # if a gpu image contain no GT instance, do not do cross-gpu training; return to single-gpu
                        print('fake new taragets ', torch.as_tensor(0.).to('cuda').device)
                        gt_instances_copy = copy.deepcopy(gt_instances)
                        targets['content'] = self.prepare_targets_visual_openset_batch_cross_gpu(gt_instances_copy, images, task=task, prediction_switch=prediction_switch, cross_gpu=False)
                        if contain_fake:
                            for i in range(len(targets['content'])):
                                targets['content'][i]['fake'] = True
                        for i in range(len(targets['content'])):
                            targets['content'][i]['cross_gpu'] = False
                    targets['generic'] = targets['content']

            else:
                targets['generic'] = None
                targets['content'] = None

            outputs, mask_dict = self.sem_seg_head(features, targets=targets, task=task, extra=prediction_switch)
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, mask_dict, task=task, extra=prediction_switch)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        elif task == 'get_content':
            # for inference, pre-process to get the content prompts of the whole dataset
            prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
            # mask classification target
            targets = {}
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets['content'] = self.prepare_targets_visual_openset_batch_cross_gpu(gt_instances, images, task=task,
                                                                        prediction_switch=prediction_switch, cross_gpu=False)
                targets['generic'] = targets['content']
            else:
                targets['generic'] = None
                targets['content'] = None
                print("empty targets", targets, task)
            input_tokens_all, labels = self.sem_seg_head(features, targets=targets, task=task, extra=prediction_switch)
            return input_tokens_all, labels

    def evaluate_visual_openset(self, batched_inputs, task='seg', prediction_switch={'part': True, 'whole': True, 'seg': True, 'det': True}, coco_track=False, dataset_name=''):
        """
        for inference:
        randomly sample some prompts from the pre-processed content prompts as visual examples
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        ###
        task = 'coco'
        prediction_switch = {'part': False, 'whole': True, 'seg': True, 'det': True}
        targets = {}
        targets['generic'] = None
        targets['content'] = None

        outputs, _ = self.sem_seg_head(features, targets=targets, task='visual_openset', extra=prediction_switch)

        mask_cls_results = outputs["pred_logits"]
        mask_box_results = outputs["pred_boxes"]
        if 'det' not in task:
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        else:
            self.semantic_on = self.panoptic_on = self.sem_seg_postprocess_before_inference = False
            self.instance_on = True
            mask_pred_results = torch.zeros(mask_box_results.shape[0], mask_box_results.shape[1],2, 2).to(mask_box_results)

        del outputs

        processed_results = []

        for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})
            new_size = (images.tensor.shape[-2], images.tensor.shape[-1])  # padded size (divisible to 32)
            vis = self.vis
            if vis:
                height, width = image_size[0], image_size[1]
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                mask_box_result = mask_box_result.to(mask_pred_result)
                height = new_size[0]/image_size[0]*height
                width = new_size[1]/image_size[1]*width
                mask_box_result = box_postprocess(mask_box_result, height, width)

                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                processed_results[-1]["instances"] = instance_r

        del mask_pred_results
        return processed_results

    def prepare_targets_interactive(self, targets, images, prediction_switch, task='seg', index=None):
        """
        modified from semantic-sam, do not consider box interactive, only point interactive
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(
                targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
            # pad gt
            h, w = targets_per_image.image_size
            if not self.training:
                h_pad, w_pad = h, w
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_masks = targets_per_image.gt_masks if torch.is_tensor(
                targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor

            if not self.training:
                max_num_instance_ori = self.max_num_instance
                self.max_num_instance = len(gt_masks)
            if len(gt_masks) == 0:
                box_start = self.max_num_instance
                new_targets.append({
                    'boxes': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'points': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'boxes_dn': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    "pb": torch.cat([torch.zeros(self.max_num_instance - box_start), torch.ones(box_start)], 0),
                    'box_start': box_start
                })
                if not self.training:
                    self.max_num_instance = max_num_instance_ori
                continue
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            # do not use box for training in multi target
            box_start = self.max_num_instance
            level_target_inds = []
            # FIXME randomly sample one point as the user input
            if self.regenerate_point and box_start > 0:
                point_coords = []
                for i in range(box_start):
                    mask = gt_masks[index[i]].clone()
                    center_point = True  # FIXME for evaluation sample the center as clicks
                    if not self.training and center_point:
                        mask = mask[None, None, :]
                        n, _, h, w = mask.shape
                        mask_dt = (
                        distance_transform((~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:, :,
                        1:-1, 1:-1])
                        selected_point = torch.tensor([mask_dt.argmax() / w, mask_dt.argmax() % w]).long().cuda().flip(
                            0)
                    else:
                        candidate_indices = mask.nonzero()
                        if len(candidate_indices) == 0:
                            selected_point = torch.tensor([0, 0]).cuda()
                        else:
                            selected_index = random.randint(0, len(candidate_indices) - 1)
                            selected_point = candidate_indices[selected_index].flip(0)
                        # only build level targets for sam data
                        if not prediction_switch['whole'] and not prediction_switch['part']:
                            level_target_ind = []
                            for ind, m in enumerate(gt_masks):
                                if m[tuple(selected_point.flip(0))]:
                                    level_target_ind.append(ind)
                            assert len(level_target_ind) > 0, "each point must have at least one target"
                            # randomly sample some target index if targets exceeds the maximum tokens
                            # FIXME another way is to filter small objects when too many level targets
                            if len(level_target_ind) > self.num_mask_tokens:
                                random.shuffle(level_target_ind)
                                level_target_ind = level_target_ind[:self.num_mask_tokens]
                            level_target_inds.append(level_target_ind)
                    selected_point = torch.cat([selected_point - 3, selected_point + 3], 0)
                    point_coords.append(selected_point)
                point_coords = torch.stack(point_coords).to('cuda')
            else:
                point_coords = targets_per_image.gt_boxes.tensor[index[:box_start]]
            max_num_tgt_per_click = -1
            if len(level_target_inds) > 0:
                num_tgt = [len(l) for l in level_target_inds]
                max_num_tgt_per_click = max(num_tgt)
                if max_num_tgt_per_click > 5:
                    print("max number of levels ", max(num_tgt))
            new_target = {
                "ori_mask_num": len(targets_per_image.gt_classes),
                "level_target_inds": level_target_inds,
                "max_num_tgt_per_click": max_num_tgt_per_click,
                "labels": targets_per_image.gt_classes[index] if prediction_switch['whole'] else None,
                "masks": padded_masks[index],
                "ori_masks": padded_masks,
                "boxes": box_ops.box_xyxy_to_cxcywh(gt_boxes[index]) / image_size_xyxy,
                "ori_boxes": box_ops.box_xyxy_to_cxcywh(gt_boxes) / image_size_xyxy,
                "points": box_ops.box_xyxy_to_cxcywh(point_coords) / image_size_xyxy,
                "pb": torch.cat([torch.zeros(self.max_num_instance - box_start), torch.ones(box_start)], 0),
                "gt_whole_classes": targets_per_image.gt_whole_classes[index] if targets_per_image.has(
                    'gt_whole_classes') and prediction_switch['whole'] else None,
                "gt_part_classes": targets_per_image.gt_part_classes[index] if targets_per_image.has(
                    'gt_part_classes') and prediction_switch['part'] else None,
            }
            # handle coco data format
            if prediction_switch['whole'] and not prediction_switch['part']:
                new_target['gt_whole_classes'] = targets_per_image.gt_classes[index]

            if not self.training:
                self.max_num_instance = max_num_instance_ori
                new_target["pb"] = torch.zeros_like(new_target["pb"])
                auto_points = torch.tensor(build_point_grid(self.point_per_side)).cuda() * image_size_xyxy[:2].to(
                    new_target["points"]).to(torch.float32)

                new_target["points"] = box_ops.box_xyxy_to_cxcywh(
                    torch.cat([auto_points - 3, auto_points + 3], 1)) / image_size_xyxy
                new_target["points"] = torch.tensor(new_target["points"], dtype=torch.float32)
                new_target["pb"] = torch.zeros(new_target["points"].shape[0])
                ####
                height = images[0].shape[1]
                width = images[0].shape[2]
                padded_h = images.tensor.shape[-2]  # divisable to 32
                padded_w = images.tensor.shape[-1]
                new_target['points'] = new_target['points'] * torch.as_tensor([width, height, width, height],
                                                                              dtype=torch.float,
                                                                              device=self.device) / torch.as_tensor(
                    [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
                new_target['boxes'] = new_target['boxes'] * torch.as_tensor([width, height, width, height],
                                                                            dtype=torch.float,
                                                                            device=self.device) / torch.as_tensor(
                    [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
            new_target["boxes_dn"] = torch.cat([new_target["points"], new_target["boxes"][box_start:]], 0)
            new_target['box_start'] = box_start
            new_targets.append(new_target)

        return new_targets

    def prepare_targets_visual_refer_seg(self, targets, images, prediction_switch, task='seg', index=None):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(
                targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
            # pad gt
            h, w = targets_per_image.image_size
            if not self.training:
                h_pad, w_pad = from_divisablity(h, self.size_divisibility), from_divisablity(w, self.size_divisibility)

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_masks = targets_per_image.gt_masks if torch.is_tensor(
                targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor
            if not self.training:
                max_num_instance_ori = self.max_num_instance_content
                self.max_num_instance_content = len(gt_masks)
            len_gt = len(gt_masks)
            if len_gt == 0:
                box_start = self.max_num_instance_content
                new_target = {
                    'boxes': torch.ones(self.max_num_instance_content, 4).to(gt_masks).float(),
                    'rand_shape': torch.ones(self.max_num_instance_content, h_pad, w_pad).to(gt_masks).float(),
                    'points': torch.ones(self.max_num_instance_content, 4).to(gt_masks).float(),
                    'boxes_dn': torch.ones(self.max_num_instance_content, 4).to(gt_masks).float(),
                    "pb": torch.cat([torch.zeros(self.max_num_instance_content - box_start), torch.ones(box_start)], 0),
                    'box_start': box_start,
                }
                if prediction_switch['whole']:
                    new_target["gt_whole_classes"] = torch.tensor([len(self.train_class_names[task]) - 1]).repeat(
                        self.max_num_instance_content).to(gt_masks.device)
                new_targets.append(new_target)

                if not self.training:
                    self.max_num_instance_content = max_num_instance_ori
                continue
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            # do not use box for training in multi target
            box_start = self.max_num_instance_content
            point_coords = torch.ones(self.max_num_instance_content, 4).to(padded_masks).float()
            point_coords[:, :2] = 0.
            # randomly sample one point as the user input
            new_padded_masks = padded_masks[index]
            if self.use_shape_sampler:
                # sample a random scribble as visual prompt
                content_dict = self.shape_sampler(new_padded_masks, gt_boxes[index], self.max_num_instance_content)
                rand_shape = content_dict['rand_shape'].cuda()
            else:
                rand_shape = new_padded_masks
            new_target = {
                "rand_shape": rand_shape,
                "ori_mask_num": len(targets_per_image.gt_classes),
                "labels": targets_per_image.gt_classes[index] if prediction_switch['whole'] else None,
                "masks": padded_masks[index],
                "ori_masks": padded_masks,
                "boxes": box_ops.box_xyxy_to_cxcywh(gt_boxes[index]) / image_size_xyxy,
                "ori_boxes": box_ops.box_xyxy_to_cxcywh(gt_boxes) / image_size_xyxy,
                "points": point_coords,
                "pb": torch.cat([torch.zeros(self.max_num_instance_content - box_start), torch.ones(box_start)], 0),
                "gt_whole_classes": targets_per_image.gt_whole_classes[index] if targets_per_image.has(
                    'gt_whole_classes') and prediction_switch['whole'] else None,
                "gt_part_classes": targets_per_image.gt_part_classes[index] if targets_per_image.has(
                    'gt_part_classes') and prediction_switch['part'] else None,
            }
            # handle coco data format
            if prediction_switch['whole'] and not prediction_switch['part']:
                new_target['gt_whole_classes'] = targets_per_image.gt_classes[index]

            if not self.training:
                self.max_num_instance_content = max_num_instance_ori
                new_target["masks_unpadded"] = new_target["masks"][:, :h, :w]
                height = images[0].shape[1]
                width = images[0].shape[2]
                padded_h = images.tensor.shape[-2]  # divisable to 32
                padded_w = images.tensor.shape[-1]
                new_target["boxes_dn_ori"] = torch.cat(
                    [new_target["points"].clone(), new_target["boxes"][box_start:].clone()], 0)
                new_target['boxes'] = new_target['boxes'] * torch.as_tensor([width, height, width, height],
                                                                            dtype=torch.float,
                                                                            device=self.device) / torch.as_tensor(
                    [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
            new_target["boxes_dn"] = torch.cat([new_target["points"], new_target["boxes"][box_start:]], 0)
            new_target['box_start'] = box_start
            new_targets.append(new_target)

        return new_targets

    def prepare_targets_visual_openset_batch_cross_gpu(self, targets, images, task='seg', prediction_switch=None, cross_gpu=False):
        """
        Prepare visual prompt examples from a large batch to construct positive and negative examples
        :param cross_gpu: if set to true, will prepare from multi-gpus
        :return:
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        id_start = 0
        id_start_list = [0]
        empty_flag = False
        for targets_per_image in targets:
            gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(
                targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
            # pad gt
            h, w = targets_per_image.image_size
            if not self.training:
                h_pad, w_pad = from_divisablity(h, self.size_divisibility), from_divisablity(w, self.size_divisibility)

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            if hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks if torch.is_tensor(
                    targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor
            else:
                # create a fake target
                gt_masks = torch.zeros((gt_boxes.shape[0], h_pad, w_pad), dtype=gt_boxes.dtype,
                                       device=gt_boxes.device)
                for i, box in enumerate(gt_boxes):
                    gt_masks[i][int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1.
            max_num_instance_openset = len(gt_masks)
            len_gt = len(gt_masks)
            if len_gt == 0:
                empty_flag = True
                max_num_instance_openset = self.default_num_instance_openset
                box_start = max_num_instance_openset
                new_target = {
                    'boxes': torch.ones(max_num_instance_openset, 4).to(gt_masks).float(),
                    'rand_shape': torch.ones(max_num_instance_openset, h_pad, w_pad).to(gt_masks).float(),
                    'points': torch.ones(max_num_instance_openset, 4).to(gt_masks).float(),
                    'boxes_dn': torch.ones(max_num_instance_openset, 4).to(gt_masks).float(),
                    "pb": torch.cat([torch.zeros(max_num_instance_openset - box_start), torch.ones(box_start)], 0),
                    'box_start': box_start,
                    'fake': True,
                }
                if prediction_switch['whole']:
                    new_target["gt_whole_classes"] = new_target["labels"] = torch.tensor(0).repeat(max_num_instance_openset).to(
                        gt_masks.device)
                    new_target["masks"] = new_target["rand_shape"]
                new_targets.append(new_target)
                continue
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            num_mask = targets_per_image.gt_classes.shape[0]
            # process class annotations for visual open-set targets
            gt_classes = targets_per_image.gt_classes
            # do not use box for training in multi target
            box_start = max_num_instance_openset
            point_coords = torch.ones(max_num_instance_openset, 4).to(padded_masks).float()
            point_coords[:, :2] = 0.
            if self.openset_use_shape_sampler:
                content_dict = self.shape_sampler(padded_masks, gt_boxes, self.max_num_instance)
                rand_shape = content_dict['rand_shape'].cuda()
            else:
                rand_shape = padded_masks
            ####
            new_target = {
                "rand_shape": rand_shape,
                "ori_mask_num": len(targets_per_image.gt_classes),
                "labels": targets_per_image.gt_classes if prediction_switch['whole'] else None,
                "masks": padded_masks,
                "boxes": box_ops.box_xyxy_to_cxcywh(gt_boxes) / image_size_xyxy,
                "points": point_coords,
                "pb": torch.cat([torch.zeros(max_num_instance_openset - box_start), torch.ones(box_start)], 0),
                'gt_whole_classes': targets_per_image.gt_classes if prediction_switch['whole'] else None,
                "gt_part_classes": targets_per_image.gt_part_classes if targets_per_image.has(
                    'gt_part_classes') and prediction_switch['part'] else None,
            }

            if not self.training:
                new_target["masks_unpadded"] = new_target["masks"][:, :h, :w]
                height = images[0].shape[1]
                width = images[0].shape[2]
                padded_h = images.tensor.shape[-2]  # divisable to 32
                padded_w = images.tensor.shape[-1]
                new_target["boxes_dn_ori"] = torch.cat(
                    [new_target["points"].clone(), new_target["boxes"][box_start:].clone()], 0)
                new_target['boxes'] = new_target['boxes'] * torch.as_tensor([width, height, width, height],
                                                                            dtype=torch.float,
                                                                            device=self.device) / torch.as_tensor(
                    [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
            new_target["boxes_dn"] = torch.cat([new_target["points"], new_target["boxes"][box_start:]], 0)
            new_target['box_start'] = box_start
            new_targets.append(new_target)

            unique_category_examples_by_category = getIdx(gt_classes, id_start)
            id_start += num_mask
            id_start_list.append(id_start)
            batch_examples_by_category = {}
            for k, v in unique_category_examples_by_category.items():
                if k in batch_examples_by_category.keys():
                    batch_examples_by_category[k] = torch.cat(
                        [batch_examples_by_category[k], unique_category_examples_by_category[k]])
                else:
                    batch_examples_by_category[k] = unique_category_examples_by_category[k]

        # HACK to handle if 1 image does not have target annotations
        if empty_flag:
            for i, target in enumerate(new_targets):
                target['fake'] = True
        print("cross_gpu, new_targets ", cross_gpu, len(new_targets))
        if not empty_flag and not cross_gpu:
            # handle batch in 1 gpu only, do not need cross gpu sync
            # if cross_gpu=True, sync will be performed in the decoder
            if self.training:
                self.criterion.num_classes = len(batch_examples_by_category)
            sampled_examples_by_catetory = {}
            max_example_num = self.max_train_example
            # re-arrange targets by combining different images in one batch
            # i.e, two images, I1 with 3 instances in 2 categories, I2 with 4 instances in 3 categories, they may have
            # a share category (positive samples) and different category (negative samples)
            new_labels = []
            label_index = []
            start = 1
            for i, (cat, examples) in enumerate(batch_examples_by_category.items()):
                end = max_example_num if max_example_num < len(examples) else len(examples)
                if task == 'get_content':
                    start = end = len(examples)
                example_num = random.randint(start, end)
                shuffle_examples = examples[torch.randperm(len(examples))[:example_num]]
                sampled_examples_by_catetory[cat] = shuffle_examples
                new_labels.append(torch.full_like(examples, i).long())
                label_index.append(examples)
            label_index = torch.cat(label_index)
            # two images may share some lables
            value, indices = label_index.sort()
            # the index used for content feature extraction are candidates for shape sampling
            sample_index = torch.cat(list(sampled_examples_by_catetory.values()))
            all_rand_shape = torch.cat([t['rand_shape'] for t in new_targets], 0)
            if self.use_shape_sampler and task != 'get_content':
                sample_ratio = torch.rand_like(sample_index.float())
                keep = sample_ratio > -0.7  # 30% sample shapes
                sample_index = sample_index[keep]
                max_num_instance_openset = len(all_rand_shape[sample_index])
                content_dict = self.shape_sampler(all_rand_shape[sample_index], all_rand_shape[sample_index],
                                                  max_num_instance_openset)
                all_rand_shape[sample_index] = content_dict['rand_shape'].cuda()
            new_rand_shape_per_instance = [all_rand_shape[id_start_list[i]:id_start_list[i + 1]] for i in
                                           range(len(id_start_list) - 1)]
            new_labels = torch.cat(new_labels)
            new_labels = new_labels[indices]
            new_labels_per_instance = [new_labels[id_start_list[i]:id_start_list[i + 1]] for i in
                                       range(len(id_start_list) - 1)]

            for i, target in enumerate(new_targets):
                target['labels'] = target['gt_whole_classes'] = new_labels_per_instance[i]
                target['rand_shape'] = new_rand_shape_per_instance[i]
                target['sampled_examples_by_catetory'] = sampled_examples_by_catetory

        return new_targets

    def prepare_image(self, batched_inputs, key='image'):
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def filter_data_refer(self, src_boxes, mask_pred_results, src_ious, pred_ious):
        # filter segmentation mask with prediction scores
        def keep_data(box, mask, iou, match_score, keep):
            return box[keep], mask[keep], iou[keep], match_score[:, keep]

        keep = src_ious > 0.5
        src_boxes, mask_pred_results, src_ious, pred_ious = keep_data(src_boxes, mask_pred_results, src_ious, pred_ious,
                                                                      keep)

        item_indice = nms(box_ops.box_cxcywh_to_xyxy(src_boxes), src_ious, self.nms_thersh)  # FIXME iou threshold

        mask_pred_results = mask_pred_results[item_indice]
        src_boxes = src_boxes[item_indice]
        src_ious = src_ious[item_indice]
        pred_ious = torch.index_select(pred_ious, -1, item_indice)
        # remove small objects
        keep = (mask_pred_results > 0).flatten(-2, -1).sum(-1) > 50
        src_boxes, mask_pred_results, src_ious, pred_ious = keep_data(src_boxes, mask_pred_results, src_ious,
                                                                      pred_ious, keep)

        pred_ious = F.softmax(pred_ious, dim=-1)
        if pred_ious.shape[-1]<6:
            print(pred_ious.shape)
        scores_per_image, level = pred_ious.topk(1)
        src_ious = torch.gather(src_ious, 0, level[0])
        pred_ious = scores_per_image
        mask_pred_results = torch.gather(mask_pred_results[None].repeat(level.shape[0], 1, 1, 1), 1,
                                         level[:, :, None, None].repeat(1, 1, mask_pred_results.shape[-2],
                                                                        mask_pred_results.shape[-1]))

        return src_boxes, mask_pred_results, src_ious, pred_ious

    def get_encoder_feature(self, batched_inputs):
        # get the image encoder features (multi-scale)
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        images = self.prepare_image(batched_inputs)
        padded_h = images.tensor.shape[-2]  # divisable to 32
        padded_w = images.tensor.shape[-1]
        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(
            features, None)

        return multi_scale_features, mask_features, padded_h, padded_w

    def get_visual_prompt_content_feature(self, multi_scale_features, rand_shape, padded_h, padded_w, vid_name='default'):
        # prepare visual prompt features
        targets = [{'rand_shape': None, 'boxes_dn': None, 'box_start': None}]
        height = rand_shape.shape[-2]
        width = rand_shape.shape[-1]
        num_instance = len(rand_shape)

        point_coords = torch.ones(num_instance, 4).to(multi_scale_features[0]).float()
        point_coords[:, :2] = 0.
        targets[0]['rand_shape'] = rand_shape.cuda()
        targets[0]['boxes_dn'] = point_coords
        new_rand_shape = torch.zeros(num_instance, padded_h, padded_w).to(targets[0]['rand_shape'])
        new_rand_shape[:, :height, :width] = targets[0]['rand_shape']

        targets[0]['rand_shape'] = new_rand_shape
        targets[0]['box_start'] = num_instance
        targets[0]['pb'] = torch.ones(num_instance).to(targets[0]['rand_shape'])

        input_query_label_content, input_query_bbox_content, attn_mask_content = self.sem_seg_head.predictor.\
            forward_get_content_feature(multi_scale_features, None, targets=targets)

        return input_query_label_content, input_query_bbox_content, attn_mask_content

    def evaluate_visual_prompt_refer_multi_with_content_features(self, batched_inputs, mask_features, multi_scale_features,
                                                                input_query_label_content,
                                                                input_query_bbox_content, attn_mask_content,
                                                                padded_h, padded_w,
                                                                level=[0,1,2,3,4,5], return_src_ious=False):
        # evaluate referring segmentation
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
        images = self.prepare_image(batched_inputs)
        image_size_xyxy = torch.tensor([padded_w, padded_h, padded_w, padded_h]).cuda()
        # dense points 20x20 to get all mask proposals in one image
        auto_points = torch.tensor(
            build_point_grid(self.point_per_side)).cuda() * image_size_xyxy[:2]

        boxes_dn = box_ops.box_xyxy_to_cxcywh(
            torch.cat([auto_points - 3, auto_points + 3], 1)) / image_size_xyxy
        boxes_dn = torch.tensor(boxes_dn, dtype=torch.float32)
        pb = torch.ones(boxes_dn.shape[0])
        targets_p = [{}]
        targets_p[0]['boxes_dn'] = boxes_dn
        targets_p[0]['pb'] = pb

        targets = targets_p
        outputs, mask_dict = self.sem_seg_head.predictor.forward_refer_image_with_extracted_content(multi_scale_features,
                                                         mask_features, None, input_query_label_content, input_query_bbox_content, attn_mask_content, targets,
                                                         extra=prediction_switch)
        src_boxes = outputs["pred_boxes"][0]
        mask_pred_results = outputs["pred_masks"][0]
        pred_ious = outputs["pred_match_score"][0]
        src_ious = outputs["pred_ious"][0]
        # decide which level to use; default use all 6 levels
        level = torch.tensor(level).cuda()
        mask_pred_results = torch.index_select(
            mask_pred_results.view(src_ious.shape[0], src_ious.shape[1], mask_pred_results.shape[1],
                                   mask_pred_results.shape[2]), 1, level).flatten(0, 1)
        src_boxes = torch.index_select(src_boxes.view(src_ious.shape[0], src_ious.shape[1], src_boxes.shape[-1]), 1,
                                       level).flatten(0, 1)
        pred_ious = torch.index_select(pred_ious.view(pred_ious.shape[0], src_ious.shape[0], src_ious.shape[1]), -1, level).flatten(-2,-1)
        src_ious = torch.index_select(src_ious, -1, level)
        src_ious = src_ious.flatten(0, 1)
        src_boxes, mask_pred_results, src_ious, pred_ious = self.filter_data_refer(src_boxes, mask_pred_results, src_ious,
                                                                             pred_ious)

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        pred_masks = mask_pred_results
        image_size = images.image_sizes[0]
        pred_masks = pred_masks[:, 0]
        pred_ious = pred_ious[:, 0]
        height = batched_inputs[0].get('height', image_size[0])
        width = batched_inputs[0].get('width', image_size[1])
        ori_masks = pred_masks[:, : image_size[0], : image_size[1]].expand(1, -1, -1, -1)[0]
        if self.sem_seg_postprocess_before_inference:
            pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                pred_masks, image_size, height, width
            )
        return pred_masks, pred_ious, ori_masks

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        mask_cls = mask_cls.float()
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        self.test_topk_per_image = 300
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.float().sigmoid()  # [100, 80]
        assert self.sem_seg_head.num_classes == scores.shape[-1]
        if hasattr(self.metadata, 'cat_dirs'):
            # know the real number of classes in visual prompt
            cat_dirs = self.metadata.cat_dirs
            not_keep = [i not in cat_dirs for i in range(self.sem_seg_head.num_classes)]
            scores[:, not_keep] = 0.0  # set the invalid place as 0.0 score
            # handle seginw bad entry
            if self.sem_seg_head.num_classes == 2 and scores.shape[-1] == 1:
                assert ValueError

        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
        test_topk_per_image_ori = self.test_topk_per_image
        if scores.flatten(0, 1).shape[0]<self.test_topk_per_image:
            self.test_topk_per_image = scores.flatten(0, 1).shape[0]
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        num_classes_ori = self.sem_seg_head.num_classes
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        self.sem_seg_head.num_classes = num_classes_ori
        mask_pred = mask_pred[topk_indices]##
        self.test_topk_per_image = test_topk_per_image_ori
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # calculate average mask prob
        if self.sem_seg_postprocess_before_inference:
            mask_scores_per_image = (mask_pred.float().sigmoid().flatten(1) * result.pred_masks.float().flatten(1)).sum(1) / (result.pred_masks.float().flatten(1).sum(1) + 1e-6)
        else:
            mask_scores_per_image = 1.0
            # labels_per_image = labels_per_image + 1  # HACK for o365 classification
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

@register_model
def get_segmentation_model(cfg, **kwargs):
    return DINOv(cfg)