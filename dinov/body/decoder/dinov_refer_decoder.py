# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------
import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d
from timm.models.layers import trunc_normal_

from .registry import register_decoder
from .utils.dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from .utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from ...utils import configurable
from ..transformer_blocks import CrossAttentionLayer


class DINOvRefer(nn.Module):
    @configurable
    def __init__(
            self,
            lang_encoder: nn.Module,
            in_channels,
            mask_classification,
            num_classes: int,
            hidden_dim: int,
            dim_proj: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_project: bool,
            two_stage: bool,
            dn: str,
            noise_scale: float,
            dn_num: int,
            initialize_box_type: bool,
            initial_pred: bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,
            num_mask_tokens: int = 3,
            num_content_tokens: int = 3,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn = dn
        self.learn_tgt = learn_tgt
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes = num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        self.dim_proj = dim_proj
        self.lang_encoder = lang_encoder
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          )

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

        # whole category classification from semantic-sam, not used
        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)
        # part category classification from semantic-sam, not used
        self.class_embed_part = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed_part, std=.02)
        self.num_mask_tokens = num_mask_tokens  # sam uses 4 to handle multi prompts
        self.num_all_tokens = self.num_mask_tokens  # sam uses 4 to handle multi prompts
        self.iou_prediction_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_dim)
        self.pb_embedding = nn.Embedding(2, hidden_dim)
        self.label_enc = nn.Embedding(2, hidden_dim)

        # for content tokens
        self.num_content_tokens = num_content_tokens
        self.content_tokens = nn.Embedding(self.num_content_tokens, hidden_dim)
        self.prediction_switch = None

        # visual prompt extractor
        self.pos_embed = nn.Embedding(self.num_content_tokens, hidden_dim)
        self.project_cross_attention_layers = nn.ModuleList()
        for i in range(self.num_feature_levels):
            self.project_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

    def prepare_for_point_query(self, targets, tgt, refpoint_emb, batch_size):
        # for interactive segmentation, prepare point query
        noise_scale = self.noise_scale
        pb_labels = torch.stack([t['pb'] for t in targets])
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.stack([t['boxes_dn'] for t in targets])
        if self.training:
            box_start = [t['box_start'] for t in targets]

        known_labels = labels
        known_pb_labels = pb_labels

        known_bboxs = boxes
        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale > 0 and self.training:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :, :2] = known_bbox_expand[:, :, 2:] / 2
            diff[:, :, 2:] = known_bbox_expand[:, :, 2:]
            # only add very small noise to the input point for more robust training
            sc = 0.01
            for i, st in enumerate(box_start):
                diff[i, :st] = diff[i, :st] * sc
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                           diff).cuda() * noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to('cuda')
        m_pb = known_pb_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m) + self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(self.num_mask_tokens,
                                                                1) + self.mask_tokens.weight.unsqueeze(0).repeat(
            input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_mask_tokens, 1)

        single_pad = self.num_mask_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_mask_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'pad_size': pad_size,
            'scalar': scalar,
        }

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_for_visual_query(self, targets, src_features, size_list):
        """
        single image referring for SA-1B data
        """
        num_examples = [len(t['pb']) for t in targets]
        max_num = max(num_examples)
        bs = len(targets)
        pb_labels = torch.cat([t['pb'] for t in targets])[None].repeat(bs, 1)
        # placeholder for content embedding
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.stack([t['boxes_dn'] for t in targets])
        m = labels.long().to('cuda')
        m_pb = pb_labels.long().to('cuda')
        input_label_embed = self.label_enc(m) + self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(boxes)
        # consider batchsize=1
        bs = input_label_embed.shape[0]
        project_attention_mask = torch.stack([t['rand_shape'] for t in targets]).repeat_interleave(
            self.num_content_tokens, 1)
        input_tokens = self.content_tokens.weight.unsqueeze(0).repeat(input_label_embed.shape[0],
                                                                      input_label_embed.shape[1], 1).transpose(0, 1)
        query_embed = self.pos_embed.weight.unsqueeze(1).repeat(input_label_embed.shape[1], bs, 1)
        attn_mask = project_attention_mask
        attn_mask_list = []
        h, w = project_attention_mask.shape[-2:]
        max_size = size_list[0][0]
        size_list_all = [torch.Size([int(h / (2 ** i)), int(w / (2 ** i))]) for i in
                         range(1, int(math.log(int(h / max_size), 2)))] + size_list
        for size in size_list_all:
            attn_mask = F.interpolate(attn_mask.float(), size=size, mode="bilinear", align_corners=False)
            attn_mask_list.append(attn_mask)
        for i, (src, size) in enumerate(zip(src_features, size_list[::-1])):
            attn_mask = attn_mask_list[-(i + 1)]
            attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) <= 0.).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # see the whole image when the attention mask is fully masked

            input_tokens = self.project_cross_attention_layers[i](input_tokens, src, memory_mask=attn_mask,
                                                                  memory_key_padding_mask=None, pos=None,
                                                                  query_pos=query_embed)
        ###
        input_label_embed = input_label_embed.repeat_interleave(self.num_content_tokens, 1) + input_tokens.transpose(0,
                                                                                                                     1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_content_tokens, 1)
        single_pad = self.num_content_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_content_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'pad_size': pad_size,
            'scalar': scalar,
        }

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_all_query_sam(self, targets, tgt, refpoint_emb, batch_size, src_features, size_list):
        input_query_label_position, input_query_bbox_position, attn_mask_position, mask_dict = self.prepare_for_point_query(
            targets['position'], None, None, batch_size)
        input_query_label_content, input_query_bbox_content, attn_mask_content, _ = self.prepare_for_visual_query(
            targets['content'], src_features, size_list)
        query_num_dict = {}
        if not (self.dn != "no" and mask_dict is not None):
            # avoid no gradient on mask attention
            return input_query_label_position + 0.0 * input_query_label_content.sum() + 0.0 * input_query_bbox_content.sum(), input_query_bbox_position, attn_mask_position, mask_dict, query_num_dict

        input_query_label = torch.cat([input_query_label_position, input_query_label_content], 1)
        input_query_bbox = torch.cat([input_query_bbox_position, input_query_bbox_content], 1)

        num_position = attn_mask_position.shape[0]
        num_content = attn_mask_content.shape[0]
        mask_dict['num_position'] = num_position
        mask_dict['num_content'] = num_content
        # all True, no one can see each other
        attn_mask = attn_mask_position.new_ones(num_position + num_content, num_position + num_content) > 0
        attn_mask[:num_position, :num_position] = attn_mask_position
        attn_mask[num_position:, num_position:] = attn_mask_content
        # attn_mask[num_position:, :num_position] = True  # FIXME content query cannot see generic query
        return input_query_label, input_query_bbox, attn_mask, mask_dict, query_num_dict

    def prepare_for_dn_content(self, targets, src_features, size_list, return_all_content_tokens=False):
        """
        prepare visual prompt tokens
        modified from denoising training in DN-DETR
        :param targets:
        :param src_features:
        :param size_list:
        :param return_all_content_tokens:
        :return:
        """
        num_examples = [len(t['pb']) for t in targets]
        max_num = max(num_examples)
        bs = len(targets)
        pb_labels = torch.cat([t['pb'] for t in targets])[None].repeat(bs, 1)
        # placeholder for content embedding
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.cat([t['boxes_dn'] for t in targets])[None].repeat(bs, 1, 1)
        m = labels.long().to('cuda')
        m_pb = pb_labels.long().to('cuda')
        input_label_embed = self.label_enc(m)+self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(boxes)
        # consider batchsize=1
        h, w = targets[0]['rand_shape'].shape[1:]
        project_attention_mask = torch.stack(
            [torch.cat([t['rand_shape'], (torch.zeros(max_num - len(t['pb']), h, w) < 1).to(t['rand_shape'])]) for t in
             targets]).repeat_interleave(self.num_content_tokens, 1)
        input_tokens = self.content_tokens.weight.unsqueeze(0).repeat(bs, project_attention_mask.shape[1], 1).transpose(0,1)
        query_embed = self.pos_embed.weight.unsqueeze(1).repeat(project_attention_mask.shape[1], bs, 1)
        attn_mask = project_attention_mask
        attn_mask_list = []
        h, w = project_attention_mask.shape[-2:]
        max_size = size_list[0][0]
        size_list_all = [torch.Size([int(h/(2**i)), int(w/(2**i))]) for i in range(1, int(math.log(int(h/max_size), 2)))] + size_list

        for size in size_list_all:
            attn_mask = F.interpolate(attn_mask.float(), size=size, mode="bilinear", align_corners=False)
            attn_mask_list.append(attn_mask)
        for i, (src, size) in enumerate(zip(src_features, size_list[::-1])):
            attn_mask = attn_mask_list[-(i+1)]
            attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) <= 0.).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # see the whole image when the attention mask is fully masked
            input_tokens = self.project_cross_attention_layers[i](input_tokens, src, memory_mask=attn_mask, memory_key_padding_mask=None, pos=None, query_pos=query_embed)

        input_tokens_all = []
        for i, target in enumerate(targets):
            input_token = input_tokens[:, i][:num_examples[i]]
            input_tokens_all.append(input_token)
        input_tokens_all = torch.cat(input_tokens_all, 0)
        if return_all_content_tokens:
            return input_tokens_all

        input_label_embed = input_label_embed.repeat_interleave(self.num_content_tokens, 1) + input_tokens.transpose(0,
                                                                                                                     1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_content_tokens, 1)
        single_pad = self.num_content_tokens
        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_content_tokens)
        pad_size = input_label_embed.shape[1]
        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'pad_size': pad_size,
            'scalar': scalar,
        }

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def forward_refer_image_with_extracted_content(self, x, mask_features, masks, input_query_label_content, input_query_bbox_content, attn_mask_content, targets=None, extra={}):
        """
        task: seg/det TODO add sam
        """
        prediction_switch = extra
        self.prediction_switch = prediction_switch
        assert len(x) == self.num_feature_levels
        do_seg = True  # if task is det, not do segmentation training
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src
                     in x]
        src_features = []
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            flatten = self.input_proj[idx](x[idx]).flatten(2)
            src_features.append(flatten.permute(2, 0, 1))
            src_flatten.append(flatten.transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        predictions_iou_score = []
        predictions_match_score = []

        assert targets is not None
        input_query_label_position, input_query_bbox_position, attn_mask_position, mask_dict = self.prepare_for_point_query(targets, None, None, bs)
        input_query_label = torch.cat([input_query_label_position, input_query_label_content], 1)
        input_query_bbox = torch.cat([input_query_bbox_position, input_query_bbox_content], 1)

        num_position = attn_mask_position.shape[0]
        num_content = attn_mask_content.shape[0]
        mask_dict['num_position'] = num_position
        mask_dict['num_content'] = num_content
        # all True, no one can see each other
        attn_mask = attn_mask_position.new_ones(num_position + num_content, num_position + num_content) > 0
        attn_mask[:num_position, :num_position] = attn_mask_position
        attn_mask[num_position:, num_position:] = attn_mask_content

        tgt, refpoint_embed, tgt_mask = input_query_label, input_query_bbox, attn_mask

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        new_hs = []
        for i, output in enumerate(hs):
            outputs_class, outputs_mask, iou_score, decoder_output_mask, match_score = self.forward_prediction_heads_refer(output.transpose(0, 1), mask_features, (self.training or (i == len(hs)-1)) and do_seg, mask_dict)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_match_score.append(match_score)
            if iou_score is not None:
                predictions_iou_score.append(iou_score)
                new_hs.append(decoder_output_mask)
        if new_hs is not None:
            hs = new_hs
        # iteratively box prediction
        assert mask_dict is not None
        references = [r[:, :mask_dict['num_position']] for r in references]
        out_boxes = self.pred_box(references, hs)
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': None if not do_seg else predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'pred_ious': predictions_iou_score[-1],
            'pred_match_score': predictions_match_score[-1],
        }

        return out, mask_dict

    def dn_post_process(self, outputs_class, outputs_coord, mask_dict, outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        output_known_mask = None
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],
               'pred_masks': None if output_known_mask is None else output_known_mask[-1]}
        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask, output_known_coord)
        mask_dict['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord, outputs_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward_get_content_feature(self, x, masks, targets=None, extra={}):
        x_resize = x
        assert len(x_resize) == self.num_feature_levels
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        def build_src(x):
            size_list = []
            src_features = []
            src_flatten = []
            mask_flatten = []
            spatial_shapes = []
            if enable_mask == 0:
                masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for
                         src in x]
            ss = []
            for i in range(len(x)):
                ss.append(x[i].shape)
            for i in range(self.num_feature_levels):
                if i >= len(x):
                    print("not enough dim")
                idx = self.num_feature_levels - 1 - i
                bs, c, h, w = x[idx].shape
                size_list.append(x[i].shape[-2:])
                spatial_shapes.append(x[idx].shape[-2:])
                flatten = self.input_proj[idx](x[idx]).flatten(2)
                src_features.append(flatten.permute(2, 0, 1))
                src_flatten.append(flatten.transpose(1, 2))
                mask_flatten.append(masks[i].flatten(1))
            src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
            mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
            return src_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios, src_features, size_list

        src_flatten, mask_flatten, spatial_shapes, level_start_index_, valid_ratios, src_features, size_list = build_src(
            x_resize)

        input_query_label_content, input_query_bbox_content, attn_mask_content, _ = self.prepare_for_dn_content(
            targets, src_features, size_list)

        return input_query_label_content, input_query_bbox_content, attn_mask_content

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward_train_refer(self, x, mask_features, masks, targets=None, target_queries=None, target_vlp=None, task='seg',
                      extra={}):
        """
        task: seg/det TODO add sam
        """
        task = 'sam'
        prediction_switch = extra
        self.prediction_switch = prediction_switch
        assert len(x) == self.num_feature_levels
        do_seg = (task != 'det')  # if task is det, not do segmentation training
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src
                     in x]
        src_features = []
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            flatten = self.input_proj[idx](x[idx]).flatten(2)
            src_features.append(flatten.permute(2, 0, 1))
            src_flatten.append(flatten.transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        predictions_iou_score = []
        predictions_match_score = []

        tgt_mask = None
        mask_dict = None
        if self.dn != "no":
            assert targets is not None
            tgt, refpoint_embed, tgt_mask, mask_dict, query_num_dict = \
                self.prepare_all_query_sam(targets, None, None, x[0].shape[0], src_features, size_list)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        new_hs = []
        for i, output in enumerate(hs):
            outputs_class, outputs_mask, iou_score, decoder_output_mask, match_score = self.forward_prediction_heads_refer(
                output.transpose(0, 1), mask_features, (self.training or (i == len(hs) - 1)) and do_seg, mask_dict)
            outputs_class_whole = outputs_class
            predictions_class.append(outputs_class_whole)
            predictions_mask.append(outputs_mask)
            predictions_match_score.append(match_score)
            if iou_score is not None:
                predictions_iou_score.append(iou_score)
                new_hs.append(decoder_output_mask)
        if new_hs is not None:
            hs = new_hs
        # iteratively box prediction
        assert mask_dict is not None
        references = [r[:, :mask_dict['num_position']] for r in references]
        out_boxes = self.pred_box(references, hs)
        out_boxes[-1] = out_boxes[-1] + 0.0 * (self.label_enc.weight.sum() + self.pb_embedding.weight.sum()
                                               + self.mask_tokens.weight.sum())
        if mask_dict is not None:
            if predictions_mask is None:
                predictions_class[-1] = predictions_class[-1]
                for i in range(self.mask_embed.num_layers):
                    predictions_class[-1] = predictions_class[-1] + 0.0 * (
                                self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[
                            0])  # avoid no mask loss
                predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss

            if do_seg:
                predictions_mask = list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            for i in range(self.mask_embed.num_layers):
                predictions_class[-1] = predictions_class[-1] + 0.0 * (
                        self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[
                    0])  # avoid no mask loss
            predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': None if not do_seg else predictions_mask[-1],
            'pred_boxes': out_boxes[-1],
            'pred_ious': predictions_iou_score[-1],
            'pred_match_score': predictions_match_score[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, out_boxes,
                predictions_iou_score, predictions_match_score=predictions_match_score
            )
        }

        return out, mask_dict

    def forward_prediction_heads_refer(self, output, mask_features, pred_mask=True, mask_dict=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        decoder_output = decoder_output + 0.0 * (self.class_embed_part.sum() + self.class_embed.sum())
        outputs_mask = None
        class_embed_whole = decoder_output @ self.class_embed
        match_score = None
        if mask_dict is not None:
            num_content = mask_dict['num_content']
            class_embed_whole_content = class_embed_whole[:, -num_content:]
            class_embed_whole_content = class_embed_whole_content.view(class_embed_whole.shape[0], -1,
                                                                       self.num_content_tokens,
                                                                       class_embed_whole.shape[
                                                                           -1])  # remove content embedding
            class_embed_whole_content = class_embed_whole_content[:, :, -1, :]  # select the last one of all  tokens
            class_embed_whole_generic = class_embed_whole[:, :-num_content]
            match_score = class_embed_whole_content @ class_embed_whole_generic.transpose(1, 2)
            decoder_output = decoder_output[:, :-num_content]  # remove content embedding
        outputs_class = match_score
        if self.prediction_switch['seg']:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        iou_score = self.iou_prediction_head(decoder_output).squeeze(-1).view(decoder_output.shape[0], -1,
                                                                              self.num_mask_tokens)

        return outputs_class, outputs_mask, iou_score, decoder_output, match_score

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class=None, outputs_seg_masks=None, out_boxes=None, predictions_iou_score=None,
                      predictions_class_part=None, predictions_match_score=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        elif outputs_seg_masks is None:
            return [
                {"pred_logits": a, "pred_boxes": c}
                for a, c in zip(outputs_class[:-1], out_boxes[:-1])
            ]
        elif predictions_match_score is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                for a, b, c in
                zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]
        elif predictions_iou_score is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_match_score": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_match_score[:-1])
            ]
        elif predictions_class_part is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_ious": d, "pred_match_score": e}
                for a, b, c, d, e in
                zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_iou_score[:-1], predictions_match_score[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_ious": d, "pred_logits_part": e,
                 "pred_match_score": f}
                for a, b, c, d, e, f in
                zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_iou_score[:-1],
                    predictions_class_part[:-1], predictions_match_score[:-1])
            ]


@register_decoder
def get_dinov_refer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return DINOvRefer(cfg, in_channels, lang_encoder, mask_classification, extra)
