# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------
"""
DINOv Training Script based on Semantic-SAM.
"""
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import time
from typing import Any, Dict, List, Set
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog

from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig, instantiate
from utils.misc import init_wandb
import wandb

from datasets import (
    build_train_dataloader,
    build_evaluator,
    build_eval_dataloader,

)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref

from dinov import build_model
from dinov.BaseModel import BaseModel

from utils.misc import hook_metadata, hook_switcher, hook_opt
from dinov.utils import get_class_names
from detectron2.utils.logger import log_every_n_seconds
import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class Ped to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg['OUTPUT_DIR'],
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg['SOLVER']['MAX_ITER']
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg['OUTPUT_DIR'],
            **kwargs,
        )

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = copy.deepcopy(self.cfg)
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=1))
        return ret

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = BaseModel(cfg, build_model(cfg)).cuda()
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_train_dataloader(cfg, )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        loader = build_eval_dataloader(cfg, )
        return loader

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        cfg_solver = cfg['SOLVER']
        weight_decay_norm = cfg_solver['WEIGHT_DECAY_NORM']
        weight_decay_embed = cfg_solver['WEIGHT_DECAY_EMBED']
        weight_decay_bias = cfg_solver.get('WEIGHT_DECAY_BIAS', 0.0)

        defaults = {}
        defaults["lr"] = cfg_solver['BASE_LR']
        defaults["weight_decay"] = cfg_solver['WEIGHT_DECAY']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        lr_multiplier = cfg['SOLVER']['LR_MULTIPLIER']
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                for key, lr_mul in lr_multiplier.items():
                    if key in "{}.{}".format(module_name, module_param_name):
                        hyperparams["lr"] = hyperparams["lr"] * lr_mul
                        if comm.is_main_process():
                            logger.info("Modify Learning rate of {}: {}".format(
                                "{}.{}".format(module_name, module_param_name), lr_mul))

                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                if "bias" in module_name:
                    hyperparams["weight_decay"] = weight_decay_bias
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg_solver['CLIP_GRADIENTS']['CLIP_VALUE']
            enable = (
                    cfg_solver['CLIP_GRADIENTS']['ENABLED']
                    and cfg_solver['CLIP_GRADIENTS']['CLIP_TYPE'] == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg_solver['OPTIMIZER']
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg_solver['BASE_LR'], momentum=cfg_solver['MOMENTUM']
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg_solver['BASE_LR']
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = copy.deepcopy(cfg)

        assert (
                cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )
        return cfg

    @classmethod
    def test_tracking_prev(cls, cfg, model, evaluators=None):
        import numpy as np
        from torch.nn import functional as F
        from PIL import Image
        from queue import Queue, LifoQueue, PriorityQueue
        cfg['DATASETS']['TEST'] = ['davis17_val']
        maxsize = cfg['MODEL']['DECODER']['MAX_MEMORY_SIZE']
        dataloaders = cls.build_test_loader(cfg, dataset_name=None)
        model = model.eval().cuda()

        output_root = cfg['OUTPUT_DIR']
        save_score = cfg['MODEL'].get('save_score', False)
        preix = cfg['MODEL']['WEIGHTS'].split('/')[-1]
        output_pth = os.path.join(output_root, preix + '_Predictions2017')

        with torch.no_grad() and torch.autocast(device_type='cuda', dtype=torch.float16):
            for dataloader in dataloaders:
                for idx, video in enumerate(dataloader):
                    print(idx, len(dataloader), video[0].vid_name)
                    assert len(video) == 1
                    # find the first video frame that contains 'instances'
                    cur_idx = 0
                    for i in range(len(video[0])):
                        if 'instances' in video[0][i]:
                            cur_idx = i
                            break
                    rand_shape = video[0][cur_idx]['instances'].gt_masks.tensor[:, None] & False
                    acc_key = video[0][0]['key_frame'] & False
                    end_key = video[0][0]['key_frame'] & False

                    frame = video[0][0]
                    frame['targets'] = [dict()]
                    memory_content_label = [PriorityQueue(maxsize=maxsize) for i in range(len(rand_shape))]
                    ious = None
                    for fid, frame2 in enumerate(video[0]):
                        if frame2['key_frame'].sum() > 0:
                            # a list of True/False for all instance. only the first one's key frame are True
                            rand_shape[frame2['key_frame']] = frame2['instances'].gt_masks.tensor[:, None][
                                frame2['key_frame']]  # ReTrack when the object disapper in some frames
                            rand_shape = rand_shape.squeeze(1)
                            print("fid, video[0].vid_name ", fid, video[0].vid_name, frame2['key_frame'])
                        frame['targets'][0]['rand_shape'] = rand_shape
                        frame2['targets'] = frame['targets']
                        batched_inputs = [frame]
                        batched_inputs2 = [frame2]
                        multi_scale_features2, mask_features2, _, _ = model.model.get_encoder_feature(batched_inputs2)
                        if fid==0 or ious[i]>-1.0:  # only put these with large confidence score
                            multi_scale_features, _, padded_h, padded_w = model.model.get_encoder_feature(batched_inputs)
                            input_query_label_content, input_query_bbox_content, attn_mask_content = model.model. \
                                get_visual_prompt_content_feature(multi_scale_features, rand_shape, padded_h, padded_w)
                            # focus on the most recent frame; record with a score
                            score = fid if fid else 10000  # always keep the reference first frame
                            if input_query_label_content.shape[1]>len(memory_content_label):
                                input_query_label_content = input_query_label_content.view(input_query_label_content.shape[0], len(memory_content_label), -1, input_query_label_content.shape[-1])
                            for i, query in enumerate(input_query_label_content.squeeze(0)):

                                if memory_content_label[i].full() and len(memory_content_label[i].queue)==1:
                                    continue
                                if memory_content_label[i].full():
                                    a = memory_content_label[i].get()

                                memory_content_label[i].put((score, query.detach().clone()))
                                # frame = frame2
                            torch.cuda.empty_cache()
                        frame = frame2
                        # record the current instance
                        input_query_label_content_current = []
                        for instance_memory_content in memory_content_label:
                            instance_memory_content_current = torch.zeros_like(instance_memory_content.queue[0][-1]).to(instance_memory_content.queue[0][-1])
                            for q in range(len(instance_memory_content.queue)):
                                # combine with memory
                                instance_memory_content_current = instance_memory_content_current + instance_memory_content.queue[q][-1].detach().clone()
                            instance_memory_content_current = instance_memory_content_current/len(instance_memory_content.queue)
                            input_query_label_content_current.append(instance_memory_content_current)
                        input_query_label_content_current = torch.stack(input_query_label_content_current)[None]
                        if len(input_query_label_content_current.shape)>3:
                            input_query_label_content_current = input_query_label_content_current.flatten(1,2)
                        masks, ious, ori_masks = model.model.evaluate_visual_prompt_refer_multi_with_content_features(
                            batched_inputs2, mask_features2, multi_scale_features2, input_query_label_content_current,
                            input_query_bbox_content, attn_mask_content, padded_h, padded_w)

                        acc_key = acc_key | frame['key_frame']
                        if fid:  # use ground thruth first frame mask for the second frame
                            rand_shape = ori_masks > 0.0
                        if fid == 0:  # use the given first frame as gt
                            height = batched_inputs[0].get('height', -1)
                            width = batched_inputs[0].get('width', -1)
                            masks = F.interpolate(
                                rand_shape[None].float(),
                                size=(height, width),
                                mode="bilinear",
                                align_corners=False,
                            )[0]
                        output_mask = (
                                (masks>0).cpu() & acc_key[:, None, None] & (~end_key[:, None, None])).numpy()

                        output_mask_numpy = np.zeros(output_mask.shape[1:])
                        for i in range(output_mask.shape[0]):
                            output_mask_numpy[output_mask[i]] = video[0].mappers[i]

                        output_image = Image.fromarray(output_mask_numpy).convert('P')
                        output_image.putpalette(video[0].palette)

                        output_folder = os.path.join(output_pth, video[0].vid_name)
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        if not save_score:
                            output_image.save(os.path.join(output_folder, "{}.png".format(frame['frame_id'])))
                        else:
                            output_image.save(os.path.join(output_folder, "{}_{}.png".format(frame['frame_id'], str(ious.cpu().detach().numpy()))))

                        end_key = end_key | frame['end_frame']

    @classmethod
    def test_save_features(cls, cfg, model, evaluators=None):
        # build dataloader
        dataloaders = cls.build_test_loader(cfg, dataset_name=None)
        dataset_names = cfg['DATASETS']['TEST']
        weight_path = cfg['MODEL']['WEIGHTS']
        ckpt = weight_path.split('/')
        # output_dir_ = cfg['OUTPUT_DIR']+'_'+ckpt[-1]
        output_dir_ = cfg['OUTPUT_DIR']
        if comm.is_main_process() and not os.path.exists(output_dir_):
            os.mkdir(output_dir_)
        model = model.eval().cuda()
        model_without_ddp = model
        if not type(model) == BaseModel:
            model_without_ddp = model.module
        for dataloader, dataset_name in zip(dataloaders, dataset_names):
            print("begin inference ", dataset_name)
            if 'seginw' in dataset_name:
                dir_name = dataset_name.split('_')[1]
            else:
                dir_name = dataset_name.replace('train', 'val')
            output_dir = os.path.join(output_dir_, dir_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            with torch.no_grad():
                # setup model
                hook_switcher(model_without_ddp, dataset_name)
                # setup timer
                total = len(dataloader)
                num_warmup = min(5, total - 1)
                total_data_time = 0
                start_data_time = time.perf_counter()

                for idx, batch in enumerate(dataloader):
                    if batch[0]['instances'].gt_boxes.tensor.shape[0]<1:
                        continue
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        total_data_time = 0
                    # forward
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        task = 'get_content'
                        input_tokens_all, labels = model(batch, inference_task=task, dataset_name=dataset_name)
                    image_id = batch[0]['image_id']
                    from safetensors.torch import save_file
                    label_dict = {l: 0 for l in list(set(labels.cpu().numpy()))}
                    labels =labels.cpu().numpy()
                    for label, embedding in zip(labels, input_tokens_all):
                        label_dict[label] += 1
                        save_dict = {}
                        save_dict['embedding'] = embedding
                        save_cate_folder = os.path.join(output_dir, str(label))
                        save_path = os.path.join(save_cate_folder, 'id_{}_idx_{}.safetensors'.format(image_id, label_dict[label]))
                        if not os.path.exists(save_cate_folder):
                            os.system(f'mkdir -p {save_cate_folder}')
                        save_file(save_dict, save_path)

                    print(idx)

    @classmethod
    def test_visual_openset(cls, cfg, model, evaluators=None):
        # build dataloade
        dataloaders = cls.build_test_loader(cfg, dataset_name=None)
        dataset_names = cfg['DATASETS']['TEST']
        model = model.eval().cuda()
        model_without_ddp = model
        if not type(model) == BaseModel:
            model_without_ddp = model.module
        # score list
        score_mask_ap = {}
        score_box_ap = {}
        output_dir_ = cfg['OUTPUT_DIR']
        for dataloader, dataset_name in zip(dataloaders, dataset_names):
            print("begin evaluate ", dataset_name)
            # prepare for seginw
            if 'seginw' in dataset_name:
                dir_name = dataset_name.split('_')[1]
            else:
                dir_name = dataset_name.replace('train', 'val')
            # output_dir = output_dir_
            output_dir = os.path.join(output_dir_, dir_name)
            model_without_ddp.model.sem_seg_head.predictor.out_dir = output_dir
            # build evaluator
            evaluator = build_evaluator(cfg, dataset_name, cfg['OUTPUT_DIR'])
            evaluator.reset()
            with torch.no_grad():
                # setup model
                if 'odinw' in dataset_name:
                    names = MetadataCatalog.get(dataset_name).thing_classes
                else:
                    names = get_class_names(dataset_name, cfg['MODEL'].get('BACKGROUND', True))
                model_without_ddp.model.metadata = MetadataCatalog.get(dataset_name)
                if 'background' in names:
                    model_without_ddp.model.sem_seg_head.num_classes = len(names) - 1
                else:
                    model_without_ddp.model.sem_seg_head.num_classes = len(names)
                # HACK for inference random select query
                cat_dirs = os.listdir(output_dir)
                cat_dirs = [int(cat) for cat in cat_dirs if os.path.isdir(os.path.join(output_dir, cat))]
                cat_dirs.sort()
                model_without_ddp.model.metadata.set(cat_dirs=cat_dirs)
                task = 'visual_openset'
                hook_switcher(model_without_ddp, dataset_name)

                # setup timer
                total = len(dataloader)
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
                start_data_time = time.perf_counter()

                for idx, batch in enumerate(dataloader):
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0
                    start_compute_time = time.perf_counter()

                    # forward
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # FIXME hack for visual prompt
                        # task = 'inference_select'
                        outputs = model(batch, inference_task=task, dataset_name=dataset_name)

                    total_compute_time += time.perf_counter() - start_compute_time
                    start_eval_time = time.perf_counter()

                    evaluator.process(batch, outputs)
                    total_eval_time += time.perf_counter() - start_eval_time

                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    eval_seconds_per_iter = total_eval_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

                    if comm.is_main_process() and (idx >= num_warmup * 2 or compute_seconds_per_iter > 5):
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            n=5,
                        )
                    start_data_time = time.perf_counter()

            # evaluate
            print("gather results for ", dataset_name)
            results = evaluator.evaluate()
            print("dataset_name, results ", dataset_name, results)
            if comm.is_main_process():
                if 'seginw' in dataset_name or 'odinw' in dataset_name:
                    if 'seginw' in dataset_name:
                        score_mask_ap[dataset_name.split('_')[1]] = results['segm']['AP']
                    # score_box_ap[dataset_name.split('_')[1]] = results['bbox']['AP']
                    score_box_ap[dataset_name] = results['bbox']['AP']
                    print("score_mask_ap ", score_mask_ap)
                    print("score_box_ap ", score_box_ap)
                    lent = len(list(score_box_ap.values()))
                    if 'seginw' in dataset_name:
                        print("score_mask_ap ", sum(list(score_mask_ap.values()))/lent)
                    print("score_box_ap ", sum(list(score_box_ap.values()))/lent)
        model = model.train().cuda()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args=None):
    cfg = setup(args)
    print("Command cfg:", cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if args.eval_visual_openset:
            res = Trainer.test_visual_openset(cfg, model)
        elif args.eval_track_prev:
            res = Trainer.test_tracking_prev(cfg, model)
        elif args.eval_get_content_features:
            res = Trainer.test_save_features(cfg, model)
        else:
            res = Trainer.test(cfg, model)
        return res

    if comm.get_rank() == 0 and args.WANDB:
        wandb.login(key=args.wandb_key)
        init_wandb(cfg, cfg['OUTPUT_DIR'], entity=args.wandb_usr_name, job_name=cfg['OUTPUT_DIR'])

    trainer = Trainer(cfg)

    print("load pretrained model weight!!!!!!")
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_visual_openset', action='store_true')
    parser.add_argument('--eval_track_prev', action='store_true')
    parser.add_argument('--eval_get_content_features', action='store_true')
    parser.add_argument('--WANDB', action='store_true')
    parser.add_argument('--wandb_usr_name', type=str, default='')
    parser.add_argument('--wandb_key', type=str, default='')
    args = parser.parse_args()
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
