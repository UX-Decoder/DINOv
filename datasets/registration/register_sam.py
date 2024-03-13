# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_instance.py
# ------------------------------------------------------------------------------------------------

import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm
import torch.distributed as dist

import os.path as op

SAM_CATEGORIES = [{'id': 1, 'name': 'stuff'}]

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "sam_train": (
        "",
    ),
    "sam_val": (
        "",
    ),
}


def _get_sam_instances_meta():
    thing_ids = [k["id"] for k in SAM_CATEGORIES]
    assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SAM_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def load_sam_index(tsv_file, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.
    """
    dataset_dicts = []
    tsv_id = 0
    files = os.listdir(tsv_file)
    start = int(os.getenv("SAM_SUBSET_START", "90"))
    end = int(os.getenv("SAM_SUBSET_END", "100"))
    if len(files)>0 and 'part' in files[0]:  # for hgx
                files = [f for f in files if '.tsv' in f and int(f.split('.')[1].split('_')[-1])>=start and int(f.split('.')[1].split('_')[-1])<end]
    else:  # for msr
        files = [f for f in files if '.tsv' in f and int(f.split('.')[0].split('-')[-1])>=start and int(f.split('.')[0].split('-')[-1])<end]
        
    for tsv in files:
        if op.splitext(tsv)[1] == '.tsv':
            print('register tsv to create index', "tsv_id", tsv_id, tsv)
            lineidx = os.path.join(tsv_file, op.splitext(tsv)[0] + '.lineidx')
            line_name = op.splitext(tsv)[0] + '.lineidx'
            
            with open(lineidx, 'r') as fp:
                lines = fp.readlines()
                _lineidx = [int(i.strip().split()[0]) for i in lines]

            dataset_dict =[{'idx': (tsv_id, i)} for i in range(len(_lineidx))]
            dataset_dicts = dataset_dicts + dataset_dict
            tsv_id += 1
    return dataset_dicts

def register_sam_instances(name, metadata, tsv_file):
    assert isinstance(name, str), name

    DatasetCatalog.register(name, lambda: load_sam_index(tsv_file, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        tsv_file=tsv_file, evaluator_type="sam_interactive",  **metadata
    )


def register_all_sam_instance(root):
    for key, tsv_file in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_sam_instances(
            key,
            _get_sam_instances_meta(),
            os.path.join(root, tsv_file[0]),
        )

_root = os.getenv("SAM_DATASETS", "datasets")
# _root_local = os.getenv("SAM_LOCAL", "no")
# if _root_local != 'no' or 'comp_robot' in _root:
#     if 'comp_robot' not in _root:
#         _root = _root_local
if _root != 'no':
    print('registering sam datasets from', _root)
    register_all_sam_instance(_root)
