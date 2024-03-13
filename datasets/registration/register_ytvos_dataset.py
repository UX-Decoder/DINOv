# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
import json
from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy.io import loadmat

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


__all__ = ["load_ytovs_instances", "register_ytvos_context"]

def load_ytvos_instances(name: str, dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    meta_json = os.path.join(dirname, split, "meta.json")
    video_dir = os.path.join(dirname, split, 'JPEGImages')
    mask_dir = os.path.join(dirname, split, 'Annotations')
    video_names = os.listdir(video_dir)
    meta = json.load(open(meta_json))['videos']

    dicts = []
    for vid_name in video_names:
        objects = meta[vid_name]['objects']
        r = {
            "file_name": os.path.join(video_dir, vid_name),
            "mask_name": os.path.join(mask_dir, vid_name),
            "objects": objects,
        }
        dicts.append(r)

    return dicts

def register_ytvos_context(name, dirname, split):
    if not os.path.exists(dirname):
        print("not register for ", name)
        return -1
    DatasetCatalog.register("{}".format(name), lambda: load_ytvos_instances(name, dirname, split))
    MetadataCatalog.get("{}".format(name)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )

def register_all_davis(root):
    SPLITS = [
            ("ytvos19_val", "ytvos2019", "valid"),
            ("ytvos18_val", "ytvos2018", "valid"),
        ]

    for name, dirname, split in SPLITS:
        register_ytvos_context(name, os.path.join(root, dirname), split)
        MetadataCatalog.get("{}".format(name)).evaluator_type = None

_root = os.getenv("TRACKING_DATASET", "datasets")
if _root!='datasets':
    register_all_davis(_root)