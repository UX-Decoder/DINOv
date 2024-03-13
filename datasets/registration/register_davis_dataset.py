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


__all__ = ["load_davis_instances", "register_davis_context"]

def load_davis_instances(name: str, dirname: str, split: str, year: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    meta_txt = os.path.join(dirname, 'ImageSets', year, "{}.txt".format(split))
    meta_json = os.path.join(dirname, 'video_objects_info.json')
    meta_json = json.load(open(meta_json))['videos']
    video_names = [line.strip() for line in open(meta_txt).readlines()]

    video_dir = os.path.join(dirname, 'JPEGImages', '480p')
    mask_dir = os.path.join(dirname, 'Annotations', '480p')
    scibble_dir = os.path.join(dirname, 'Scribbles', '480p')
    semantic_dir = os.path.join(dirname, 'Annotations_semantics', '480p')

    dicts = []
    for vid_name in video_names:
        objects = meta_json[vid_name]['objects']
        r = {
            "file_name": os.path.join(video_dir, vid_name),
            "mask_name": os.path.join(mask_dir, vid_name),
            "scibble_name": os.path.join(scibble_dir, vid_name),
            "semantic_name": os.path.join(semantic_dir, vid_name),
            "objects": objects,
        }
        dicts.append(r)
    return dicts

def register_davis_context(name, dirname, split, year):
    if not os.path.exists(dirname):
        print("not register for ", name)
        return -1
    load_davis_instances(name, dirname, split, year)
    DatasetCatalog.register("{}".format(name), lambda: load_davis_instances(name, dirname, split, year))
    MetadataCatalog.get("{}".format(name)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )

def register_all_davis(root):
    SPLITS = [
            ("davis17_val", "DAVIS17", "val", "2017"),
            ("davis16_val", "DAVIS17", "val", "2016"),
        ]

    for name, dirname, split, year in SPLITS:
        register_davis_context(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get("{}".format(name)).evaluator_type = None

_root = os.getenv("TRACKING_DATASET", "datasets")
if _root!='datasets':
    register_all_davis(_root)
