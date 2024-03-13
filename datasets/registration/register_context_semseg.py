# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from utils.constants import PASCAL_CONTEXT_459, PASCAL_CONTEXT_59, PASCAL_CONTEXT_33

__all__ = ["load_context_instances", "register_pascal_context"]
dataset2class = {"context_459_val_seg": PASCAL_CONTEXT_459,
                 "context_59_val_seg": PASCAL_CONTEXT_59}
dataset2labelfolder = {"context_459_val_seg": "trainval",
                       "context_59_val_seg": "59_context_labels"}
dataset2postfix = {"context_459_val_seg": ".mat",
                   "context_59_val_seg": ".png"}
dataset2segloader = {"context_459_val_seg": "MAT",
                     "context_59_val_seg": "PIL"}


def load_context_instances(name: str, dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "VOC2010", "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    image_dirname = PathManager.get_local_path(os.path.join(dirname, "VOC2010"))
    semseg_dirname = PathManager.get_local_path(os.path.join(dirname, dataset2labelfolder[name]))

    dicts = []
    for fileid in fileids:
        jpeg_file = os.path.join(image_dirname, "JPEGImages", fileid + ".jpg")
        seg_file = os.path.join(semseg_dirname, fileid + dataset2postfix[name])

        r = {
            "file_name": jpeg_file,
            "sem_seg_file_name": seg_file,
            "image_id": fileid,
        }
        dicts.append(r)
    return dicts


def register_pascal_context(name, dirname, split, year, class_names=dataset2class):
    DatasetCatalog.register(name, lambda: load_context_instances(name, dirname, split, class_names))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names[name],
        dirname=dirname,
        year=year,
        split=split,
        ignore_label=[0],
        thing_dataset_id_to_contiguous_id={},
        class_offset=1,
        semseg_loader=dataset2segloader[name],
        keep_sem_bgd=False
    )


def register_all_context_seg(root):
    SPLITS = [
        ("context_459_val_seg", "pascal_context", "val"),
        ("context_59_val_seg", "pascal_context", "val"),
    ]
    year = 2010
    for name, dirname, split in SPLITS:
        register_pascal_context(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


_root = os.getenv("DATSETW", "datasets")
if _root!='datasets':
    register_all_context_seg(_root)