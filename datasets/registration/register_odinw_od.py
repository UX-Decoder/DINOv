# Copyright (c) Facebook, Inc. and its affiliates.
import json
import yaml
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

YAML_FILE_LIST = [
        "AerialMaritimeDrone_large.yaml",
        "AerialMaritimeDrone_tiled.yaml",
        "AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco.yaml",
        "Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml",
        "BCCD_BCCD.v3-raw.coco.yaml",
        "ChessPieces_Chess_Pieces.v23-raw.coco.yaml",
        "CottontailRabbits.yaml",
        "DroneControl_Drone_Control.v3-raw.coco.yaml",
        "EgoHands_generic.yaml",
        "EgoHands_specific.yaml",
        "HardHatWorkers_raw.yaml",
        "MaskWearing_raw.yaml",
        "MountainDewCommercial.yaml",
        "NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco.yaml",
        "OxfordPets_by-breed.yaml",
        "OxfordPets_by-species.yaml",
        "PKLot_640.yaml",
        "Packages_Raw.yaml",
        "PascalVOC.yaml",
        "Raccoon_Raccoon.v2-raw.coco.yaml",
        "ShellfishOpenImages_raw.yaml",
        "ThermalCheetah.yaml",
        "UnoCards_raw.yaml",
        "VehiclesOpenImages_416x416.yaml",
        "WildfireSmoke.yaml",
        "boggleBoards_416x416AutoOrient_export_.yaml",
        "brackishUnderwater_960x540.yaml",
        "dice_mediumColor_export.yaml",
        "openPoetryVision_512x512.yaml",
        "pistols_export.yaml",
        "plantdoc_416x416.yaml",
        "pothole.yaml",
        "selfdrivingCar_fixedLarge_export_.yaml",
        "thermalDogsAndPeople.yaml",
        "websiteScreenshots.yaml"
    ]

ODINW_35_DATASETS = ['AerialMaritimeDrone_large', 'AerialMaritimeDrone_tiled', 'AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco', 'Aquarium_Aquarium_Combined.v2-raw-1024.coco', 'BCCD_BCCD.v3-raw.coco', 'ChessPieces_Chess_Pieces.v23-raw.coco', 'CottontailRabbits', 'DroneControl_Drone_Control.v3-raw.coco', 'EgoHands_generic', 'EgoHands_specific', 'HardHatWorkers_raw', 'MaskWearing_raw', 'MountainDewCommercial', 'NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco', 'OxfordPets_by-breed', 'OxfordPets_by-species', 'PKLot_640', 'Packages_Raw', 'PascalVOC', 'Raccoon_Raccoon.v2-raw.coco', 'ShellfishOpenImages_raw', 'ThermalCheetah', 'UnoCards_raw', 'VehiclesOpenImages_416x416', 'WildfireSmoke', 'boggleBoards_416x416AutoOrient_export_', 'brackishUnderwater_960x540', 'dice_mediumColor_export', 'openPoetryVision_512x512', 'pistols_export', 'plantdoc_416x416', 'pothole', 'selfdrivingCar_fixedLarge_export_', 'thermalDogsAndPeople', 'websiteScreenshots']

_PREDEFINED_SPLITS_ODINW = {}
def load_image_annotation_root_ori(root):
    preifx = 'data/odinw/DataDownload/detection/original'
    for dataset in ODINW_35_DATASETS:
        with open(os.path.join(root, "data/odinw/DataDownload/detection/odinw_35/{}.yaml".format(dataset)), 'r') as f:
            info = yaml.safe_load(f)
        _PREDEFINED_SPLITS_ODINW.update(
            {"odinw_{}_val".format(dataset): (
                os.path.join(preifx, info['DATASETS']['REGISTER']['test']['img_dir'][6:]),    # remove the odinw start
                os.path.join(preifx, info['DATASETS']['REGISTER']['test']['ann_file'][6:]),
            )}
        )

def load_image_annotation_root(root):
    preifx = 'odinw/detection/DATASET/odinw'
    for dataset in ODINW_35_DATASETS:
        with open(os.path.join(root, "odinw/detection/odinw_35/{}.yaml".format(dataset)), 'r') as f:
            info = yaml.safe_load(f)
        _PREDEFINED_SPLITS_ODINW.update(
            {"odinw_{}_val".format(dataset): (
                os.path.join(preifx, info['DATASETS']['REGISTER']['test']['img_dir'][6:]),    # remove the odinw start
                os.path.join(preifx, info['DATASETS']['REGISTER']['test']['ann_file'][6:]),
            )}

        )
    for dataset in ODINW_35_DATASETS:
        with open(os.path.join(root, "odinw/detection/odinw_35/{}.yaml".format(dataset)), 'r') as f:
            info = yaml.safe_load(f)
        _PREDEFINED_SPLITS_ODINW.update(
            {"odinw_{}_train".format(dataset): (
                os.path.join(preifx, info['DATASETS']['REGISTER']['train']['img_dir'][6:]),    # remove the odinw start
                os.path.join(preifx, info['DATASETS']['REGISTER']['train']['ann_file'][6:]),
            )}

        )

ODINW_CATEGORIES = {}

def get_odinw_metadata(od_json):
    with PathManager.open(od_json) as f:
        json_info = json.load(f)
    ODINW_CATEGORIES = json_info['categories']
    thing_ids = [k["id"] for k in ODINW_CATEGORIES]
    # assert len(thing_ids) == 365, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ODINW_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret
    # meta = {}
    # return meta


def load_odinw_json(od_json, image_root, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            return True
        else:
            return False
    with PathManager.open(od_json) as f:
        json_info = json.load(f)
    imgid2bbox = {x['id']: [] for x in json_info['images']}
    imgid2cat = {x['id']: [] for x in json_info['images']}
    imgid2ann = {x['id']: [] for x in json_info['images']}

    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in imgid2bbox and _convert_category_id(ann, meta):
            imgid2bbox[image_id] += [ann["bbox"]]
            imgid2cat[image_id] += [ann["category_id"]]
            imgid2ann[image_id] += [ann]

    imgid2pth = {}
    for image_info in json_info['images']:
        imgid2pth[image_info['id']] = image_info['file_name']

    ret = []
    for image_info in json_info['images']:
        image_id = int(image_info["id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_root, image_info['file_name'])

        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "bbox": imgid2bbox[image_id],
                "categories": imgid2cat[image_id],
                "annotations": imgid2ann[image_id],
            }
        )

    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_odinw_od(
    name, metadata, image_root, od_json
):
    DatasetCatalog.register(
        name,
        lambda: load_odinw_json(od_json, image_root, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=od_json,
        evaluator_type="odinw_od",
        **metadata,
    )


def register_all_object365_od(root_json, root_image):
    load_image_annotation_root(root_json)
    for (
        prefix,
        (image_root, od_json),
    ) in _PREDEFINED_SPLITS_ODINW.items():
        register_odinw_od(
            prefix,
            get_odinw_metadata(os.path.join(root_json, od_json)),
            os.path.join(root_image, image_root),
            os.path.join(root_json, od_json),
        )


_root_json = os.getenv("DATSETW", "datasets")
_root_image = os.getenv("DATSETW", "datasets")
print("_root_json, _root_image", _root_json, _root_image)
if _root_json!='datasets' and _root_image!='datasets':
    register_all_object365_od(_root_json, _root_image)
