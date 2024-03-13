# Copyright (c) Facebook, Inc. and its affiliates.
from . import (
    register_ade20k_full,
    register_ade20k_panoptic,
    register_coco_stuff_10k,
    register_coco_panoptic_annos_semseg,
    register_coco_panoptic_annos_semseg_interactive,
    register_coco_panoptic_annos_semseg_interactive_jointboxpoint,
    register_ade20k_instance,
    register_sam,
    register_sunrgbd_semseg,
    register_scannet_semseg,
    register_bdd100k_semseg,
    register_scannet_panoptic,
    register_bdd100k_panoseg,
    register_object365_od,
    register_pascal_part_all,
    register_pascal_part_all_interactive,
    register_paco_part_all,
    register_partimagenet_part_all,
)

from . import (
    register_ytvos_dataset,
    register_davis_dataset,
    register_seginw_instance,
    register_lvis_eval,
    register_context_semseg,
    register_odinw_od,
)