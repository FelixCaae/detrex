
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager
SGOD_CATEGORIES = [{'id': 0, 'name': 'bus'}, {'id': 1, 'name': 'bike'}, 
{'id': 2, 'name': 'car'}, {'id': 3, 'name': 'motor'},
 {'id': 4, 'name': 'person'}, {'id': 5, 'name': 'rider'}, {'id': 6, 'name': 'truck'}]
 
_PREDEFINED_SPLITS = {
    # point annotations without masks
    "sgod_dc_instance_train": (
        "crossdomain_urban/daytime_clear",
        "crossdomain_urban/annotations/daytime_train.json",
    ),
    "sgod_dc_instance_val": (
        "crossdomain_urban/daytime_clear",
        "crossdomain_urban/annotations/daytime_val.json",
    ),
    "sgod_nc_instance_val": (
        "crossdomain_urban/night_sunny",
        "crossdomain_urban/annotations/night_sunny_val.json",
    ),
    "sgod_nr_instance_val": (
        "crossdomain_urban/night_rainy",
        "crossdomain_urban/annotations/night_rainy_val.json",
    ),
    "sgod_dr_instance_val": (
        "crossdomain_urban/dusk_rainy",
        "crossdomain_urban/annotations/dusk_rainy_val.json",
    ),
    "sgod_df_instance_val": (
        "crossdomain_urban/daytime_foggy",
        "crossdomain_urban/annotations/daytime_foggy_val.json",
    ),

}


def _get_sgod_instances_meta():
    thing_ids = [k["id"] for k in SGOD_CATEGORIES]
    assert len(thing_ids) == 7, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SGOD_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_sgod_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_sgod_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_sgod_instance(_root)
