from detectron2.data.datasets import load_coco_json
import os
from detectron2.data  import DatasetCatalog, MetadataCatalog,get_detection_dataset_dicts
def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    extra_annotation_keys= ['score'] if 'generated' in name else  None
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name, extra_annotation_keys=extra_annotation_keys))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

_PREDEFINED_SPLITS_CROWD_HUMAN=[
    ("crowdhuman_train", 'CrowdHuman/train.json', "CrowdHuman/annotation_train.odgt",'CrowdHuman/Images'),
     ("crowdhuman_trainvis", 'CrowdHuman/train_visible.json', "CrowdHuman/annotation_train.odgt",'CrowdHuman/Images'),
    ("crowdhuman_trainvis_generated", 'CrowdHuman/train_visible_gen_thresh01.json', "CrowdHuman/annotation_train.odgt",'CrowdHuman/Images'),
    ("crowdhuman_train_generated","CrowdHuman/train_generated_full.json","CrowdHuman/annotation_train.odgt","CrowdHuman/Images"),   
    ("crowdhuman_train_partial_001_visible", 'CrowdHuman/partial_annotations/0.01_partial_train.json', "CrowdHuman/annotation_train.odgt",'CrowdHuman/Images'),
    ("crowdhuman_train_partial_01_visible", 'CrowdHuman/partial_annotations/0.1_partial_train.json', "CrowdHuman/annotation_train.odgt",'CrowdHuman/Images'),
      ("crowdhuman_train_partial_05_visible", 'CrowdHuman/partial_annotations/0.5_partial_train.json', "CrowdHuman/annotation_train.odgt",'CrowdHuman/Images'),
    ("crowdhuman_val", 'CrowdHuman/val.json', "CrowdHuman/annotation_val.odgt",'CrowdHuman/Images'),
    ("crowdhuman_midvalvis", 'CrowdHuman/midval_visible.json', "CrowdHuman/annotation_val.odgt",'CrowdHuman/Images'),
    ("crowdhuman_valvis", 'CrowdHuman/val_visible.json', "CrowdHuman/annotation_val.odgt",'CrowdHuman/Images'),
   
    # ("crowdhuman_minival", 'CrowdHuman/minival.json', 'CrowdHuman/Images'),
 ]

CROWDHUMAN_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
]

def get_metadata_for_crowdhuman(odgt_file):
    thing_ids = [k["id"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
    
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "odgt_file": odgt_file,
    }
    return ret
def register_crowdhuman(root):
    for key, json_file , odgt_file, image_root in _PREDEFINED_SPLITS_CROWD_HUMAN:
        register_coco_instances(
            key,
            get_metadata_for_crowdhuman( os.path.join(root,odgt_file)),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_crowdhuman(_root)


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from detectron2.data.datasets import load_coco_json
    import sys
    from PIL import Image
    import numpy as np

    logger = setup_logger(name=__name__)

    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])
    dataset= DatasetCatalog.get(sys.argv[3])
    
    dataset=get_detection_dataset_dicts('crowdhuman_train')
    dataset=get_detection_dataset_dicts('crowdhuman_val')
    dataset=get_detection_dataset_dicts('crowdhuman_train_noise')
    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))
    print(meta)

    dirname = sys.argv[4]
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        print(d)
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
