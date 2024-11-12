from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator


from detrex.data import DGDetrDatasetMapper, DetrDatasetMapper

dataloader = OmegaConf.create()
a = 768
b = 1111
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="sgod_dc_instance_train"),
    mapper=L(DGDetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, a),
                max_size=b,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, a),
                max_size=b,
                sample_style="choice",
            ),
        ],
        corruptions = [
            "defocus_blur",     "gaussian_blur","motion_blur",
            "speckle_noise", "shot_noise", "impulse_noise", "gaussian_noise",
            "jpeg_compression", "pixelate", "elastic_transform", "brightness", "saturate", "contrast",
            "high_pass_filter", "phase_scaling"
        ],    
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
 
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="sgod_dc_instance_val", filter_empty=False),
    mapper=L(DGDetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        corruptions=[],
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)