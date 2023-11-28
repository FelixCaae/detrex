from .deformable_detr_r50_50ep import train, dataloader, optimizer, lr_multiplier, model
from detrex.evaluation import CrowdHumanEvaluator
from detectron2.config import LazyCall as L
# modify model config
model.with_box_refine = True
model.as_two_stage = True
lr_multiplier = get_config("common/crowdhuman_schedule.py").lr_multiplier_12ep
# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/ddetr-ts-r50-12ep-train"
dataloader.train.dataset.names = "crowdhuman_train"
dataloader.test.dataset.names = "crowdhuman_val"

train.max_iter = 937 * 12
train.eval_period = 3000
train.log_period = 20
train.checkpointer.period = 3000