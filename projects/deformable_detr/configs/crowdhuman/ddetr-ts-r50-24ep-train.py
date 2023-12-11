from .deformable_detr_r50_50ep import train, dataloader, optimizer, lr_multiplier, model
from detrex.evaluation import CrowdHumanEvaluator
from detectron2.config import LazyCall as L
from detrex.config import get_config
# modify model config
model.with_box_refine = True
model.as_two_stage = True
lr_multiplier = get_config("common/crowdhuman_schedule.py").lr_multiplier_24ep
# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/ddetr-ts-r50-24ep-train"
dataloader.train.dataset.names = "crowdhuman_train"
dataloader.test.dataset.names = "crowdhuman_val"
model.num_classes = 1
train.max_iter = 937 * 24
train.eval_period = 3000
train.log_period = 20
train.checkpointer.period = 3000

dataloader.evaluator = L(CrowdHumanEvaluator)( dataset_name="${..test.dataset.names}", visible_flag=False)
dataloader.evaluator.output_dir = train.output_dir
