from detrex.config import get_config
from ..models.dino_r50 import model
from detrex.evaluation import CrowdHumanEvaluator
from detectron2.config import LazyCall as L
# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/crowdhuman_schedule.py").lr_multiplier_24ep
train = get_config("common/train.py").train
dataloader.train.dataset.names = "crowdhuman_trainvis_generated"
dataloader.test.dataset.names = "crowdhuman_val"
model.num_classes = 1
# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/dino_r50_4scale_24ep_trainvis_gen"

# max training iterations
train.max_iter = 937 * 24
train.eval_period = 3000
train.log_period = 100
train.checkpointer.period = 3000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 8

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator = L(CrowdHumanEvaluator)( dataset_name="${..test.dataset.names}",visible_flag=False)
dataloader.evaluator.output_dir = train.output_dir
