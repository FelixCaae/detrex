from detrex.config import get_config
from ..models.dgdino_r50 import model

dataloader = get_config("common/data/sgod_detr_corrupt.py").dataloader
lr_multiplier = get_config("common/sgod_schedule.py").lr_multiplier_20e
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/sgod_dinoaug2_distill_dino_t4_r50_20ep"
dataloader.test.dataset.names="sgod_dc_instance_val"
# dataloader.test.dataset.names="sgod_nc_instance_val"
# dataloader.test.dataset.names="sgod_dr_instance_val"
# dataloader.test.dataset.names="sgod_nr_instance_val"
# dataloader.test.dataset.names="sgod_df_instance_val"

model.distill_dino = True
model.num_classes = 7
model.criterion.num_classes = 7
# max training iterations
train.max_iter = 25000

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8
# dataloader.test.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
