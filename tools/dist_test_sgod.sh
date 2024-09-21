#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
echo 'start testing on night_clear'
python  tools/train_net.py --eval --config-file $1 --num-gpus 8  train.init_checkpoint=$2  dataloader.test.dataset.names=sgod_nc_instance_val | grep 'AP'
echo 'start testing on dusk rainy weather..'
python  tools/train_net.py --eval --config-file $1 --num-gpus 8  train.init_checkpoint=$2  dataloader.test.dataset.names=sgod_dr_instance_val | grep 'AP'
echo 'start testing on night rainy weather..'
python  tools/train_net.py --eval --config-file $1 --num-gpus 8  train.init_checkpoint=$2  dataloader.test.dataset.names=sgod_nr_instance_val | grep 'AP'
echo 'start testing on foggy weather..'
python  tools/train_net.py --eval --config-file $1 --num-gpus 8  train.init_checkpoint=$2  dataloader.test.dataset.names=sgod_df_instance_val | grep 'AP'

# bash tools/dist_test_foggy.sh $CONFIG $CHECKPOINT $GPUS 