if [ "$1" == "sup" ];
then 
    train="tools/train_net.py"
else
    train="tools/train_net_semi.py"
fi
if [ "$2" = "node_0" ];
then 
    rlaunch --gpu 8 --cpu 20 --memory 150000   --positive-tags 2080ti  -- python $train --config-file $3 --num-gpus 8 --machine-rank 0 --num-machines 2  --dist-url tcp://127.0.0.1:12345 
elif [ "$2"  = "node_1" ]
then
    ip=`cat tmpfile`
    rlaunch --gpu 8 --cpu 20 --memory 150000  --positive-tags 2080ti  -- python $train --config-file $3  --num-gpus 8 --machine-rank 1 --num-machines 2  --dist-url tcp://$ip:12345 
elif [ "$2"  = "one_node" ] #one node
then
    srun -c 8 --mem 100G --gres=gpu:8 python $train  --config-file $3   --num-gpus 8   train.amp.enabled=True dataloader.train.total_batch_size=16  
elif [ "$2"  = "half_node" ] #half node
then
    srun -c 8 --mem 120G --gres=gpu:4 python $train --config-file $3    --num-gpus 4  train.amp.enabled=True  
elif [ "$2"  = "resume" ] #one node
then
    rlaunch --gpu 8 --cpu 8 --memory 150000   --positive-tags 2080ti -- python $train  --config-file $ --num-gpus 8 --resume  dataloader.train.total_batch_size=8  
elif [ "$2"  = "eval" ] #one node
then
    srun -c 10 --mem 50G --gres=gpu:8 python $train  --config-file $3  --num-gpus 8  --resume --eval dataloader.test.mapper.is_train=True

elif [ "$2"  = "vis_data" ] #one node
then
    rlaunch --gpu 1 --cpu 1 --memory 150000   --positive-tags 2080ti -- python tools/visualize_data.py --source dataloader --config-file $3 --output-dir vis_dataloader
elif [ "$2"  = "vis_output" ] #one node
then
     srun -c 8 --mem 50G --gres=gpu:1 python tools/visualize_json_results.py  --output vis_output --input  $3
elif [ "$2"  = "debug" ] #debug
then
    srun -c 8 --mem 16G --gres=gpu:1 python $train  --config-file $3   --num-gpus 1  train.semi.burn_up_step=1  train.max_iter=9990000  dataloader.train.total_batch_size=1 dataloader.test.mapper.is_train=True dataloader.train.num_workers=0 #model.criterion.matcher.debug_mode=True
elif [ "$2"  = "debug_eval" ] #debug
then
    srun -c 8 --mem 16G --gres=gpu:1 python $train  --config-file $3 --eval --num-gpus 1 --resume   train.max_iter=9990000  dataloader.train.total_batch_size=1   dataloader.test.mapper.is_train=True #model.criterion.matcher.debug_mode=Tru
fi