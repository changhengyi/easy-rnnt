echo ============================================================================
echo "                      CHINESE ASR WITH 1.7W HOURS                         "
echo ============================================================================

ps aux | grep eval.py | awk '{print $2}' | xargs kill -9
echo ============================================================================
echo "                           ASR Training Start                             "
echo ============================================================================
gpu=0,1,2,3,4,5,6,7
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

export OMP_NUM_THREADS=${n_gpus}
CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=${n_gpus} --nnodes=1 --node_rank=0 \
    /home/changhengyi/easy-rnnt/rnnt/bin/train.py \
    local_world_size=${n_gpus} \
    config=/home/changhengyi/easy-rnnt/examples/chinese/conf.yaml
    

echo "Finish ASR model training"
