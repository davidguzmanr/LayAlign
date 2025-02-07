#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
random_port(){
    # Random port
    MASTER_PORT=$((30000 + RANDOM % (99999-30000+1)))
    echo "MASTER_PORT=$MASTER_PORT"
}



export_world_info() {
    # Set world info for deepspeed
    # ref: https://github.com/microsoft/DeepSpeed/issues/1331
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "CUDA_VISIBLE_DEVICES is not set"
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        # generate GPUS from index 0
        CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
        echo "Use all GPUs"
        export "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        WID=`echo  {\"localhost\": [$CUDA_VISIBLE_DEVICES]} | base64`
    else
        # count CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        WID=`echo  {\"localhost\": [$CUDA_VISIBLE_DEVICES]} | base64`
    fi
}
random_port
export_world_info
source activate multi-lingual
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
accelerate launch \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero2_dp.json \
    finetune.py \
    --llm_path LLaMAX/LLaMAX2-7B-XNLI \
    --mt_path google/mt5-xl \
    --stage_name mapping \
    --task xnli \
    --train_num 100000 \
    --train_batch_size $TOTAL_BATCH_SIZE \
    --train_micro_batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --gradient_accumulation $GRADIENT_ACC_STEPS \
    --augmentation False \
    --epoch_num 3 \
    --max_seq_len 200 \
    --max_gen_len 200 \
    --eval_batch_size 4 \
    --dev_size 3000 \
    --acc_cal_step 200 \
    --logging_steps 10 \
    --lr 4e-5 \
    --save_name LayAlign-xnli \
    --warm_rate 0.05 \
    --structure Linear \
    --lr_scheduler_name cosine \



BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

accelerate launch \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero2_dp.json \
    finetune.py \
    --llm_path LLaMAX/LLaMAX2-7B-XNLI \
    --mt_path google/mt5-xl \
    --stage_name augmentation \
    --task xnli \
    --train_num 30000 \
    --train_batch_size $TOTAL_BATCH_SIZE \
    --train_micro_batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --gradient_accumulation $GRADIENT_ACC_STEPS \
    --augmentation True \
    --epoch_num 3 \
    --max_seq_len 200 \
    --max_gen_len 200  \
    --eval_batch_size $BATCH_SIZE_PER_GPU \
    --dev_size 3000 \
    --logging_steps 10 \
    --lr 3e-5 \
    --save_name LayAlign-xnli-test1 \
    --warm_rate 0.05 \
    --structure Linear \
    --lr_scheduler_name cosine \



