#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
    echo "Usage: $0 <num>"
    exit 1
fi

# 获取参数
NUM=$1

# 从 1000 到 NUM 的循环，步长为 1000
for ((i=1000; i<=$NUM; i+=1000)); do
    # 计算当前迭代使用的 GPU 设备
    GPU_ID=$((4 + (i / 1000 - 1) % 4))
    
    # 定义常量路径
    CKPT_DIR_BASE="/home/hdd2/zhanggangjian/FrontDiffuser_wids/outputs/sim2obc_ids_new_data_0714/global_step_"
    REFINE_INTER_RESULT_PATH="/home/hdd2/zhanggangjian/new_data/test/input_result_0608_aug/run_1_wids_stage2infer_${i}_vis"

    # 生成 checkpoint 目录
    CKPT_DIR="${CKPT_DIR_BASE}${i}"
    
    # 运行 Python 脚本并传入参数
    echo "Running with checkpoint directory: $CKPT_DIR on GPU $GPU_ID"
    echo "Refine intermediate result path: $REFINE_INTER_RESULT_PATH"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python refine.py --ckpt_dir "$CKPT_DIR" --refine_inter_result_path "$REFINE_INTER_RESULT_PATH" &
    
    # 控制并行进程数量为 4
    if (( $(jobs -r | wc -l) >= 4 )); then
        wait -n
    fi
done

# 等待所有后台进程完成
wait

echo "All processes completed."