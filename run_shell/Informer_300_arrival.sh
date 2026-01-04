#!/bin/bash

# 指定使用的 GPU 编号
export CUDA_VISIBLE_DEVICES=0

# 模型名称
model_name=Informer

# run.py 的绝对路径
run_script=/root/autodl-tmp/Time-Series-Library/run.py

# 数据根目录与子文件夹的路径配置
root_path_name=/root/autodl-tmp/Preprocessing/avg10_split300/
data_path_name=Arrival
# data_path_name=Departure

# 特征维度配置 (Lat, Lon, Altitude, Speed, Dir_sin, Dir_cos, hour_sin, min_sin 共 8 维)
enc_in_dim=8
dec_in_dim=8
c_out_dim=8

# 循环运行不同预测长度的实验 (24, 48, 96, 192)
for pred_len in 24 48 96 192
do
  python -u $run_script \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id Flight_${data_path_name}_96_${pred_len} \
    --model $model_name \
    --data trajectory \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $enc_in_dim \
    --dec_in $dec_in_dim \
    --c_out $c_out_dim \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 10 \
    --batch_size 512 \
    --patience 3
done