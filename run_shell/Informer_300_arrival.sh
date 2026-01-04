#！/bin/bash

# 指定使用的 GPU 编号 (AutoDL 默认为 0)
export CUDA_VISIBLE_DEVICES=0

# 模型名称
model_name=Informer

# 基础路径配置
# root_path: 存放 Preprocessing 文件夹的父目录
# data_path: 具体的数据子文件夹 (Arrival 或 Departure)
root_path_name=/root/autodl-tmp/Preprocessing/avg10_split300/
data_path_name=Arrival

# 特征维度配置
# 你的数据包含: Lat, Lon, Altitude, Speed, Dir_sin, Dir_cos, hour_sin, min_sin (共 8 维)
# 设置 enc_in, dec_in, c_out 均为 8
enc_in_dim=8
dec_in_dim=8
c_out_dim=8

# 循环运行不同预测长度的实验 (例如预测未来 24, 48, 96, 192 个点)
for pred_len in 24 48 96 192
do
  python -u run.py \
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
    --batch_size 32 \
    --patience 3
done