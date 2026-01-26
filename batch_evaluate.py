import os
import torch
import sys
import re
import glob
# 引入正确的类名 (单数形式)
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# =========================================================
#  1. 路径与环境配置 (根据您的 Shell 脚本修改)
# =========================================================

# 【重要】Checkpoints 的绝对路径
# 假设 run.py 在 /root/autodl-tmp/Time-Series-Library/
# 那么 checkpoints 通常在 /root/autodl-tmp/Time-Series-Library/checkpoints/
CHECKPOINT_ROOT = '/root/autodl-tmp/Time-Series-Library/checkpoints/'

# 【对应脚本】root_path_name
DATA_ROOT_PATH = '/root/autodl-tmp/Preprocessing/avg10_split300/'

# 【对应脚本】data_path_name
DATA_PATH_NAME = 'Arrival'


# =========================================================

class Args:
    def __init__(self):
        # --- 基础配置 (对应 Shell 脚本) ---
        self.model_id = 'Flight_Batch_Eval'  # 占位符，会被覆盖
        self.model = 'iTransformer'  # 【对应脚本】model_name
        self.data = 'trajectory'  # 【对应脚本】data="trajectory"
        self.features = 'M'  # 【对应脚本】features="M"
        self.task_name = 'long_term_forecast'

        # --- 路径配置 ---
        self.root_path = DATA_ROOT_PATH
        self.data_path = DATA_PATH_NAME

        # --- 维度定义 (对应 Shell 脚本) ---
        self.enc_in = 8  # 【对应脚本】enc_in_dim
        self.dec_in = 8  # 【对应脚本】dec_in_dim
        self.c_out = 8  # 【对应脚本】c_out_dim

        # --- 序列参数 (默认值，会被正则解析覆盖) ---
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24  # 默认值

        # --- 模型结构参数 (对应 Shell 脚本) ---
        self.d_model = 512  # 默认值，脚本中未显式指定，通常为512
        self.n_heads = 8  # 默认值
        self.e_layers = 2  # 【对应脚本】e_layers=2
        self.d_layers = 0  # 【对应脚本】d_layers=0
        self.d_ff = 2048  # 默认值
        self.factor = 3  # 【对应脚本】factor=3
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False

        # --- 运行参数 ---
        self.num_workers = 0  # 推理时建议设为0，避免多进程报错
        self.itr = 1
        self.batch_size = 512  # 【对应脚本】batch_size
        self.freq = 'h'  # <--- 必须加上这一行 (可以是 'h', 't', 's' 等，通常 'h' 通用)
        self.target = 'OT'  # [新增] 预测目标列名，虽然多变量预测不用，但Dataset初始化可能需要
        self.seasonal_patterns = 'Monthly'
        self.use_amp = False
        self.distil = True
        self.moving_avg = 25
        self.patch_len = 16

        self.use_gpu = True
        self.gpu = 0
        self.gpu_type = 'cuda'
        self.use_multi_gpu = False
        self.devices = '0'

        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2

        # --- 关键控制参数 ---
        self.is_training = 0  # 强制为测试模式
        self.inverse = True  # 开启反归一化，确保得到物理值
        self.use_dtw = False  # 是否计算DTW


# 检查 Checkpoint 路径
if not os.path.exists(CHECKPOINT_ROOT):
    print(f"❌ 错误：找不到 Checkpoint 路径 {CHECKPOINT_ROOT}")
    print("请检查代码中 CHECKPOINT_ROOT 变量是否配置正确。")
    sys.exit(1)

# 获取所有子文件夹
subfolders = os.listdir(CHECKPOINT_ROOT)
subfolders.sort()

print(f"📂 在 {CHECKPOINT_ROOT} 发现 {len(subfolders)} 个实验文件夹")
print("🚀 开始批量评估...\n")

# 准备写入结果的文件
output_file = "result_long_term_forecast.txt"

for folder_name in subfolders:
    full_folder_path = os.path.join(CHECKPOINT_ROOT, folder_name)

    # 忽略非文件夹
    if not os.path.isdir(full_folder_path):
        continue

    # 过滤：只处理包含您数据路径名(Arrival)或模型名(iTransformer)的文件夹
    if "iTransformer" not in folder_name:
        continue

    # =========================================================
    #  2. 正则表达式解析参数 (从文件夹名反推参数)
    # =========================================================
    try:
        args = Args()

        # 1. 提取序列长度 (sl), 标签长度 (ll), 预测长度 (pl)
        # 文件夹名示例: ..._sl96_ll48_pl24_...
        sl_match = re.search(r'sl(\d+)', folder_name)
        ll_match = re.search(r'll(\d+)', folder_name)
        pl_match = re.search(r'pl(\d+)', folder_name)

        if sl_match: args.seq_len = int(sl_match.group(1))
        if ll_match: args.label_len = int(ll_match.group(1))
        if pl_match: args.pred_len = int(pl_match.group(1))

        # 2. 提取模型参数 (dm, nh, el, dl, df, fc)
        dm_match = re.search(r'dm(\d+)', folder_name)
        nh_match = re.search(r'nh(\d+)', folder_name)
        el_match = re.search(r'el(\d+)', folder_name)
        dl_match = re.search(r'dl(\d+)', folder_name)
        df_match = re.search(r'df(\d+)', folder_name)
        fc_match = re.search(r'fc(\d+)', folder_name)

        if dm_match: args.d_model = int(dm_match.group(1))
        if nh_match: args.n_heads = int(nh_match.group(1))
        if el_match: args.e_layers = int(el_match.group(1))
        if dl_match: args.d_layers = int(dl_match.group(1))
        if df_match: args.d_ff = int(df_match.group(1))
        if fc_match: args.factor = int(fc_match.group(1))

        print(f"\n>>> 正在评估: iTransformer, Pred={args.pred_len}")
        print(f"    Folder: {folder_name}")
        print(f"    Structure: el{args.e_layers}-dl{args.d_layers}-dmodel{args.d_model}")

    except Exception as e:
        print(f"⚠️ 解析文件夹 {folder_name} 参数失败, 使用默认参数. 错误: {e}")
        # 继续尝试运行，使用默认 Args

    # =========================================================
    #  3. 加载权重并运行
    # =========================================================

    checkpoint_path = os.path.join(full_folder_path, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        print(f"   ⚠️ 跳过：未找到 checkpoint.pth")
        continue

    try:
        # 初始化实验
        exp = Exp_Long_Term_Forecast(args)

        # 手动加载权重
        print(f"   Loading weights...")
        exp.model.load_state_dict(torch.load(checkpoint_path))

        # 运行测试
        # setting 参数仅用于日志记录名字
        exp.test(setting=folder_name, test=0)

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()

print(f"\n✅ 全部完成！请查看根目录下的 {output_file}")