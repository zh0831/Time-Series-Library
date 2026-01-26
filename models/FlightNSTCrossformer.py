import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from math import ceil

# 引用 Crossformer 的组件
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead

# 引用 Nonstationary Transformer 的 Projector
from models.Nonstationary_Transformer import Projector


class Model(nn.Module):
    """
    Flight-NST-Crossformer
    1. Base: Crossformer
    2. Add: NST Normalization (Learnable Delta)
    3. Add: Dimension-Aware Embedding
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12  # Patch 大小，可调
        self.win_size = 2
        self.task_name = configs.task_name
        self.d_model = configs.d_model

        # --- 1. NST Projector Components (用于学习非平稳偏移) ---
        # 我们暂时只使用 Delta (Shift) 来校正输出，因为不改动内部 Attention，所以暂不需要 Tau
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.pred_len)

        # --- 2. Crossformer Data Padding Calculation ---
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # --- 3. Embedding Layers (含改进的维度嵌入) ---
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len,
                                                  self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # [NEW] 维度感知嵌入: 给 Lat, Lon, Alt 加上独立的身份证
        # Shape: [1, n_vars, 1, d_model] -> 可以广播到所有 patch
        self.dim_embedding = nn.Parameter(torch.randn(1, configs.enc_in, 1, configs.d_model))

        # --- 4. Crossformer Encoder ---
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l == 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )

        # --- 5. Crossformer Decoder ---
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model,
                                           configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                )
                for l in range(configs.e_layers + 1)
            ],
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()

        # ==========================================
        # Part 1: NST Normalization (归一化与去平稳统计)
        # ==========================================
        # 1. 计算均值和标准差
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # 2. 学习 Delta (漂移项)
        # B x S x E, B x 1 x E -> B x L (注意: 这里 Projector 输出的是 pred_len 长度的 delta)
        delta = self.delta_learner(x_raw, mean_enc)

        # ==========================================
        # Part 2: Embedding with Dimension Awareness
        # ==========================================
        # 原始 Patch Embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        # [NEW] 注入维度嵌入 (Add Dimension Embedding)
        # x_enc shape 此时为: [(Batch*Vars), Seg_Num, D_model]
        # 我们需要先将其还原为 [Batch, Vars, Seg_Num, D_model] 以便加上 dim_embedding
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)

        # 加上可学习的维度向量 (广播加法)
        # self.dim_embedding: [1, Vars, 1, D_model]
        x_enc = x_enc + self.dim_embedding

        # 加上位置嵌入
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        # ==========================================
        # Part 3: Encoder Inference
        # ==========================================
        enc_out, attns = self.encoder(x_enc)

        # ==========================================
        # Part 4: Decoder Inference
        # ==========================================
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)

        # Crossformer 的输出通常就是预测结果
        # dec_out shape: [Batch, Vars, Pred_Len] -> 需要转置回 [Batch, Pred_Len, Vars]
        # dec_out = dec_out.permute(0, 2, 1)

        # ==========================================
        # Part 5: NST De-Normalization (反归一化)
        # ==========================================
        # 1. 基础反归一化 ( * std + mean )
        dec_out = dec_out * std_enc + mean_enc

        # 2. 加上学习到的非平稳漂移 ( + delta )
        # delta shape: [Batch, Pred_Len, Vars] ?
        # Projector 输出通常是 [Batch, Pred_Len, Vars] 或者 [Batch, Pred_Len]
        # 如果是多变量，Projector 需要适配。标准的 NST Projector 输出维度是 seq_len (或 pred_len).
        # 在这里我们假设 Projector 输出 [Batch, Pred_Len]，我们需要 unsqueeze 以匹配 Vars
        # 或者如果 Projector 是针对每个变量独立学习的，shape 会匹配。
        # 简单起见，我们假设 delta 是所有变量共享的趋势，或者我们需要调整 Projector 输出维度。
        # 鉴于代码复用，我们这里简单将 delta 扩展到所有变量 (或者假设它学习的是全局时序偏移)
        if delta.dim() == 2:  # [Batch, Pred_Len]
            delta = delta.unsqueeze(-1)  # [Batch, Pred_Len, 1]

        dec_out = dec_out + delta

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None


# =================================================================
# 附：物理约束损失函数 (请在 exp/exp_basic.py 中替换 Criterion)
# =================================================================
class PhysicsTrajectoryLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.05):
        """
        weights: [Lat_weight, Lon_weight, Alt_weight]
        建议权重: Lat/Lon 设为 1.0, Alt 设为 1.0 / (高度的典型方差)
        例如高度误差通常是 1000m^2 = 10^6，为了让它变成 1.0，乘以 1e-6
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # 改为 none，不进行平均，我们要自己算
        self.alpha = alpha
        self.beta = beta


    def forward(self, pred, true):
        # 1. 位置误差 (加权)
        loss_pos = self.weighted_mse(pred, true)

        # 2. 速度误差 (一阶差分)
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        true_vel = true[:, 1:, :] - true[:, :-1, :]
        loss_vel = self.weighted_mse(pred_vel, true_vel)

        # 3. 加速度误差 (二阶差分)
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        true_acc = true_vel[:, 1:, :] - true_vel[:, :-1, :]
        loss_acc = self.weighted_mse(pred_acc, true_acc)

        return loss_pos + self.alpha * loss_vel + self.beta * loss_acc