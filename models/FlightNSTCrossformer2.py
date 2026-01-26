import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from math import ceil

# 复用基础组件
from layers.Crossformer_EncDec import scale_block
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention


# =============================================================================
# 1. 核心组件：去平稳化注意力机制 (DS-Attention & Wrapper)
# =============================================================================

class DS_FullAttention(nn.Module):
    """
    带有去平稳化缩放因子 (tau) 的全注意力机制
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DS_FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # queries: [B, L, H, E]
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        scale = self.scale or (1. / (E ** 0.5))

        # 1. 计算 Attention Score
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 2. [关键点] 引入 NST 的 Tau 进行去平稳化缩放
        # tau shape: [B, 1, 1, 1] (经过外部 reshape 后传入)
        if tau is not None:
            # exp(tau) 作为缩放系数，注入到 Softmax 之前
            scores = scores * scale * torch.exp(tau)
        else:
            scores = scores * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L, S), diagonal=1).bool().to(queries.device)
            scores.masked_fill_(attn_mask, -np.inf)

        # 3. Softmax & Value Aggregation
        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class DS_TwoStageAttentionLayer(nn.Module):
    """
    改进版的双阶段注意力层：
    Stage 1 (Time): 使用 DS-Attention，注入 tau。
    Stage 2 (Dim):  保持原有的 Router 机制。
    """

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(DS_TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # Stage 1: Time Attention (替换为 DS_FullAttention)
        self.time_attention = AttentionLayer(
            DS_FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )

        # Stage 2: Dimension Attention (Router) - 保持 Crossformer 原样
        self.dim_sender = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, tau=None):
        # x shape: [Batch, D_vars, Seg_num, d_model]
        batch, n_vars, seg_num, d_model = x.shape

        # ================= Stage 1: Time Attention (with Tau) =================
        # Reshape for time attention: [Batch * D_vars, Seg_num, d_model]
        time_in = x.reshape(batch * n_vars, seg_num, d_model)

        # [关键对齐] Tau Alignment
        # tau 原始 shape: [Batch, n_vars]
        # 需要变换为: [Batch * n_vars, 1, 1, 1] 以匹配 attention score (B, H, L, S)
        if tau is not None:
            tau_time = tau.reshape(batch * n_vars, 1, 1, 1)
        else:
            tau_time = None

        time_enc, _ = self.time_attention(time_in, time_in, time_in, attn_mask=None, tau=tau_time)
        time_enc = time_enc.reshape(batch, n_vars, seg_num, d_model)
        x = self.norm1(x + self.dropout(time_enc))
        x = self.norm2(x + self.dropout(self.MLP1(x)))

        # ================= Stage 2: Dimension Attention (Router) =================
        # Router mechanism 保持不变，无需 tau (因为是维度间通信，不涉及时序非平稳)
        dim_send = x.permute(0, 2, 1, 3).reshape(batch * seg_num, n_vars, d_model)
        router = self.router.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * seg_num, -1, d_model)

        # Router collects info
        router_enc, _ = self.dim_sender(router, dim_send, dim_send, attn_mask=None)

        # Router distributes info
        dim_rec, _ = self.dim_receiver(dim_send, router_enc, router_enc, attn_mask=None)

        dim_rec = dim_rec.reshape(batch, seg_num, n_vars, d_model).permute(0, 2, 1, 3)
        x = self.norm3(x + self.dropout(dim_rec))
        x = self.norm4(x + self.dropout(self.MLP2(x)))

        return x


class DS_Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(DS_Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, tau=None):
        for layer in self.attn_layers:
            x = layer(x, tau=tau)
        if self.norm is not None:
            x = self.norm(x)
        return x


# =============================================================================
# 2. 辅助组件：Projector & FlattenHead
# =============================================================================

class NST_Projector(nn.Module):
    """
    学习非平稳统计特性 (Tau, Delta)
    """

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim):
        super(NST_Projector, self).__init__()
        # Tau Learner: 输出 [Batch, enc_in]
        self.tau_learner = nn.Sequential(
            nn.Linear(seq_len, 1),
            nn.Sigmoid()  # 限制在 0-1 之间比较稳定，或者去掉也可以
        )

        # Delta Learner: 输出 [Batch, output_dim, enc_in]
        # 为了简单，我们对每个变量独立学习漂移
        self.delta_learner = nn.Sequential(
            nn.Linear(seq_len, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], output_dim)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Vars]
        x_trans = x.permute(0, 2, 1)  # [Batch, Vars, Seq_Len]

        # 1. Learn Tau
        # [Batch, Vars, Seq_Len] -> [Batch, Vars, 1] -> [Batch, Vars]
        tau = self.tau_learner(x_trans).squeeze(-1)

        # 2. Learn Delta
        # [Batch, Vars, Seq_Len] -> [Batch, Vars, Pred_Len] -> [Batch, Pred_Len, Vars]
        delta = self.delta_learner(x_trans).permute(0, 2, 1)

        return tau, delta


class FlattenHead(nn.Module):
    """
    Channel-Independent FlattenHead
    将 (Seg_Num * d_model) 映射到 Pred_Len
    """

    def __init__(self, seg_num, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seg_num * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [Batch, Vars, Seg_Num, D_Model]
        x = self.flatten(x)  # [Batch, Vars, Seg_Num * D_Model]
        x = self.dropout(x)
        x = self.linear(x)  # [Batch, Vars, Pred_Len]
        x = x.permute(0, 2, 1)  # [Batch, Pred_Len, Vars]
        return x


# =============================================================================
# 3. 主模型：FlightNSTCrossformer
# =============================================================================

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12  # Patch Segment Length
        self.win_size = 2
        self.task_name = configs.task_name


        # --- 1. NST & Pre-process ---
        # 简单的 Projector 配置，防止 configs 缺失参数报错
        p_hidden_dims = getattr(configs, 'p_hidden_dims', [64, 64])
        p_hidden_layers = getattr(configs, 'p_hidden_layers', 2)

        self.projector = NST_Projector(
            enc_in=configs.enc_in,
            seq_len=configs.seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=configs.pred_len
        )

        # --- 2. Embedding & Padding ---
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len

        self.enc_value_embedding = PatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model)
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # [改进点] 维度感知嵌入 (Dimension Embedding)
        # 为每个变量 (Lat, Lon, Alt...) 学习一个独立的向量
        self.dim_embedding = nn.Parameter(torch.randn(1, configs.enc_in, 1, configs.d_model))

        # --- 3. DS-Encoder ---
        # 使用自定义的 DS_TwoStageAttentionLayer
        self.encoder = DS_Encoder(
            [
                DS_TwoStageAttentionLayer(
                    configs, self.in_seg_num, configs.factor, configs.d_model, configs.n_heads,
                    configs.d_ff, configs.dropout
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # --- 4. FlattenHead ---
        self.flatten_head = FlattenHead(
            self.in_seg_num, configs.d_model, configs.pred_len, head_dropout=configs.dropout
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [Batch, Seq_Len, Vars] (真实物理数值)

        # ================= Part 1: NST Normalization =================
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Learn Tau & Delta
        # tau: [Batch, Vars]
        # delta: [Batch, Pred_Len, Vars]
        tau, delta = self.projector(x_enc)

        # ================= Part 2: Embedding =================
        # Patch Embed: [Batch, Vars, Seg_Num, D_Model] (Crossformer handle vars inside)
        # Note: PatchEmbedding output is [Batch*Vars, Seg, D]. Need reshape.
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) s m -> b d s m', d=n_vars)

        # Add Pos Embed + Dim Embed
        x_enc += self.enc_pos_embedding
        x_enc += self.dim_embedding  # Broadcast add
        x_enc = self.pre_norm(x_enc)

        # ================= Part 3: DS-Encoder =================
        # 传入 tau 用于去平稳化注意力
        enc_out = self.encoder(x_enc, tau=tau)

        # ================= Part 4: FlattenHead =================
        # enc_out: [Batch, Vars, Seg_Num, D_Model]
        dec_out = self.flatten_head(enc_out)  # -> [Batch, Pred_Len, Vars]

        # ================= Part 5: De-Normalization =================
        # 1. Standard DeNorm
        dec_out = dec_out * std_enc + mean_enc

        # 2. NST Shift Correction
        dec_out = dec_out + delta

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 截取有效预测长度
            return dec_out[:, -self.pred_len:, :]
        return None


# =============================================================================
# 4. 物理约束损失函数 (附带在文件末尾，方便调用)
# =============================================================================

def forward(self, pred, true):
    # pred, true shape: [Batch, Length, Vars]

    # 1. 基础 MSE Loss
    loss_mse = self.mse(pred, true)

    # 2. 速度约束 (一阶差分)
    # 计算相邻时间点的差值
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    true_diff = true[:, 1:, :] - true[:, :-1, :]
    loss_vel = self.mse(pred_diff, true_diff)

    # 3. 平滑/加速度约束 (二阶差分)
    # 限制预测轨迹的二阶导数不要太大（防止抖动）
    pred_acc = pred_diff[:, 1:, :] - pred_diff[:, :-1, :]
    # 我们希望加速度尽可能平滑，或者接近真实的加速度
    # 这里简化处理：让预测的加速度接近真实的加速度
    true_acc = true_diff[:, 1:, :] - true_diff[:, :-1, :]
    loss_acc = self.mse(pred_acc, true_acc)

    # === 修改点：先对各部分取 mean() 再相加，解决形状不匹配问题 ===
    total_loss = loss_mse.mean() + (self.alpha * loss_vel.mean()) + (self.beta * loss_acc.mean())

    return total_loss