import torch
import torch.nn as nn
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import TwoStageAttentionLayer


class Model(nn.Module):
    """
    TP-iTransformer: Temporal Patching Inverted Transformer

    A hybrid architecture for Trajectory Prediction:
    1. Patching: Captures local inertia dynamics (PatchTST).
    2. Inverted: Models global multivariate correlations (iTransformer).
    3. Two-Stage: Disentangles temporal and spatial dependencies (Crossformer).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # --- 1. Patching & Embedding ---
        # Calculation of number of patches (seg_num) to ensure alignment with Attention Layer
        # Assuming padding strategy matches stride to keep info
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # Calculate expected number of patches (seg_num)
        # Formula: (L - P) / S + 1. If padding is needed to be divisible, math changes slightly.
        # Here we align with standard PatchTST padding logic usually found in configs
        patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 2)

        # Padding logic to pass to PatchEmbedding
        # Note: In TSLib, PatchEmbedding takes explicit padding size
        padding = configs.stride

        self.enc_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=configs.patch_len,
            stride=configs.stride,
            padding=padding,
            dropout=configs.dropout
        )

        # --- 2. Encoder: Two-Stage Attention (Temporal + Spatial) ---
        # Uses TwoStageAttentionLayer from layers.SelfAttention_Family
        # This layer expects input [Batch, Num_vars, Num_patches, d_model]
        self.encoder = nn.ModuleList([
            TwoStageAttentionLayer(
                configs=configs,
                seg_num=patch_num,  # Number of patches (segments)
                factor=configs.factor,  # Factor for Router mechanism
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                dropout=configs.dropout
            ) for _ in range(configs.e_layers)
        ])

        # --- 3. Prediction Head ---
        # Project Flattened Patches -> Prediction Length
        # Input dim: Num_patches * d_model
        # Output dim: pred_len
        self.head = nn.Linear(patch_num * configs.d_model, configs.pred_len)

        # --- 4. Normalization (Non-stationary Transformer / RevIN) ---
        # Though not strictly a layer, we define logic in forward to keep parameters local if needed
        # We rely on instance normalization statistics calculated on the fly.
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc shape: [Batch, Seq_len, Enc_in] (e.g., [B, 96, 9])
        """

        # =============================================================
        # Step 1: Normalization (Non-stationary Transformer style)
        # =============================================================
        # Statistics for ReVIN to handle distributional shift in trajectories
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # =============================================================
        # Step 2: Embedding & Reshaping (The "Inverted" Logic)
        # =============================================================
        # Input: [Batch, Seq_len, n_vars]
        # Permute for PatchEmbedding (Channel Independence): [Batch, n_vars, Seq_len]
        x_enc = x_enc.permute(0, 2, 1)

        # Embedding:
        # Output of provided PatchEmbedding is [Batch * n_vars, Num_patches, d_model]
        # It flattens Batch and Channel into the first dimension.
        enc_out, n_vars = self.enc_embedding(x_enc)

        # Reshape to 4D for TwoStageAttentionLayer:
        # Target: [Batch, n_vars, Num_patches, d_model]
        # We need to separate Batch and n_vars again.
        # enc_out.shape[0] is Batch * n_vars
        B = int(enc_out.shape[0] / n_vars)
        _, num_patches, d_model = enc_out.shape

        # [Batch, n_vars, Num_patches, d_model]
        enc_out = enc_out.view(B, n_vars, num_patches, d_model)

        # =============================================================
        # Step 3: Two-Stage Encoder (Temporal -> Spatial Interaction)
        # =============================================================
        # TwoStageAttentionLayer handles the permuting internally:
        # 1. Temporal Attention: View as [(Batch*Vars), Patches, D]
        # 2. Spatial Attention: View as [(Batch*Patches), Vars, D]
        for i, layer in enumerate(self.encoder):
            enc_out = layer(enc_out)

        # Output enc_out is still [Batch, n_vars, Num_patches, d_model]

        # =============================================================
        # Step 4: Prediction Head
        # =============================================================
        # Flatten patches for each variable independently
        # [Batch, n_vars, Num_patches * d_model]
        dec_out = enc_out.flatten(2)

        # Project to future sequence length
        # [Batch, n_vars, Pred_len]
        dec_out = self.head(dec_out)

        # Permute back to standard TSLib output format: [Batch, Pred_len, n_vars]
        dec_out = dec_out.permute(0, 2, 1)

        # =============================================================
        # Step 5: De-Normalization
        # =============================================================
        dec_out = dec_out * stdev + means

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], None
        else:
            return dec_out[:, -self.pred_len:, :]  # [Batch, Pred_len, n_vars]