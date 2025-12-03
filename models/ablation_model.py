import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#全局自注意力（第一层改进）
class GlobalSelfAttention(nn.Module):
    """
    使用自注意力机制建模用户所有历史帖子之间的全局关系，提取全局情感共性 V_common
    """
    def __init__(self, input_dim, hidden_dim=None):
        super(GlobalSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim

        self.W_Q = nn.Linear(input_dim, self.hidden_dim)
        self.W_K = nn.Linear(input_dim, self.hidden_dim)
        self.W_V = nn.Linear(input_dim, self.hidden_dim)
        self.scale = torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_K.bias)
        nn.init.zeros_(self.W_V.bias)

    def forward(self, x, mask=None):
        # x: (B, T, d)
        B, T, d = x.size()
        # print(f"GlobalSelfAttention input x: {x.shape}")

        Q = self.W_Q(x)  # (B, T, d)
        K = self.W_K(x)
        V = self.W_V(x)
        # print(f"Q, K, V: {Q.shape}")

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, T, T)
        # print(f"attn_scores: {attn_scores.shape}")

        if mask is not None:
            # print(f"mask: {mask.shape}")
            # 修复：只需 unsqueeze(1) → (B, 1, T)
            attn_mask = mask.unsqueeze(1)  # (B, 1, T)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
            # print(f"attn_scores after mask: {attn_scores.shape}")  # 应为 (B, T, T)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)
        V_common = torch.matmul(attn_weights, V)  # (B, T, d)
        # print(f"V_common after attn: {V_common.shape}")

        if mask is not None:
            mask_float = mask.unsqueeze(-1).type_as(V_common)  # (B, T, 1)
            V_common = V_common * mask_float  # (B, T, d)
            # print(f"V_common after mask: {V_common.shape}")

        # === 关键：对时间步求和 ===
        V_common = V_common.sum(dim=1)  # (B, d)
        # print(f"V_common after sum(dim=1): {V_common.shape}")

        if mask is not None:
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)  # (B, 1)
            # print(f"mask_sum: {mask_sum.shape}")
            V_common = V_common / mask_sum  # (B, d)
            # print(f"V_common after / mask_sum: {V_common.shape}")

        return V_common  # (B, d)


# 门控融合模块（第二、三层改进）
class GatedFusion(nn.Module):
    """
    门控融合：g * a + (1-g) * b
    """
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.bias)

    def forward(self, a, b):
        assert a.shape == b.shape, f"GatedFusion输入维度不匹配: {a.shape} vs {b.shape}"
        gate = self.sigmoid(self.W1(a) + self.W2(b) + self.bias)
        fused = gate * a + (1 - gate) * b
        return fused


# ================== 原始 Attention 模块（保持不变） ==================
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        # self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.att_weights = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
        stdv = 1.0 / np.sqrt(self.hidden_size)
        nn.init.uniform_(self.att_weights, -stdv, stdv)

    def forward(self, inputs, mask=None):
        batch_size, seq_len, hidden_dim = inputs.size()

        # 修复：直接点积，计算每个时间步的注意力得分
        weights = torch.matmul(inputs, self.att_weights)  # (B, T)

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        weights = F.softmax(weights, dim=-1)  # (B, T)
        weights = weights.unsqueeze(-1)  # (B, T, 1)

        outputs = inputs * weights  # (B, T, H)
        outputs = outputs.sum(dim=1)  # (B, H)

        if mask is not None:
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            outputs = outputs / mask_sum

        return outputs, weights


# ================== MLP 分类器（保持不变） ==================
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, class_num, output_layer=True):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, class_num))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ================== 主模型：GRU_CNN_Attention（完整重构） ==================
class GRU_CNN_Attention(nn.Module):
    def __init__(self, args, words_id, vocab_size, device, weights=None, is_pretrain=False):
        super(GRU_CNN_Attention, self).__init__()
        self.args = args
        self.device = device
        self.words_id = words_id
        self.dropout = args.dropout
        self.gru_size = args.gru_size
        self.class_num = args.class_num
        self.embedding_dim = args.embedding_dim

        # 词嵌入层
        if is_pretrain and weights is not None:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, self.embedding_dim)

        # 1. 词嵌入后添加Dropout
        self.embed_drop = nn.Dropout(args.dropout)  # 使用args中的dropout参数

        # 词级双向GRU
        self.word_gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.gru_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 2. 词级GRU输出后添加Dropout
        self.word_gru_drop = nn.Dropout(args.dropout)

        # 词级注意力参数
        self.word_fc = nn.Linear(2 * self.gru_size, 2 * self.gru_size)
        self.word_query = nn.Parameter(torch.Tensor(2 * self.gru_size, 1), requires_grad=True)

        # 句子级双向GRU
        self.sentence_gru = nn.GRU(
            input_size=2 * self.gru_size,
            hidden_size=self.gru_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 3. 句子级GRU输出后添加Dropout
        self.sentence_gru_drop = nn.Dropout(args.dropout)

        # 注意力机制
        self.attention = Attention(2 * self.gru_size, batch_first=True)

        # ✅ 第一层改进：全局自注意力（替代 PEAM）
        self.global_attention = GlobalSelfAttention(input_dim=2 * self.gru_size)

        # ✅ 第二层改进：门控融合长期动态与全局共性
        self.gated_long_fusion = GatedFusion(2 * self.gru_size)

        # ✅ 第三层改进：门控融合长期与短期
        self.gated_final_fusion = GatedFusion(2 * self.gru_size)

        self.use_global_attention = args.use_global_attention
        self.use_gated_fusion = args.use_gated_fusion
        self.word_level_only = args.word_level_only  # 添加：词级仅用开关

        # 添加：预定义融合用线性层（绑定到模型状态）
        self.fusion_fc1 = nn.Linear(4 * self.gru_size, 2 * self.gru_size)  # 用于V_long融合
        self.fusion_fc2 = nn.Linear(4 * self.gru_size, 2 * self.gru_size)  # 用于V_final融合

        # 4. 最终特征融合后添加Dropout
        self.final_drop = nn.Dropout(args.dropout)

        # 分类器
        # self.class_fc = MultiLayerPerceptron(
        #     input_dim=2 * self.gru_size,
        #     embed_dims=[2 * self.gru_size, self.gru_size],
        #     dropout=self.dropout,
        #     class_num=self.class_num
        # )
        #
        self.class_fc = MultiLayerPerceptron(
            input_dim=2 * self.gru_size,
            embed_dims=[2 * self.gru_size, self.gru_size, self.gru_size // 2],  # 增加一层
            dropout=self.dropout,
            class_num=self.class_num
        )
        self._init_weights()
        # 添加：初始化日志打印标记
        self.debug_printed = False  # 确保在forward中可调用

    def _init_weights(self):
        nn.init.xavier_uniform_(self.word_query)
        nn.init.xavier_uniform_(self.word_fc.weight)
        for name, param in self.word_gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.sentence_gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, text_masks, post_masks, use_gpu=False):
        """
        x: (batch_size, sentence_num, sentence_len)
        text_masks: (batch_size, sentence_num)  # 帖子有效标记
        post_masks: (batch_size, sentence_num, sentence_len)  # 词有效标记
        """
        # 添加调试信息
        if hasattr(self, 'debug_printed') and not self.debug_printed:
            print(f"[DEBUG] use_global_attention: {self.use_global_attention}")
            print(f"[DEBUG] use_gated_fusion: {self.use_gated_fusion}")
            self.debug_printed = True
        batch_size, sentence_num, sentence_len = x.size()

        # Reshape for word-level processing
        x = x.view(-1, sentence_len)  # (B*S, L)
        post_masks = post_masks.view(-1, sentence_len)  # (B*S, L)

        # 获取每条文本的词数
        word_lengths = post_masks.sum(dim=1)  # (B*S,)
        non_zero_indices = (word_lengths > 0).nonzero(as_tuple=True)[0]
        if len(non_zero_indices) == 0:
            return torch.zeros(batch_size, self.class_num, device=self.device)

        # 处理非空文本
        texts_non_zero = x[non_zero_indices]
        word_lengths_non_zero = word_lengths[non_zero_indices]
        embed_x = self.word_embed(texts_non_zero)  # (N, L, D)
        embed_x_mask = post_masks[non_zero_indices]  # (N, L)

        # Pack and run word GRU
        packed_words = nn.utils.rnn.pack_padded_sequence(
            embed_x, word_lengths_non_zero.cpu(), batch_first=True, enforce_sorted=False
        )
        word_output, _ = self.word_gru(packed_words)
        word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)  # (N, L, 2H)
        word_output = self.word_gru_drop(word_output)  # 应用词级GRU输出Dropout

        # Word-level attention
        word_attention = torch.tanh(self.word_fc(word_output))  # (N, L', 2H)
        weights = torch.matmul(word_attention, self.word_query)  # (N, L', 1)

        # ✅ 关键：构造对齐的 mask
        _, max_output_len, _ = word_output.size()
        aligned_mask = torch.arange(max_output_len, device=self.device).unsqueeze(0) < word_lengths_non_zero.unsqueeze(
            1)
        # aligned_mask.shape = (N, L')

        weights = weights.masked_fill(~aligned_mask.unsqueeze(-1), -1e9)  # (N, L', 1)
        weights = F.softmax(weights, dim=1)  # (N, L', 1)
        sentence_vectors = (weights * word_output).sum(dim=1)  # (N, 2H)
        # Restore to batch format
        pair_vector = torch.zeros(batch_size * sentence_num, 2 * self.gru_size, device=self.device)
        pair_vector[non_zero_indices] = sentence_vectors
        pair_vector = pair_vector.view(batch_size, sentence_num, 2 * self.gru_size)


        # Mask for valid posts
        post_valid_mask = text_masks.bool()  # (B, T)

        # # ✅ 第一层：全局情感共性 V_common（根据配置开关）
        # if self.use_global_attention:
        #     V_common = self.global_attention(pair_vector, mask=post_valid_mask)  # (B, 2H)
        # else:
        #     # 不使用全局注意力时，用平均代替
        #     mask_float = post_valid_mask.unsqueeze(-1).type_as(pair_vector)
        #     V_common = (pair_vector * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)

        # 提取长期动态特征 V_dynamic
        post_lengths = text_masks.sum(dim=1).cpu()  # (B,)
        non_zero_post_indices = (post_lengths > 0).nonzero(as_tuple=True)[0]
        if len(non_zero_post_indices) == 0:
            return torch.zeros(batch_size, self.class_num, device=self.device)

        valid_pair_vector = pair_vector[non_zero_post_indices]
        valid_text_masks_full = text_masks[non_zero_post_indices]  # (N, 50)
        post_lengths_tensor = post_lengths[non_zero_post_indices]  # (N,)

        # ✅ 防御性编程：确保 max_len 是 int
        if post_lengths_tensor.numel() == 0:
            return torch.zeros(batch_size, self.class_num, device=self.device)

        max_len_tensor = post_lengths_tensor.max()
        max_len = int(max_len_tensor.item())  # 强制转 int

        # ✅ 安全检查
        if max_len <= 0:
            max_len = 1  # 至少保留一个时间步


        valid_text_masks = valid_text_masks_full[:, :max_len]  # ✅ 现在肯定不会报错！

        packed_sentences = nn.utils.rnn.pack_padded_sequence(
            valid_pair_vector, post_lengths_tensor, batch_first=True, enforce_sorted=False
        )
        sentence_output, _ = self.sentence_gru(packed_sentences)
        sentence_output, _ = nn.utils.rnn.pad_packed_sequence(sentence_output, batch_first=True)  # (N, T_max, 2H)
        sentence_output = self.sentence_gru_drop(sentence_output)  # 应用句子级GRU输出Dropout

        V_dynamic, _ = self.attention(sentence_output, valid_text_masks)  # ✅ 形状匹配！

        V_dynamic_full = torch.zeros(batch_size, 2 * self.gru_size, device=self.device)
        V_dynamic_full[non_zero_post_indices] = V_dynamic

        if self.use_global_attention:
            V_common = self.global_attention(pair_vector, mask=post_valid_mask)  # (B, 2H)
        else:
            # 彻底移除全局共性特征（用零向量替代）
            V_common = torch.zeros_like(V_dynamic_full)  # (B, 2H) 与V_dynamic_full同形状
        # ✅ 第二层：融合 V_dynamic 和 V_common → V_long（根据配置开关）
        # 修改为：
        if self.use_gated_fusion:
            V_long = self.gated_long_fusion(V_dynamic_full, V_common)  # (B, 2H)
        else:
            # 使用预定义线性层（已绑定到模型）
            V_long = torch.cat([V_dynamic_full, V_common], dim=-1)
            V_long = self.fusion_fc1(V_long)  # 用预定义层替代动态创建
        # ✅ 第三层：短期特征 = 最后一条有效帖子
        last_indices = (text_masks.cumsum(dim=1) == text_masks.sum(dim=1, keepdim=True)).float()
        V_short = (pair_vector * last_indices.unsqueeze(-1)).sum(dim=1)  # (B, 2H)

        # ✅ 修正后：先判断是否仅用词级特征，再处理门控融合
        # 1. 若仅用词级特征，直接跳过所有融合逻辑
        if self.word_level_only:
            V_final = V_short  # 仅使用最后一条帖子的词级特征（词级仅用逻辑）
        else:
            # 2. 若非词级仅用，再根据门控融合开关处理 V_long 和 V_short 的融合
            if self.use_gated_fusion:
                V_final = self.gated_final_fusion(V_long, V_short)  # 门控融合
            else:
                V_final = torch.cat([V_long, V_short], dim=-1)  # 拼接融合
                V_final = self.fusion_fc2(V_final)  # 用预定义层降维

        # 分类（无论哪种逻辑，最终特征都通过分类器输出）
        logits = self.class_fc(V_final)  # (B, C)

        return logits
