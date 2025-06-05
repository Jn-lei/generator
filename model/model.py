import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """RMSNorm归一化层
    使用均方根归一化方法，相比LayerNorm计算更简单且效果相当
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        # 计算均方根归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """预计算旋转位置编码
    Args:
        dim: 编码维度
        end: 最大序列长度
        theta: 旋转角度参数
    Returns:
        预计算的位置编码矩阵
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    """应用旋转位置编码到查询和键向量
    Args:
        xq: 查询向量
        xk: 键向量
        pos_cis: 预计算的位置编码
    Returns:
        应用了位置编码的查询和键向量
    """

    def unite_shape(pos_cis, x):
        # 调整位置编码的形状以匹配输入张量
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    if pos_cis is None:
        return xq, xk   
    # 将输入转换为复数形式
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    # 应用旋转位置编码
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复键值向量以匹配注意力头数
    Args:
        x: 输入张量
        n_rep: 重复次数
    Returns:
        重复后的张量
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """注意力层
    实现了多头注意力机制，支持KV缓存和Flash Attention
    """

    def __init__(self, args: LMConfig, isMask = True, mode = 0):
        # mode 0：普通注意力 1：压缩注意力 

        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.mode = mode
        # 初始化注意力相关的线性层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 是否使用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 创建注意力掩码
        if isMask:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        # 线性变换得到查询、键、值
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # 处理KV缓存
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序
        # 调整维度顺序，将序列长度和注意力头的维度交换
        # xq: [batch_size, seq_len, n_heads, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)

        # 对于键值矩阵，需要重复以匹配查询头的数量
        # 当n_kv_heads < n_heads时，每个KV头需要被复制n_rep次
        # repeat_kv函数将KV张量从[batch_size, seq_len, n_kv_heads, head_dim]
        # 扩展为[batch_size, seq_len, n_heads, head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # 同样交换维度顺序
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # 同样交换维度顺序

        # 最终维度:
        # xq, xk, xv: [batch_size, n_heads, seq_len, head_dim]
        # 这种格式适合后续的注意力计算
        # 使用Flash Attention或传统注意力计算
        if self.flash and seq_len != 1 and self.mode != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.mask is not None:
                scores += self.mask[:, :, :seq_len, :seq_len]
            if self.mode == 1:
                scores = self.attention_intergate(scores, 2, 2)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 输出处理
        output = output.transpose(1, 2).reshape(bsz, -1, self.n_local_heads * self.head_dim)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv

    def attention_intergate(self,attention: torch.Tensor, kernel_size: int, stride: int):
        """
        将S*S的矩阵进行average pooling，窗口大小为(kernel_size,1),左右步长为1，上下步长为stride
        """
        # 获取注意力矩阵的形状
        batch_size, num_heads, seq_len, _ = attention.shape

        # 创建一个一维平均池化层
        avg_pool = nn.AvgPool2d(kernel_size=(kernel_size, 1), stride=(stride, 1))

        # 重塑张量以适应池化操作
        # 将batch_size和num_heads维度合并
        attention_reshaped = attention.reshape(-1, seq_len, seq_len)

        # 应用池化操作
        pooled_attention = avg_pool(attention_reshaped.unsqueeze(1)).squeeze(1)

        # 重塑回原始形状
        pooled_attention = pooled_attention.reshape(batch_size, num_heads, -1, seq_len)

        return pooled_attention

class FeedForward(nn.Module):
    """前馈神经网络层，实现了SwiGLU激活函数的前馈网络"""

    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    """混合专家系统的门控网络
    负责选择最合适的专家处理输入
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 专家总数

        self.scoring_func = config.scoring_func  # 评分函数类型
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux  # 是否使用序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化top-k概率
        self.gating_dim = config.dim  # 门控网络维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数"""
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        # 计算每个专家的得分
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择top-k个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化top-k概率
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """混合专家前馈网络
    实现了基于专家系统的前馈网络
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        # 可选的共享专家
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式：使用所有选中的专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式：只使用最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """推理时的专家选择逻辑"""
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 按专家分组处理输入
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """MiniMind模型的基本构建块
    包含注意力层和前馈网络层
    """

    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        # 注意力层
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn
        # 前馈网络层
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    """MiniMind语言模型
    基于Transformer架构的语言模型，支持混合专家系统
    """
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        # 初始化模型组件
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        # 注册位置编码
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

        # 初始化残差权重为极小的负数
        # self.residual_weights = nn.ParameterList([
        #     nn.Parameter(torch.full((1, 1, 1), 0.0))  # 初始化为-10，经过sigmoid后约为4.5e-5
        #     for _ in range(self.n_layers//2)
        # ])

        # 添加全连接层用于计算alpha值
        # self.alpha_fc = nn.Linear(params.dim, 1, bias=False)
        # 初始化权重为较小的值，使得sigmoid后的值接近0.5
        # nn.init.normal_(self.alpha_fc.weight, mean=0.0, std=0.01)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                is_generator: bool = False,
                **args):
        """前向传播
        Args:
            input_ids: 输入token ID
            past_key_values: 用于加速生成的KV缓存
            use_cache: 是否使用KV缓存
        Returns:
            模型输出，包含logits和past_key_values
        """
        past_key_values = [None] * len(self.layers) if past_key_values is None else past_key_values
        start_pos = args.get('start_pos', 0)
        # 词嵌入
        h = self.dropout(self.tok_embeddings(input_ids)) if not is_generator else input_ids
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        # 通过所有层
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)

            # if l > 0 and l % 2 == 1:  # 从第二层开始应用加权求和
            #     # 应用权重进行加权求和
            #     alpha = torch.sigmoid(self.alpha_fc(h.mean(dim=1, keepdim=True)))  # 使用全连接层计算alpha值
            #     h = alpha * h + (1 - alpha) * prev_h
            #     # 保存当前层的输出用于下一层
            #     prev_h = h.clone()  # 使用clone()创建新的张量，保持梯度计算图的连接

        # 输出层
        logits = h if is_generator else self.output(self.norm(h))
        # 计算辅助损失
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        """生成文本
        Args:
            input_ids: 输入token ID
            eos_token_id: 结束符ID
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: 核采样参数
            stream: 是否使用流式生成
            rp: 重复惩罚参数
            use_cache: 是否使用KV缓存
            pad_token_id: 填充token ID
        Returns:
            生成的文本序列
        """
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 批量生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        # 填充到相同长度
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        """流式生成文本
        逐个token生成，支持KV缓存加速
        """
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            # 应用重复惩罚
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            # 应用核采样
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            # 采样下一个token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break

    def print_model_parameters(self):
        """打印模型的所有参数信息
        包括参数名称、形状和参数量
        """
        # 特别打印residual_weights参数
        if hasattr(self, 'residual_weights'):
            for i, weight in enumerate(self.residual_weights):
                print(f"第{i+1}组 数值: {weight.data} 经过sigmoid后的值: {torch.sigmoid(weight).data}")
        else:
            print("没有residual_weights参数")

    def get_residual_weights(self):
        if hasattr(self, 'residual_weights'):
            return [torch.sigmoid(weight).item() for weight in self.residual_weights]
        else:
            return []



