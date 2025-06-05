from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .model import *        
        

class CasualAttention(nn.Module):
    def __init__(self, args: LMConfig):
        # mode 0：普通注意力 1：压缩注意力 

        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
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
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

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
        if self.flash and seq_len != 1:
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
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 输出处理
        output = output.transpose(1, 2).reshape(bsz, -1, self.n_local_heads * self.head_dim)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv

    

class CoderBlock(nn.Module):
    """
    包含注意力层和前馈网络层
    """
                            
    def __init__(self, layer_id: int, config: LMConfig,isMask=True, mode = 0):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.mode = mode
        self.attention = Attention(config, isMask, mode = self.mode)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x,pos_cis, use_cache=False, past_key_value=None):
        # 注意力层
        h, past_key_value = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value,
            use_cache
        )
        if self.mode != 1:
            h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_key_value     


class DecodeModel(nn.Module):

    def __init__(self ,params: LMConfig = None):
        super().__init__()
        self.decoder_layer =params.decoder_layers
        self.encoder = nn.ModuleList([CoderBlock(l, params, isMask=False) for l in range(params.decoder_layers)])
        self.decoder = nn.ModuleList([CoderBlock(l, params) for l in range(params.decoder_layers)])

    def forward(self, x , y , pos_cis,pos_cis_decoder,use_cache=False):
        all_kv = []
        for l, layer in enumerate(self.encoder):
            
            x, past_key_value = layer(x, pos_cis, True)
            all_kv.append(past_key_value)
        for l, layer in enumerate(self.decoder):
            y, _ = layer(y, pos_cis_decoder, use_cache, all_kv[l])
        return y



class CoderLM(PreTrainedModel):
    """
       完整的模型
    """
    config_class = LMConfig

    def __init__(self,  params: LMConfig = None):
        '''mode: 0 只编码 ， 1 编码-解码 ， 2 编码-生成-解码'''
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.mode = params.mode
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.encoder_layers=params.encoder_layers
        self.lc = params.lc
        # 初始化模型组件
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)

        self.encoder = nn.ModuleList([CoderBlock(l, params, isMask=True, mode = 1 if l == self.encoder_layers- self.lc else 0) for l in range(self.encoder_layers)])
        self.generater = MiniMindLM(params)
        self.decoder = DecodeModel(params)
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        # 注册位置编码
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()
        self.PE = 0

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                
                **args):
        """前向传播
        Args:
            input_ids: 输入token ID
            past_key_values: 用于加速生成的KV缓存
            use_cache: 是否使用KV缓存
        Returns:
            模型输出，包含logits和past_key_values
        """

        start_pos = args.get('start_pos', 0)
        # 词嵌入
        h = self.dropout(self.tok_embeddings(input_ids))
        text = h
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]

        for l, layer in enumerate(self.encoder):
            h ,_ = layer(h, pos_cis if self.PE == 1 else None)
            if layer.mode == 1:
                pos_cis = self.pos_cis[start_pos:start_pos + h.size(1)]

        if self.mode == 0:
            return h
        
        if self.mode == 2:
            h = self.generater(h,is_generator=True)
            h = h.logits
        pos_cis = self.pos_cis[start_pos:start_pos + h.size(1)]
        pos_cis_decoder = self.pos_cis[start_pos:start_pos + text.size(1)]
        h = self.decoder(h,text,pos_cis if self.PE == 'PE' else None,pos_cis_decoder if self.PE == 'PE' else None)

        # 输出层
        logits = self.output(self.norm(h))
        # 计算辅助损失
        # aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__('logits', logits)
        # self.OUT.__setitem__('aux_loss', aux_loss)
        return self.OUT
    

    def get_residual_weights(self):
        if hasattr(self, 'residual_weights'):
            return [torch.sigmoid(weight).item() for weight in self.residual_weights]
        else:
            return []
    def print_model_parameters(self):
        # print(f"Encoder layers: {self.encoder_layers}")
        # print(f"Decoder layers: {self.decoder_layers}")
        # print(f"Encoder layers: {self.encoder_layers}")
        print(f"")