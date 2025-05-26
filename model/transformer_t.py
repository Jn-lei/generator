
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .LMConfig import LMConfig
from .model import apply_rotary_emb, repeat_kv, RMSNorm, FeedForward, MOEFeedForward


class TransformerAttention(nn.Module):
    """多头注意力机制实现"""
    
    def __init__(self, args: LMConfig, is_causal=True):
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
        self.is_causal = is_causal
        
        # 创建注意力掩码（仅在因果模式下）
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.register_buffer("mask", None, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        bsz, seq_len, _ = x.shape
        
        # 线性变换得到查询、键、值
        xq = self.wq(x)
        
        # 如果有编码器输出，则用于交叉注意力
        if encoder_hidden_states is not None:
            xk = self.wk(encoder_hidden_states)
            xv = self.wv(encoder_hidden_states)
            seq_len_k = encoder_hidden_states.size(1)
        else:
            xk, xv = self.wk(x), self.wv(x)
            seq_len_k = seq_len
            
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len_k, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len_k, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码（仅在自注意力时）
        if encoder_hidden_states is None:
            xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        
        # 处理KV缓存
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
            seq_len_k = xk.size(1)
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 使用Flash Attention或传统注意力计算
        if self.flash and seq_len != 1 and encoder_hidden_states is None:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=self.is_causal
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 应用掩码
            if self.is_causal and self.mask is not None and encoder_hidden_states is None:
                scores += self.mask[:, :, :seq_len, :seq_len_k]
            
            # 应用编码器注意力掩码（如果有）
            if encoder_attention_mask is not None:
                scores += encoder_attention_mask
                
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 输出处理
        output = output.transpose(1, 2).reshape(bsz, -1, self.n_local_heads * self.head_dim)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class TransformerBlock(nn.Module):
    """Transformer块，包含自注意力和前馈网络"""
    
    def __init__(self, layer_id: int, config: LMConfig, is_causal=True, is_cross_attention=False):
        super().__init__()
        self.layer_id = layer_id
        self.is_cross_attention = is_cross_attention
        
        # 自注意力层
        self.self_attention = TransformerAttention(config, is_causal=is_causal)
        self.self_attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # 交叉注意力层（仅在解码器中使用）
        if is_cross_attention:
            self.cross_attention = TransformerAttention(config, is_causal=False)
            self.cross_attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False, 
                encoder_hidden_states=None, encoder_attention_mask=None):
        # 自注意力层
        h_attn, self_attn_kv = self.self_attention(
            self.self_attention_norm(x),
            pos_cis,
            past_key_value=past_key_value[0] if past_key_value is not None else None,
            use_cache=use_cache
        )
        h = x + h_attn
        
        # 交叉注意力层（如果是解码器）
        if self.is_cross_attention and encoder_hidden_states is not None:
            cross_attn, cross_attn_kv = self.cross_attention(
                self.cross_attention_norm(h),
                pos_cis,
                past_key_value=past_key_value[1] if past_key_value is not None else None,
                use_cache=use_cache,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            h = h + cross_attn
            past_kv = (self_attn_kv, cross_attn_kv) if use_cache else None
        else:
            past_kv = self_attn_kv if use_cache else None
            
        # 前馈网络
        out = h + self.feed_forward(self.ffn_norm(h))
        
        if use_cache:
            return out, past_kv
        return out


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerBlock(i, config, is_causal=False, is_cross_attention=False)
            for i in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
    def forward(self, hidden_states, pos_cis, attention_mask=None):
        # 通过所有编码器层
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_cis)
        
        # 最终层归一化
        hidden_states = self.norm(hidden_states)
        return hidden_states


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        
        # 解码器层
        self.layers = nn.ModuleList([
            TransformerBlock(i, config, is_causal=True, is_cross_attention=True)
            for i in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
    def forward(self, hidden_states, pos_cis, encoder_hidden_states, 
                encoder_attention_mask=None, past_key_values=None, use_cache=False):
        
        # 初始化KV缓存
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            
        new_key_values = [] if use_cache else None
        
        # 通过所有解码器层
        for i, layer in enumerate(self.layers):
            if use_cache:
                hidden_states, key_value = layer(
                    hidden_states, 
                    pos_cis,
                    past_key_value=past_key_values[i],
                    use_cache=True,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
                new_key_values.append(key_value)
            else:
                hidden_states = layer(
                    hidden_states, 
                    pos_cis,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
        
        # 最终层归一化
        hidden_states = self.norm(hidden_states)
        
        if use_cache:
            return hidden_states, new_key_values
        return hidden_states


class TransformerModel(nn.Module):
    """编码器-解码器Transformer模型"""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # 词嵌入层
        self.encoder_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.decoder_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # 位置编码
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, config.max_seq_len * 2, 
            theta=config.rope_theta
        )
        
        # 编码器和解码器
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        # 输出层
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
    def forward(self, 
                encoder_input_ids, 
                decoder_input_ids, 
                encoder_attention_mask=None,
                decoder_attention_mask=None,
                past_key_values=None,
                use_cache=False):
        
        # 获取编码器输入嵌入
        encoder_hidden_states = self.encoder_embeddings(encoder_input_ids)
        
        # 获取位置编码
        encoder_seq_len = encoder_input_ids.size(1)
        encoder_freqs_cis = self.freqs_cis[:encoder_seq_len]
        if encoder_freqs_cis.device != encoder_hidden_states.device:
            encoder_freqs_cis = encoder_freqs_cis.to(encoder_hidden_states.device)
        
        # 编码器前向传播
        encoder_outputs = self.encoder(
            encoder_hidden_states, 
            encoder_freqs_cis, 
            attention_mask=encoder_attention_mask
        )
        
        # 获取解码器输入嵌入
        decoder_hidden_states = self.decoder_embeddings(decoder_input_ids)
        
        # 获取解码器位置编码
        decoder_seq_len = decoder_input_ids.size(1)
        decoder_freqs_cis = self.freqs_cis[:decoder_seq_len]
        if decoder_freqs_cis.device != decoder_hidden_states.device:
            decoder_freqs_cis = decoder_freqs_cis.to(decoder_hidden_states.device)
        
        # 解码器前向传播
        if use_cache:
            decoder_outputs, new_key_values = self.decoder(
                decoder_hidden_states,
                decoder_freqs_cis,
                encoder_outputs,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
        else:
            decoder_outputs = self.decoder(
                decoder_hidden_states,
                decoder_freqs_cis,
                encoder_outputs,
                encoder_attention_mask=encoder_attention_mask
            )
        
        # 最终输出层
        logits = self.output(decoder_outputs)
        
        if use_cache:
            return logits, new_key_values
        return logits


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算旋转位置编码的频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
