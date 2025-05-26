import argparse
from dataclasses import dataclass
from transformers import PretrainedConfig
from typing import List

@dataclass
class LMConfig(PretrainedConfig):
    # 0: 表示 "coder"  1 ：表示 generator
    model_type: int = 0
    dim: int = 10
    n_layers: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    lc: int = 2
    n_heads: int = 8
    n_kv_heads: int = 2
    vocab_size: int = 6400
    hidden_dim: int = None
    multiple_of: int = 64
    norm_eps: float = 1e-5
    max_seq_len: int = 8192
    rope_theta: int = 1e6
    dropout: float = 0.0
    flash_attn: bool = True
    ####################################################
    # Here are the specific configurations of MOE
    # When use_moe is false, the following is invalid
    ####################################################
    use_moe: bool = False
    ####################################################
    num_experts_per_tok: int = 2  # 每个token选择的专家数量
    n_routed_experts: int = 4  # 总的专家数量
    n_shared_experts: bool = True  # 共享专家
    scoring_func: str = 'softmax'  # 评分函数，默认为'softmax'
    aux_loss_alpha: float = 0.1  # 辅助损失的alpha参数
    seq_aux: bool = True  # 是否在序列级别上计算辅助损失
    norm_topk_prob: bool = True  # 是否标准化top-k概率