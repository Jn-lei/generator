"""
模型初始化工具类
提取自项目中各个模块的init_model方法，统一管理模型初始化逻辑
"""

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import apply_lora, load_lora
from model.coder_model import CoderLM
warnings.filterwarnings('ignore')


class ModelInitializer:
    """模型初始化器类，提供统一的模型初始化接口"""
    

    
    @staticmethod
    def _create_model(lm_config, CoderLM_class):
        """
        根据配置创建模型
        
        Args:
            lm_config: 语言模型配置
            CoderLM_class: CoderLM类
            
        Returns:
            model: 创建的模型实例
        """
        if hasattr(lm_config, 'model_type') and lm_config.model_type == 0:
            print("使用coder模型")
            model = CoderLM_class(lm_config)
            ModelInitializer._setup_coder_devices(model)
        else:
            print("使用generator模型")
            model = MiniMindLM(lm_config)
        return model
    
    @staticmethod
    def _setup_coder_devices(model):
        """
        为CoderLM模型设置设备分配
        
        Args:
            model: CoderLM模型实例
        """
        if hasattr(model, 'mode'):
            if model.mode < 2 and hasattr(model, 'generater'):
                model.generater.to('cpu')
            if model.mode < 1 and hasattr(model, 'decoder'):
                model.decoder.to('cpu')
    
    @staticmethod
    def _load_model_weights(model, checkpoint_path, device, strict=False):
        """
        加载模型权重
        
        Args:
            model: 模型实例
            checkpoint_path: 权重文件路径
            device: 目标设备
            strict: 是否严格匹配权重键
            
        Returns:
            model: 加载权重后的模型
        """
        state_dict = torch.load(checkpoint_path, map_location=device)
        # 如果是evaluation模型，需要过滤mask相关的键
        if 'mask' in str(checkpoint_path) or strict:
            model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)
        else:
            model.load_state_dict(state_dict, strict=False)
        return model
    
    @staticmethod
    def _load_tokenizer(tokenizer_path='./model/minimind_tokenizer'):
        """
        加载分词器
        
        Args:
            tokenizer_path: 分词器路径
            
        Returns:
            tokenizer: 分词器实例
        """
        return AutoTokenizer.from_pretrained(tokenizer_path)
    
    @staticmethod
    def _get_checkpoint_path(base_dir, model_name, dim, use_moe=False):
        """
        生成模型检查点路径
        
        Args:
            base_dir: 基础目录
            model_name: 模型名称 (pretrain, full_sft, rlhf, reason)
            dim: 模型维度
            use_moe: 是否使用MoE
            
        Returns:
            str: 检查点文件路径
        """
        moe_path = '_moe' if use_moe else ''
        return f'{base_dir}/{model_name}_{dim}{moe_path}.pth'
    
    @staticmethod
    def init_eval_model(args):
        """
        初始化评估模型（来自eval_model.py）
        
        Args:
            args: 包含模型配置参数的对象
            
        Returns:
            tuple: (model, tokenizer) 模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer()
        
        if args.load == 0:
            # 创建LMConfig
            lm_config = LMConfig(
                dim=args.dim,
                n_layers=args.n_layers,
                max_seq_len=args.max_seq_len,
                use_moe=args.use_moe
            )
            
            # 创建模型
            model = ModelInitializer._create_model(lm_config, CoderLM)
            
            # 加载权重
            modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
            ckp = ModelInitializer._get_checkpoint_path(f'./{args.out_dir}', modes[args.model_mode], args.dim, args.use_moe)
            model = ModelInitializer._load_model_weights(model, ckp, args.device, strict=True)

            # 加载LoRA权重（如果需要）
            if args.lora_name != 'None':
                apply_lora(model)
                load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
        else:
            # 使用transformers格式加载
            transformers_model_path = './MiniMind2'
            tokenizer = ModelInitializer._load_tokenizer(transformers_model_path)
            model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
        
        model.print_model_parameters()
        print(f'模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
        return model.eval().to(args.device), tokenizer

    @staticmethod
    def init_distill_reason_model(lm_config, device):
        """
        初始化蒸馏推理模型（来自train_distill_reason.py）
        
        Args:
            lm_config: 语言模型配置
            device: 设备类型
            
        Returns:
            tuple: (model, tokenizer) 模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer()
        
        # 创建模型
        model = ModelInitializer._create_model(lm_config, CoderLM)
        
        # 加载权重
        ckp = ModelInitializer._get_checkpoint_path('./out', 'rlhf', lm_config.dim, lm_config.use_moe)
        model = ModelInitializer._load_model_weights(model, ckp, device)
        
        print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        model = model.to(device)
        return model, tokenizer

    @staticmethod
    def init_lora_model(lm_config, device):
        """
        初始化LoRA模型（来自train_lora.py）
        
        Args:
            lm_config: 语言模型配置
            device: 设备类型
            
        Returns:
            tuple: (model, tokenizer) 模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer()
        
        # 创建模型
        model = ModelInitializer._create_model(lm_config, CoderLM)
        
        # 加载权重
        ckp = ModelInitializer._get_checkpoint_path('./out', 'rlhf', lm_config.dim, lm_config.use_moe)
        model = ModelInitializer._load_model_weights(model, ckp, device)
        
        return model.to(device), tokenizer

    @staticmethod
    def init_openai_api_model(args, device):
        """
        初始化OpenAI API服务模型（来自scripts/serve_openai_api.py）
        
        Args:
            args: 包含模型配置参数的对象
            device: 设备类型
            
        Returns:
            tuple: (model, tokenizer) 模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer('../model/minimind_tokenizer')
        
        if args.load == 0:
            # 创建LMConfig
            lm_config = LMConfig(
                dim=args.dim,
                n_layers=args.n_layers,
                max_seq_len=args.max_seq_len,
                use_moe=args.use_moe
            )
            
            # 创建模型
            model = ModelInitializer._create_model(lm_config, CoderLM)
            
            # 加载权重
            modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
            ckp = ModelInitializer._get_checkpoint_path(f'../{args.out_dir}', modes[args.model_mode], args.dim, args.use_moe)
            model = ModelInitializer._load_model_weights(model, ckp, device, strict=True)

            # 加载LoRA权重（如果需要）
            if args.lora_name != 'None':
                apply_lora(model)
                load_lora(model, f'../{args.out_dir}/{args.lora_name}_{args.dim}.pth')
        else:
            # 使用transformers格式加载
            model = AutoModelForCausalLM.from_pretrained(
                './MiniMind2',
                trust_remote_code=True
            )
        
        print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
        return model.eval().to(device), tokenizer

    @staticmethod
    def init_pretrain_model(lm_config, device):
        """
        初始化预训练模型（来自train_pretrain.py和encode_text.py）
        
        Args:
            lm_config: 语言模型配置
            device: 设备类型
            
        Returns:
            tuple: (model, tokenizer) 模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer()
        
        # 创建模型
        model = ModelInitializer._create_model(lm_config, CoderLM)
        model = model.to(device)
        
        print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        return model, tokenizer

    @staticmethod
    def init_full_sft_model(lm_config, device):
        """
        初始化全量SFT模型（来自train_full_sft.py）
        
        Args:
            lm_config: 语言模型配置
            device: 设备类型
            
        Returns:
            tuple: (model, tokenizer) 模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer()
        
        # 创建模型
        model = ModelInitializer._create_model(lm_config, CoderLM)
        
        # 加载权重
        ckp = ModelInitializer._get_checkpoint_path('./out', 'pretrain', lm_config.dim, lm_config.use_moe)
        model = ModelInitializer._load_model_weights(model, ckp, device)
        
        print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        model = model.to(device)
        return model, tokenizer

    @staticmethod
    def init_dpo_model(lm_config, device):
        """
        初始化DPO模型（来自train_dpo.py）
        
        Args:
            lm_config: 语言模型配置
            device: 设备类型
            
        Returns:
            tuple: (model, ref_model, tokenizer) 模型、参考模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer()
        
        # 创建模型
        model = ModelInitializer._create_model(lm_config, CoderLM)
        
        # 加载权重
        ckp = ModelInitializer._get_checkpoint_path('./out', 'full_sft', lm_config.dim, lm_config.use_moe)
        model = ModelInitializer._load_model_weights(model, ckp, device)
        
        # 初始化参考模型（使用相同的模型类型）
        ref_model = ModelInitializer._create_model(lm_config, CoderLM)
        ref_model = ModelInitializer._load_model_weights(ref_model, ckp, device)
        ref_model.eval()
        ref_model.requires_grad_(False)

        print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        model = model.to(device)
        ref_model = ref_model.to(device)

        return model, ref_model, tokenizer

    @staticmethod
    def init_convert_model():
        """
        初始化模型转换工具模型（来自scripts/convert_model.py）
        
        Returns:
            tuple: (model, tokenizer) 模型和分词器元组
        """
        tokenizer = ModelInitializer._load_tokenizer('../model/minimind_tokenizer')
        # 这里需要设置export_model_path，但在原始代码中它是局部变量
        # 暂时使用默认路径
        export_model_path = './MiniMind2'
        model = AutoModelForCausalLM.from_pretrained(export_model_path, trust_remote_code=True)
        return model, tokenizer


# 兼容性函数，保持原有接口
def init_model_eval(args):
    """兼容函数：eval_model.py的init_model"""
    return ModelInitializer.init_eval_model(args)


def init_model_distill_reason(lm_config, device='cpu'):
    """兼容函数：train_distill_reason.py的init_model"""
    return ModelInitializer.init_distill_reason_model(lm_config, device)


def init_model_lora(lm_config, device='cpu'):
    """兼容函数：train_lora.py的init_model"""
    return ModelInitializer.init_lora_model(lm_config, device)


def init_model_openai_api(args, device='cpu'):
    """兼容函数：scripts/serve_openai_api.py的init_model"""
    return ModelInitializer.init_openai_api_model(args, device)


def init_model_pretrain(lm_config, device='cpu'):
    """兼容函数：train_pretrain.py和encode_text.py的init_model"""
    return ModelInitializer.init_pretrain_model(lm_config, device)


def init_model_full_sft(lm_config, device='cpu'):
    """兼容函数：train_full_sft.py的init_model"""
    return ModelInitializer.init_full_sft_model(lm_config, device)


def init_model_dpo(lm_config, device='cpu'):
    """兼容函数：train_dpo.py的init_model"""
    return ModelInitializer.init_dpo_model(lm_config, device)


def init_model_convert():
    """兼容函数：scripts/convert_model.py的init_model"""
    return ModelInitializer.init_convert_model() 