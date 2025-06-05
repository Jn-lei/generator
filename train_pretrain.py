import os                  # 操作系统相关功能，如文件路径处理
import platform            # 获取系统平台信息
import argparse            # 命令行参数解析
import time                # 时间相关函数，用于性能计时
import math                # 数学函数，用于学习率调整
import warnings            # 警告控制
import pandas as pd        # 数据处理库
import torch               # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
from torch import optim, nn       # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行训练
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载和分布式采样器
from contextlib import nullcontext  # 空上下文管理器，用于条件性地启用/禁用上下文
import yaml  # 添加yaml支持

from transformers import AutoTokenizer  # Hugging Face分词器

# 导入自定义模型和相关类
from model.model import MiniMindLM      # MiniMind语言模型
from model.LMConfig import LMConfig     # 模型配置类
from model.dataset import PretrainDataset  # 预训练数据集类
from model.coder_model import CoderLM

# 忽略警告信息，避免输出过多无关紧要的警告
warnings.filterwarnings('ignore')


def Logger(content):
    """
    日志打印函数，在分布式训练中只在主进程上打印
    
    参数:
        content: 要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)

def Log_detail(content):
    """
    详细日志记录函数，将日志写入文件，在分布式训练中只在主进程上写入
    
    参数:
        content: 要记录的内容
    """
    if not ddp or dist.get_rank() == 0:
        with open(f'{args.save_dir}/pretrain_{lm_config.dim}.log', 'a') as f:
            f.write(' '.join(content) + '\n')

def get_lr(current_step, total_steps, lr):
    """
    计算学习率衰减，使用余弦衰减策略
    
    参数:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 初始学习率
    
    返回:
        当前步数对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    训练一个epoch的函数
    
    参数:
        epoch: 当前epoch编号
        wandb: wandb实例，用于记录训练指标
    """
    # 定义损失函数，不使用默认的reduction以便于后续应用loss_mask
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    last_time = time.time()
    
    # 遍历训练数据
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移至设备
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，用于忽略填充部分

        # 计算当前步数的学习率（余弦衰减）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度训练上下文
        with ctx:
            # 前向传播，将输入数据传入模型获取输出
            res = model(X)
            # 计算交叉熵损失
            # 将logits展平为[batch_size*seq_len, vocab_size]的形状
            # 将目标Y展平为[batch_size*seq_len]的形状
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # [batch_size*seq_len, vocab_size]
                Y.view(-1)  # [batch_size*seq_len]
            ).view(Y.size())  # 恢复原始形状 [batch_size, seq_len]
            
            # 应用损失掩码并计算平均损失
            # loss_mask用于忽略填充位置的损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 加上辅助损失（如MoE的负载平衡损失等）
            if hasattr(res, 'aux_loss'):
                loss += res.aux_loss
            # 梯度累积：除以累积步数，实现大批次训练效果
            loss = loss / args.accumulation_steps

        # 反向传播（使用梯度缩放器来避免FP16精度下的梯度溢出）
        scaler.scale(loss).backward()

        # 梯度累积：每accumulation_steps步更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 反向缩放梯度值，将梯度恢复到正常大小
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数
            scaler.step(optimizer)
            # 更新缩放因子，自适应调整缩放大小
            scaler.update()
            # 清空梯度，set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)

        # 定期打印训练日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min spend_time:{}'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 显示未缩放的损失
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,  # 预估剩余时间
                    time.time()-last_time  # 当前步耗时
                ))
            # 记录详细日志，包括残差权重（用于分析模型内部状态）
            Log_detail((epoch+1,step,loss.item()*args.accumulation_steps,time.time()-last_time,model.get_residual_weights()))
            # 打印模型参数信息，用于调试
            model.print_model_parameters()
            last_time = time.time()

            # 如果使用wandb且在主进程中，记录指标到wandb平台
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 切换到评估模式（关闭dropout等）
            model.eval()
            # 根据是否使用MoE设置检查点文件名
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}_{epoch}.pth'

            # 获取模型状态字典，处理DDP模型（需要访问module属性）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存模型参数
            torch.save(state_dict, ckp)
            # 切回训练模式
            model.train()


def init_model(lm_config:LMConfig):
    """
    初始化模型和分词器
    
    参数:
        lm_config: 语言模型配置
        
    返回:
        model: 初始化的模型
        tokenizer: 初始化的分词器
    """
    from model_init_utils import ModelInitializer
    return ModelInitializer.init_pretrain_model(lm_config, args.device)


def init_distributed_mode():
    """
    初始化分布式训练环境
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--config", type=str, default="config/pretrain_config.yaml",
                      help="配置文件路径")
    args = parser.parse_args()

    # 从配置文件加载参数
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 将配置参数添加到args对象中
    for key, value in config.items():
        setattr(args, key, value)

    # 根据命令行参数创建语言模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    # 创建输出目录，exist_ok=True表示目录已存在时不报错
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每次迭代处理的token数量（用于估算训练速度）
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 设置随机种子以确保可复现性
    torch.manual_seed(1337)
    # 确定设备类型（用于设置混合精度训练上下文）
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称，包含主要训练参数信息
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 根据设备类型设置混合精度训练上下文
    # 在CPU上使用nullcontext（不做任何操作）
    # 在GPU上使用自动混合精度，可以加速训练同时节省显存
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检测是否在分布式环境中运行（通过环境变量RANK判断）
    ddp = int(os.environ.get("RANK", -1)) != -1
    # 设置默认的本地进程编号和设备
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 如果是分布式训练，初始化分布式环境
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（仅在主进程上）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)
    # import torchinfo
    # torchinfo.summary(model, input_data=torch.randint(0, 100, (args.batch_size, lm_config.max_seq_len)).to(args.device))

    
    # 创建预训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 如果是分布式训练，使用DistributedSampler确保每个进程处理不同数据
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建数据加载器，用于批量加载训练数据
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 将数据加载到固定内存中，可加速GPU传输
        drop_last=False,  # 不丢弃最后一个不完整的批次
        shuffle=False,    # 使用sampler时不需要shuffle
        num_workers=args.num_workers,  # 数据加载的工作线程数
        sampler=train_sampler
    )

    # 创建梯度缩放器，用于混合精度训练（避免FP16下的数值下溢）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 创建AdamW优化器，适合transformer模型训练
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 在分布式训练中设置模型
    if ddp:
        # 指定不参与分布式同步的参数，通常是位置编码等非训练参数
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 将模型包装为DistributedDataParallel实例，实现数据并行训练
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数，用于学习率调整和训练进度估计
    iter_per_epoch = len(train_loader)
    # 训练主循环
    for epoch in range(args.epochs):
        start_t = time.time()
        # 训练一个epoch
        train_epoch(epoch, wandb)
        # 打印epoch总耗时
        Logger(
            'Epoch Total Cost:[{}/{}] time:{:.3f} '.format(
                epoch + 1,
                args.epochs,
                (time.time() - start_t) / 60  # 转换为分钟
            ))