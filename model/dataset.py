import json               # 用于解析和处理JSON格式数据
import random             # 提供随机数生成功能
import re                 # 正则表达式库，用于文本处理

import pandas as pd       # 数据分析处理库
import numpy as np        # 科学计算库
from torch.utils.data import Dataset, DataLoader  # PyTorch数据加载工具
import torch              # PyTorch深度学习库
from sklearn.model_selection import train_test_split  # 数据集分割工具
import os                 # 操作系统功能接口
import ast                # 用于解析Python抽象语法树

# 禁用tokenizers的并行处理，避免多进程数据加载时的潜在问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    预训练数据集类，用于处理语言模型的预训练数据
    继承自PyTorch的Dataset类，提供数据加载和处理功能
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集
        
        参数:
            data_path: JSONL格式的数据文件路径
            tokenizer: 分词器，用于将文本转换为token ID
            max_length: 序列最大长度，默认为512
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        从JSONL文件加载数据
        
        参数:
            path: 数据文件路径
            
        返回:
            包含所有样本的列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 解析JSON行数据
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        参数:
            index: 样本索引
            
        返回:
            处理后的样本，包括输入序列X、目标序列Y和损失掩码loss_mask
        """
        sample = self.samples[index]

        # 构建输入文本，添加开始和结束标记
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        # 使用分词器编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',  # 填充到最大长度
            truncation=True,       # 截断过长的序列
            return_tensors='pt'    # 返回PyTorch张量
        )
        # 获取编码后的token ID序列
        input_ids = encoding.input_ids.squeeze()
        # 创建损失掩码，用于区分真实token和填充token
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 将序列错位，用于下一个token预测任务
        # X是输入序列，不包含最后一个token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        # Y是目标序列，不包含第一个token，用于模型学习预测下一个token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # 损失掩码也需要向右偏移，与Y对齐
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    监督微调(Supervised Fine-Tuning)数据集类
    用于处理对话格式的数据，适用于对话模型的微调
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化SFT数据集
        
        参数:
            jsonl_path: JSONL格式的数据文件路径
            tokenizer: 分词器，用于将文本转换为token ID
            max_length: 序列最大长度，默认为1024
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 获取模型中助手回复开始和结束的特殊token ID
        self.bos_id = tokenizer('<s>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>', add_special_tokens=False).input_ids

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.samples)

    def load_data(self, path):
        """
        从JSONL文件加载对话数据
        
        参数:
            path: 数据文件路径
            
        返回:
            包含所有对话样本的列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        构建符合ChatML格式的对话
        
        参数:
            conversations: 对话内容列表
            
        返回:
            格式化后的对话文本
        """
        messages = []
        for i, turn in enumerate(conversations):
            # 基于轮次确定角色，奇数轮次为用户，偶数轮次为助手
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        # 应用分词器的聊天模板格式化对话
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,              # 不执行分词
            add_generation_prompt=False  # 不添加生成提示
        )

    def _generate_loss_mask(self, input_ids):
        """
        生成损失掩码，只计算助手回复部分的损失
        
        参数:
            input_ids: 输入序列的token ID
            
        返回:
            损失掩码，1表示计算损失，0表示不计算
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 寻找助手回复的开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 助手回复开始位置
                start = i + len(self.bos_id)
                end = start
                # 寻找助手回复的结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 为助手回复部分设置损失掩码为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 移动到下一个对话
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        参数:
            index: 样本索引
            
        返回:
            处理后的样本，包括输入序列X、目标序列Y和损失掩码loss_mask
        """
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # 使用分词器转换为token ID，并截断到最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 填充到最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码，只对助手回复部分计算损失
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入序列
        Y = torch.tensor(input_ids[1:], dtype=torch.long)   # 目标序列
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    直接偏好优化(Direct Preference Optimization)数据集类
    用于RLHF的偏好学习，处理包含偏好选择的数据
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化DPO数据集
        
        参数:
            file_path: 包含偏好数据的JSONL文件路径
            tokenizer: 分词器，用于将文本转换为token ID
            max_length: 序列最大长度，默认为4096
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 设置填充token ID
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 获取助手回复的特殊标记
        self.bos_id = tokenizer('<s>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>', add_special_tokens=False).input_ids
        # 加载偏好数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        参数:
            index: 样本索引
            
        返回:
            处理后的样本，包括被选择和被拒绝的回复对
        """
        item = self.data[index]
        # 被人类偏好选择的回复
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        # 被人类拒绝的回复
        rejected = item['rejected']  # 同上
        
        # 使用分词器的聊天模板格式化对话
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # 将对话文本转换为token ID序列
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 获取token ID序列
        chosen_input_ids = chosen_encoding['input_ids']
        # 生成损失掩码，只对助手回复部分计算损失
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        # 准备模型输入
        # 偏移序列以进行自回归训练
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # 返回包含所有必要数据的字典
        return {
            'x_chosen': x_chosen,      # 被选择回复的输入序列
            'y_chosen': y_chosen,      # 被选择回复的目标序列
            'mask_chosen': mask_chosen,  # 被选择回复的损失掩码
            'x_rejected': x_rejected,    # 被拒绝回复的输入序列
            'y_rejected': y_rejected,    # 被拒绝回复的目标序列
            'mask_rejected': mask_rejected  # 被拒绝回复的损失掩码
        }

    def _generate_loss_mask(self, input_ids):
        """
        生成损失掩码，只计算助手回复部分的损失
        
        参数:
            input_ids: 输入序列的token ID
            
        返回:
            损失掩码，1表示计算损失，0表示不计算
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 寻找助手回复的开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 寻找助手回复的结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 为助手回复部分设置损失掩码为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 移动到下一个对话
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


if __name__ == "__main__":
    pass
