import torch
from transformers import BertTokenizerFast
import pandas as pd
import random
import re

# 保留你原有的辅助函数
def get_special_tokens(tokenizer):
    """获取tokenizer中的所有特殊token"""
    special_tokens = set()
    if hasattr(tokenizer, 'special_tokens_map'):
        for token_type in tokenizer.special_tokens_map.values():
            if isinstance(token_type, list):
                special_tokens.update(token_type)
            else:
                special_tokens.add(token_type)
    return special_tokens

def is_pure_number(token):
    """判断token是否为纯数字（仅由数字组成）"""
    return bool(re.fullmatch(r'\d+', token))

def mask_smiles_tokens(tokens, tokenizer, mask_token="[MASK]"):
    """按照规则遮盖SMILES token列表
    1. 不遮盖：=, #, @, [, ], (, ), 纯数字token, 所有tokenizer特殊token
    2. 优先遮盖()中的token，嵌套括号时优先遮盖最先闭合的
    3. 总遮盖量为可遮盖token的15%，至少1个
    4. 先从括号中遮盖，不足再从其他部分补充
    """
    # 1. 定义不可遮盖的token集合
    fixed_unmaskable = {'=', '#', '@', '@@', '[', ']', '(', ')', '/', '\\','.','+', '-'}
    special_unmaskable = get_special_tokens(tokenizer)
    unmaskable_base = fixed_unmaskable.union(special_unmaskable)
    
    # 2. 筛选可遮盖的token位置
    maskable_positions = []
    for i, token in enumerate(tokens):
        if token in unmaskable_base:
            continue
        if is_pure_number(token):
            continue
        maskable_positions.append(i)
    
    if not maskable_positions:
        return tokens.copy()
    
    # 3. 计算需要遮盖的token数量（15%，至少1个）
    total_maskable = len(maskable_positions)
    num_to_mask = max(1, int(total_maskable * 0.15))
    
    # 4. 识别括号对（处理嵌套括号，优先最内层）
    left_stack = []
    paren_pairs = []
    
    for idx, token in enumerate(tokens):
        if token == '(':
            left_stack.append(idx)
        elif token == ')':
            if left_stack:
                left_pos = left_stack.pop()
                paren_pairs.append((left_pos, idx))
    
    paren_pairs.sort(key=lambda x: x[1])
    
    # 5. 收集括号内的可遮盖token
    paren_maskable = []
    for left_pos, right_pos in paren_pairs:
        in_paren_positions = range(left_pos + 1, right_pos)
        maskable_in_paren = [
            pos for pos in in_paren_positions 
            if pos in maskable_positions
        ]
        paren_maskable.extend(maskable_in_paren)
    
    # 6. 收集非括号内的可遮盖token
    non_paren_maskable = [pos for pos in maskable_positions if pos not in paren_maskable]
    
    # 7. 执行遮盖
    result = tokens.copy()
    masked_count = 0
    positions_to_mask = []
    
    if paren_maskable:
        take_from_paren = min(num_to_mask, len(paren_maskable))
        positions_to_mask.extend(random.sample(paren_maskable, take_from_paren))
        masked_count = take_from_paren
    
    if masked_count < num_to_mask and non_paren_maskable:
        take_from_non_paren = num_to_mask - masked_count
        positions_to_mask.extend(random.sample(non_paren_maskable, take_from_non_paren))
    
    for pos in positions_to_mask:
        result[pos] = mask_token
    
    return result, positions_to_mask  # 返回遮盖位置用于创建标签

def smiles_to_token_tensors(smiles_list, tokenizer, max_length=32):
    """
    将SMILES列表转换为token张量并应用遮盖操作
    
    参数:
        smiles_list (list): SMILES字符串列表
        tokenizer: BertTokenizerFast实例
        max_length (int): 最大序列长度
        
    返回:
        dict: 包含以下键的字典:
            - input_ids: 原始token的整数ID张量
            - masked_input_ids: 遮盖后的token整数ID张量
            - attention_mask: 注意力掩码张量
            - labels: 用于MLM训练的标签张量（仅遮盖位置有原始ID，其余为-100）
    """
    # 1. 批量编码SMILES获取基础信息
    encodings = tokenizer(
        smiles_list,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encodings["input_ids"].clone()
    attention_mask = encodings["attention_mask"]
    batch_size, seq_len = input_ids.shape
    
    # 2. 初始化遮盖后的输入和标签
    masked_input_ids = input_ids.clone()
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # -100在PyTorch中会被忽略
    
    # 3. 对每个样本进行处理
    for i in range(batch_size):
        # 将当前样本的ID转换为token列表
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
        # print(tokens)
        
        # 应用遮盖操作
        masked_tokens, mask_positions = mask_smiles_tokens(tokens, tokenizer)
        # print(masked_tokens) # F
        # print(mask_positions) # 27
        
        # 将遮盖后的token转换为ID
        masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens) # print(masked_ids)
        masked_input_ids[i] = torch.tensor(masked_ids, dtype=torch.long)
        
        # 设置标签（仅遮盖位置保留原始ID）
        for pos in mask_positions:
            labels[i, pos] = input_ids[i, pos]
    
    return {
        "input_ids": input_ids,
        "masked_input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 示例用法
if __name__ == "__main__":
    # 加载数据和tokenizer
    df = pd.read_parquet('train.parquet')
    cano_smiles_list = list(set(df['canonical_smi']))[:3]  # 取前10个作为示例
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # 转换为token张量并应用遮盖
    result_tensors = smiles_to_token_tensors(
        smiles_list=cano_smiles_list,
        tokenizer=tokenizer,
        max_length=32
    )
    
    # 查看结果
    print("原始input_ids形状:", result_tensors["input_ids"])#.shape
    print("遮盖后input_ids形状:", result_tensors["masked_input_ids"])#.shape
    print("注意力掩码形状:", result_tensors["attention_mask"])#.shape
    print("标签形状:", result_tensors["labels"])#.shape
    
    # 打印第一个样本的部分结果进行验证
    idx = 0
    print("\n第一个样本原始tokens:", tokenizer.convert_ids_to_tokens(result_tensors["input_ids"][idx].tolist()))
    print("第一个样本遮盖后tokens:", tokenizer.convert_ids_to_tokens(result_tensors["masked_input_ids"][idx].tolist()))
    print("第一个样本标签（仅非-100部分）:", result_tensors["labels"][idx][result_tensors["labels"][idx] != -100])

    idx = 1
    print("\n第一个样本原始tokens:", tokenizer.convert_ids_to_tokens(result_tensors["input_ids"][idx].tolist()))
    print("第一个样本遮盖后tokens:", tokenizer.convert_ids_to_tokens(result_tensors["masked_input_ids"][idx].tolist()))
    print("第一个样本标签（仅非-100部分）:", result_tensors["labels"][idx][result_tensors["labels"][idx] != -100])

    idx = 2
    print("\n第一个样本原始tokens:", tokenizer.convert_ids_to_tokens(result_tensors["input_ids"][idx].tolist()))
    print("第一个样本遮盖后tokens:", tokenizer.convert_ids_to_tokens(result_tensors["masked_input_ids"][idx].tolist()))
    print("第一个样本标签（仅非-100部分）:", result_tensors["labels"][idx][result_tensors["labels"][idx] != -100])
