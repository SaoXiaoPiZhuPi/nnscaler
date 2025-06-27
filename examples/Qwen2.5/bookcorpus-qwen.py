# dataset_preprocessor_qwen.py
import argparse
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

def get_qwen_tokenizer(model_path: str) -> AutoTokenizer:
    """Qwen专用分词器初始化"""
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False  # 必须关闭fast模式[7](@ref)
    )

def qwen_tokenize(sample: dict, tokenizer: PreTrainedTokenizer, text_key: str) -> dict:
    """Qwen对话模板编码"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sample[text_key]},
        {"role": "assistant", "content": ""}  # 响应占位符[5](@ref)
    ]
    text = tokenizer.apply_chat_template(  # 应用Qwen专用模板[1](@ref)
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    input_ids = tokenizer.encode(text)
    return {"input_ids": input_ids}

def qwen_concat_split(samples: dict, sample_len: int, pad_token_id: int = None) -> dict:
    """动态分块策略（支持128K长序列）"""
    buffer = []
    resized_ids = []
    lengths = []
    
    for seq in samples["input_ids"]:
        buffer.extend(seq)
        while len(buffer) >= sample_len:
            chunk = buffer[:sample_len]
            resized_ids.append(chunk)
            lengths.append(sample_len)
            buffer = buffer[sample_len:]
    
    # # 保留余数部分（可选）
    # if len(buffer) > 0:
    #     resized_ids.append(buffer)
    #     lengths.append(len(buffer))
    
    if 0 < len(buffer) < sample_len:
        # 填充策略（示例）
        pad_length = sample_len - len(buffer)
        buffer += [pad_token_id] * pad_length  # 使用分词器的pad token
        resized_ids.append(buffer)
        lengths.append(sample_len)
    
    return {
        "input_ids": resized_ids,
        "length": lengths
    }

def create_qwen_dataset(
    tokenizer: PreTrainedTokenizer,
    raw_dataset: Dataset,
    text_key: str = "text",
    sample_len: int = 8192,
    batch_size: int = 10000
) -> Dataset:
    """全流程处理管道"""
    # 第一阶段：文本编码
    tokenized_dataset = raw_dataset.map(
        qwen_tokenize,
        remove_columns=raw_dataset.column_names,
        num_proc=32,
        fn_kwargs={'tokenizer': tokenizer, 'text_key': text_key}
    )
    
    # 第二阶段：动态分块
    return tokenized_dataset.map(
        qwen_concat_split,
        batched=True,
        batch_size=batch_size,
        num_proc=32,
        fn_kwargs={
        'sample_len': 4096,
        'pad_token_id': tokenizer.pad_token_id  # 需从分词器获取[3,7](@ref)
    }
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_or_name', type=str, required=True, 
                       help='数据集路径或HuggingFace名称，如"bookcorpus"')
    parser.add_argument('--tokenizer_path_or_name', type=str, required=True,
                       help='Qwen模型路径，如"Qwen/Qwen2.5-1.5B-Instruct"')
    parser.add_argument('--save_path', type=str, required=True,
                       help='预处理结果保存路径，如"qwen_dataset"')
    parser.add_argument('--sequence_length', type=int, default=8192,
                       help='目标序列长度，建议设为模型最大支持长度[8](@ref)')
    args = parser.parse_args()

    # 加载原始数据集
    raw_dataset = load_dataset(args.data_path_or_name, split="train")
    
    # 初始化Qwen分词器
    tokenizer = get_qwen_tokenizer(args.tokenizer_path_or_name)
    
    # 执行预处理
    processed_dataset = create_qwen_dataset(
        tokenizer=tokenizer,
        raw_dataset=raw_dataset,
        sample_len=args.sequence_length
    )
    
    # 保存处理结果
    processed_dataset.save_to_disk(args.save_path)
    
    # 生成fairseq兼容的长度文件
    sizes = np.array(processed_dataset["length"])
    torch.save(sizes, f"{args.save_path}/lengths.pt")

    print(f"数据集已保存至 {args.save_path}，包含 {len(processed_dataset)} 个样本")