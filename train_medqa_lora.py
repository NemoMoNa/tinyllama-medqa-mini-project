#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 07:58:41 2025

@author: mh
"""

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# ✅ ステップ 1: データ読み込み（試験的に100件）
dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train[:100]")

# ✅ ステップ 2: Chat形式（仮想）に変換
def to_chat_format(example):
    prompt = f"{example['instruction']}\n\n{example['input']}"
    return {
        "text": f"<s>[INST] {prompt} [/INST] {example['output']}</s>"
    }

# データ変換＆Dataset化
chat_dataset = dataset.map(to_chat_format)
hf_dataset = Dataset.from_list(chat_dataset)

# ✅ ステップ 3: モデル＆トークナイザー読み込み（ChatTemplate対応モデル）
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # 将来互換を意識して "auto" に
)

# ✅ ステップ 4: LoRA設定（軽量fine-tuning）
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# ✅ ステップ 5: トークナイズ（バッチ対応）
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = hf_dataset.map(tokenize, batched=True)

# ✅ ステップ 6: 学習設定（軽量環境用）
training_args = TrainingArguments(
    output_dir="./tinyllama-medqa-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none"
)

# ✅ ステップ 7: Trainer定義（tokenizer引数は不要）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# ✅ ステップ 8: 学習＆保存
trainer.train()
model.save_pretrained("./tinyllama-medqa-lora")
tokenizer.save_pretrained("./tinyllama-medqa-lora")
