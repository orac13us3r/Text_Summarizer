from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader


def load_and_prepare_data():
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]") 
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    def preprocess_function(example):
        input_text = "summarize: " + example["article"]
        target_text = example["highlights"]
        input_ids = tokenizer(input_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt").input_ids
        target_ids = tokenizer(target_text, max_length=150, truncation=True, padding="max_length", return_tensors="pt").input_ids
        return {"input_ids": input_ids, "target_ids": target_ids}

    dataset = dataset.map(preprocess_function)
    dataset.set_format(type="torch", columns=["input_ids", "target_ids"])
    return DataLoader(dataset, batch_size=4, shuffle=True), tokenizer
