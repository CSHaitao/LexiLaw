'''
Author: lihaitao
Date: 2023-05-08 17:12:01
LastEditors: Do not edit
LastEditTime: 2023-05-09 14:10:26
FilePath: /lht/ChatGLM_LoRA/data.py
'''
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DefaultDataCollator,PreTrainedTokenizer
class InstrutionDataset(Dataset):
    def __init__(self, data_path,prefix=""):
        self.dataset = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())                
                self.dataset.append(
                    {"input": prefix + sample["instruction"] + sample["input"],"answer": sample["answer"]})
            


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):   
        return self.dataset[item]
    


class InstrutionCollator(DefaultDataCollator):
    def __init__(self, tokenizer,max_len,max_input_len):
        self.max_len = max_len
        self.max_input_len = max_input_len
        self.tokenizer = tokenizer

    def __post_init__(self):
        super().__post_init__()
        self.rng = random.Random()

    def __call__(self, examples):
        input_ids_list = []
        labels_list = []
        max_tgt_len = self.max_len - self.max_input_len - 3
        for example in examples:
            input = example["input"]
            answer = example["answer"]
            src_tokens = self.tokenizer.tokenize(input)
            if len(src_tokens) > self.max_input_len:
                src_tokens = src_tokens[:self.max_input_len]
            tgt_tokens = self.tokenizer.tokenize(answer)
            if len(tgt_tokens) > max_tgt_len:
                tgt_tokens = tgt_tokens[:max_tgt_len]
            tokens = src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]


            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            context_length = input_ids.index(self.tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            labels_list.append(torch.LongTensor(labels))
            input_ids_list.append(torch.LongTensor(input_ids))
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }



    
