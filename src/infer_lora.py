'''
Author: lihaitao
Date: 2023-05-08 17:15:42
LastEditors: Do not edit
LastEditTime: 2023-05-08 17:16:19
FilePath: /lht/ChatGLM_LoRA/infer.py
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from transformers import AutoTokenizer, AutoModel
from tokenization_chatglm import ChatGLMTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration
model = ChatGLMForConditionalGeneration.from_pretrained("./chatGLM-6B",trust_remote_code=True)
tokenizer = ChatGLMTokenizer.from_pretrained("./chatGLM-6B", trust_remote_code=True)


model = model.eval()
from peft import PeftModel
import torch
model = PeftModel.from_pretrained(model, './checkpoint-2700',torch_dtype=torch.float32).eval()
model.half().cuda()

input_text = '中华人民共和国刑法第七条'
input_text = tokenizer.tokenize(input_text)
input_text = input_text + ["[gMASK]", "<sop>"]
ids = tokenizer.convert_tokens_to_ids(input_text)
print(ids)
input_ids = torch.LongTensor([ids]).cuda()
print(input_ids)
generation_kwargs = {
                    "min_length": 10,
                    "max_new_tokens": 150,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 1,
                }

with torch.no_grad():
    out = model.generate(
                input_ids=input_ids,**generation_kwargs
            )
out_text = tokenizer.decode(out[0])
print(out_text)
