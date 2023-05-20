'''
Author: lihaitao
Date: 2023-05-20 15:06:50
LastEditors: Do not edit
LastEditTime: 2023-05-20 15:08:06
FilePath: /lht/GitHub_code/LexiLaw/inference_finetune.py
'''
from transformers import AutoModel
import torch
import os

from transformers import AutoTokenizer

from peft import PeftModel
import argparse

def generate(text):
    with torch.no_grad():
        input_text = "text"
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).cuda()
        output = peft_model.generate(
            input_ids=input_ids,
            min_length=10,
            max_length=512,
            do_sample=False,
            temperature=0.7,
            num_return_sequences=1
        )[0]
        output = tokenizer.decode(output)
        answer = output.split(input_text)[-1]
    return answer.strip()
    

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_model", type=str, default="/liuzyai04/thuir/lht/context_learning/chatGLM-6B")
    argparser.add_argument("--interactive", default=True)

    args = argparser.parse_args()

    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    

    torch.set_default_tensor_type(torch.cuda.FloatTensor)


    while True:
        text = input("Input: ")
        print(generate(text))
