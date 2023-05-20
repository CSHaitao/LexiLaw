'''
Author: lihaitao
Date: 2023-05-20 15:06:50
LastEditors: Do not edit
LastEditTime: 2023-05-20 19:34:44
'''
from transformers import AutoModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
import torch
from peft import PeftModel
import argparse

def generate(model,tokenizer,text):
    with torch.no_grad():
        input_text = text
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).cuda()
        output = model.generate(
            input_ids=input_ids,
            min_length=20,
            max_length=512,
            do_sample=False,
            temperature=0.7,
            num_return_sequences=1
        )[0]
        output = tokenizer.decode(output)
        # answer = output.split(input_text)[-1]
    return output.strip()
    

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_model", type=str, default="model/LexiLaw")
    argparser.add_argument("--adapter", type=str, default="model/adapter")
    argparser.add_argument("--interactive", default=True)

    args = argparser.parse_args()

    model = ChatGLMForConditionalGeneration.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = ChatGLMTokenizer.from_pretrained("model/LexiLaw", trust_remote_code=True)
    
    peft_model = PeftModel.from_pretrained(model, args.adapter).eval()

    print(torch.cuda.device_count())
    print(torch.cuda.current_device())

    peft_model.half().cuda()
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


    while True:
        text = input("Input: ")
        print(generate(peft_model,tokenizer,text))
