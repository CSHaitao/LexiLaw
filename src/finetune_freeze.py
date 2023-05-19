

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import sys
# sys.path.append('/liuzyai04/thuir/lht/context_learning/peft-main/src/peft')
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser,default_data_collator
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import logging
from modeling_chatglm import ChatGLMForConditionalGeneration
from data import InstrutionDataset,InstrutionCollator
from arguments import ModelArguments, DataTrainingArguments, FinetuneArguments as TrainingArguments
from transformers.trainer_utils import is_main_process
import transformers
from datasets import load_dataset
from tokenization_chatglm import ChatGLMTokenizer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")




class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss


def main():
    writer = SummaryWriter()


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  
    )


    model = model.half()
    for name, param in model.named_parameters():
        if not any(nd in name for nd in ["layers.27", "layers.26", "layers.25", "layers.24", "layers.23"]):
            param.requires_grad = False
    print_trainable_parameters(model)


    ## data
    train_data = InstrutionDataset(
        data_path = data_args.train_path)
    
    data_collator = InstrutionCollator(
        tokenizer=tokenizer,
        max_len = training_args.max_len,
        max_input_len=training_args.max_input_len
    )



    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    

    trainer.train()
    writer.close()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
