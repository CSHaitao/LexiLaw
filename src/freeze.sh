
TOT_CUDA="4,5,6"
PORT="11458"

    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=3 finetune_freeze.py \
        --train_path /liuzyai04/thuir/lht/context_learning/data/instruction_data/all_instrution.json \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path /liuzyai04/thuir/lht/context_learning/chatGLM-6B \
        --tokenizer_name /liuzyai04/thuir/lht/context_learning/chatGLM-6B \
        --lora_rank 8 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 10 \
        --save_steps 900 \
        --learning_rate 1e-5 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --output_dir /liuzyai04/thuir/lht/context_learning/LORA/output_freeze \
        --deepspeed /liuzyai04/thuir/lht/context_learning/LORA/ds_config.json \

# python -m torch.distributed.launch \
#   --nproc_per_node 1 \
#   --master_port 29508 \
#     finetune.py \
#     --train_path /liuzyai04/thuir/lht/context_learning/data/instruction_data/all_instrution.json \
#     --max_len 700 \
#     --max_input_len 350 \
#     --model_name_or_path /liuzyai04/thuir/lht/context_learning/chatGLM-6B \
#     --tokenizer_name /liuzyai04/thuir/lht/context_learning/chatGLM-6B \
#     --lora_rank 8 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --max_steps 52000 \
#     --save_steps 1000 \
#     --learning_rate 5e-6 \
#     --fp16 \
#     --remove_unused_columns false \
#     --logging_steps 50 \
#     --output_dir /liuzyai04/thuir/lht/context_learning/ChatGLM-Tuning-master/output \






# CUDA_VISIBLE_DEVICES=5



# accelerate launch --main_process_port=29500 --num_processes=2 \
#   --config_file /liuzyai04/thuir/lht/context_learning/ChatGLM-Tuning-master/acce_config/acce.yaml \
#   finetune.py \
#   --train_path /liuzyai04/thuir/lht/context_learning/data/instruction_data/all_instrution.json \
#   --max_len 700 \
#   --max_input_len 350 \
#   --model_name_or_path /liuzyai04/thuir/lht/context_learning/chatGLM-6B \
#   --tokenizer_name /liuzyai04/thuir/lht/context_learning/chatGLM-6B \
#   --lora_rank 8 \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 1 \
#   --max_steps 52000 \
#   --save_steps 1000 \
#   --learning_rate 5e-6 \
#   --fp16 \
#   --remove_unused_columns false \
#   --logging_steps 50 \
#   --output_dir /liuzyai04/thuir/lht/context_learning/ChatGLM-Tuning-master/output \
#   --deepspeed /liuzyai04/thuir/lht/context_learning/ChatGLM-Tuning-master/ds_config.json \

# CUDA_VISIBLE_DEVICES=5
# python3 -m torch.distributed.launch \
#   --nproc_per_node 1 \
#   --master_port 29505 \
#   train_gptj_summarize.py \
#   --fp16 \