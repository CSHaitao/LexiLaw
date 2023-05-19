
###
 # @Author: lihaitao
 # @Date: 2023-05-08 17:17:26
 # @LastEditors: Do not edit
 # @LastEditTime: 2023-05-09 13:59:44
 # @FilePath: /lht/ChatGLM_LoRA/lora.sh
### 
TOT_CUDA="1,2,3"
PORT="11451"

    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=3 finetune_lora.py \
        --train_path ./instrution_data.json \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path ./chatGLM-6B \
        --tokenizer_name ./chatGLM-6B \
        --lora_rank 8 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 10 \
        --save_steps 900 \
        --learning_rate 1e-5 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --output_dir /output \
        --deepspeed /ds_config.json \

