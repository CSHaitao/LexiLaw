
###
 # @Author: lihaitao
 # @Date: 2023-05-09 13:57:18
 # @LastEditors: Do not edit
 # @LastEditTime: 2023-05-09 13:58:24
 # @FilePath: /lht/ChatGLM_LoRA/ptuning.sh
### 
TOT_CUDA="0,7"
PORT="11400"
PRE_SEQ_LEN=64

    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=2 finetune_ptuning.py \
        --train_path ./instrution_data.json \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path /chatGLM-6B \
        --tokenizer_name/chatGLM-6B \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 10 \
        --save_steps 2000 \
        --learning_rate 1e-5 \
        --fp16 \
        --logging_steps 50 \
        --prefix_projection True \
        --pre_seq_len $PRE_SEQ_LEN \
        --output_dir /liuzyai04/thuir/lht/context_learning/Ptuning/output \
        --deepspeed /liuzyai04/thuir/lht/context_learning/Ptuning/ds_config.json \

