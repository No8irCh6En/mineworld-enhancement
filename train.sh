# CUDA_VISIBLE_DEVICES=1 deepspeed --master_port=12345 train.py \
#         --data_root "/data/cliang/mineworld/validation/validation" \
#         --deepspeed ./zero3.json \
#         --model_ckpt "/data/jjli/workspace/mineworld/checkpoints/300M_16f.ckpt" \
#         --config "configs/modify.yaml" \
#         --demo_num 1 \
#         --frames 15 \
#         --accelerate-algo 'naive' \
#         --top_p 0.8 \
#         --output_dir "outputs" \
#         --lora_enable False \
#         --data_path ./playground/data/llava_instruct_80k.json \
#         --vision_tower openai/clip-vit-large-patch14 \
#         --mm_vision_select_layer -2 \
#         --mm_use_im_start_end False \
#         --mm_use_im_patch_token False \
#         --bf16 False \
#         --fp16 True \
#         --num_train_epochs 24 \
#         --per_device_train_batch_size 1 \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 16 \
#         --evaluation_strategy "no" \
#         --save_strategy "steps" \
#         --save_steps 50000 \
#         --save_total_limit 1 \
#         --learning_rate 1e-4 \
#         --weight_decay 0.01 \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type "cosine" \
#         --logging_steps 1 \
#         --tf32 False \
#         --model_max_length 2048 \
#         --gradient_checkpointing False \
#         --lazy_preprocess True \
#         --dataloader_num_workers 4 \
#         --report_to tensorboard \
#         --dataset_dir "/data/cliang/mineworld/dataset/" \
#         --frame_height 224 \
#         --frame_width 384 \
#         --train_from_scratch True

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 deepspeed --master_port=12345 train.py \
        --deepspeed ./zero3.json \
        --model_ckpt "/data/jjli/workspace/mineworld/checkpoints/300M_16f.ckpt" \
        --config "configs/modify.yaml" \
        --demo_num 1 \
        --frames 15 \
        --accelerate-algo 'naive' \
        --top_p 0.8 \
        --output_dir "outputs" \
        --lora_enable False \
        --data_path ./playground/data/llava_instruct_80k.json \
        --vision_tower openai/clip-vit-large-patch14 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --bf16 False \
        --fp16 True \
        --num_train_epochs 5 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 0.01 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 False \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --lazy_preprocess True \
        --dataloader_num_workers 4 \
        --report_to tensorboard \
        --dataset_dir "/data/cliang/mineworld/dataset/" \
        --frame_height 224 \
        --frame_width 384 \
        --train_from_scratch True \
        --load_best_model_at_end False