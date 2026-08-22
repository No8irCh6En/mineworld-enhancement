rm -rf outputs_video/plain_prev_300_best
CUDA_VISIBLE_DEVICES=4 python inference/inference_speculative.py \
        --data_root "small_validation" \
        --model_ckpt "checkpoints/300M_16f.ckpt" \
        --config "configs/modify.yaml" \
        --demo_num 1 \
        --frames 15 \
        --top_p 0.8 \
        --output_dir "outputs_video/plain_prev_300_best" \
        --window_size 2 \
        > log.txt 2>&1