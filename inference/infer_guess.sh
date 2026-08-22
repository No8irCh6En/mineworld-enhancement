CUDA_VISIBLE_DEVICES=4 python inference/infer_with_guess.py \
        --data_root "/data/cliang/mineworld/validation/small_validation" \
        --model_ckpt "/home/cliang/mineworld/outputs/noise_with_bias" \
        --config "configs/modify.yaml" \
        --demo_num 1 \
        --frames 15 \
        --top_p 0.8 \
        --output_dir "/home/cliang/mineworld/outputs_video/5_step" \
        --guess_step 5 \
        > log.txt 2>&1