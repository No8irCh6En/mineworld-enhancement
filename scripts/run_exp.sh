#!/usr/bin/env bash
# 在指定 GPU 上跑 baseline（对角线解码）或投机解码，采集 FPS 数据。
# 用法: ./run_exp.sh <gpu_id> <mode> <output_dir>
#   mode: baseline | spec | oracle | profile
set -e
GPU=$1
MODE=$2
OUT=$3
LOG=$4

cd /data/cliang/workspace/mineworld
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torch/inductor
export TRITON_CACHE_DIR=~/.triton/cache

DATA_ROOT="/data/cliang/mineworld/validation/small_validation"

case "$MODE" in
  baseline)
    CUDA_VISIBLE_DEVICES=$GPU python inference.py \
      --data_root "$DATA_ROOT" \
      --model_ckpt "checkpoints/300M_16f.ckpt" \
      --config "configs/modify.yaml" \
      --demo_num 1 --save_frames \
      --frames 15 \
      --accelerate-algo "image_diagd" \
      --top_p 0.8 \
      --output_dir "$OUT" \
      --window_size 2 > "$LOG" 2>&1
    ;;
  spec)
    CUDA_VISIBLE_DEVICES=$GPU python inference_speculative.py \
      --data_root "$DATA_ROOT" \
      --model_ckpt "checkpoints/300M_16f.ckpt" \
      --config "configs/modify.yaml" \
      --demo_num 1 --save_frames \
      --frames 15 \
      --top_p 0.8 \
      --output_dir "$OUT" \
      --window_size 2 > "$LOG" 2>&1
    ;;
  oracle)
    CUDA_VISIBLE_DEVICES=$GPU python inference_speculative.py \
      --data_root "$DATA_ROOT" \
      --model_ckpt "checkpoints/300M_16f.ckpt" \
      --config "configs/modify.yaml" \
      --demo_num 1 --save_frames \
      --frames 15 \
      --top_p 0.8 \
      --output_dir "$OUT" \
      --window_size 2 --use_oracle > "$LOG" 2>&1
    ;;
  profile)
    PROFILE=1 CUDA_VISIBLE_DEVICES=$GPU python inference_speculative.py \
      --data_root "$DATA_ROOT" \
      --model_ckpt "checkpoints/300M_16f.ckpt" \
      --config "configs/modify.yaml" \
      --demo_num 1 \
      --frames 15 \
      --top_p 0.8 \
      --output_dir "$OUT" \
      --window_size 2 > "$LOG" 2>&1
    ;;
esac
