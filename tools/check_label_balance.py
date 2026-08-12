import argparse
import os
import glob
import numpy as np
import random
from tqdm import tqdm

def calculate_balance(label_dir, episode_names, set_name):
    total_zeros = 0
    total_ones = 0
    total_pixels = 0
    
    print(f"Checking {set_name} set ({len(episode_names)} episodes)...")
    
    for ep_name in tqdm(episode_names):
        label_path = os.path.join(label_dir, f"{ep_name}_labels.npy")
        if not os.path.exists(label_path):
            continue
            
        try:
            # Load labels. Shape is typically (NumFrames, H, W, 2)
            # Channel 0 is the classification label (0 or 1)
            labels = np.load(label_path)
            
            # Extract classification channel
            cls_labels = labels[..., 0]
            
            # Count
            ones = np.sum(cls_labels == 1)
            zeros = np.sum(cls_labels == 0)
            
            total_ones += ones
            total_zeros += zeros
            total_pixels += (ones + zeros)
            
        except Exception as e:
            print(f"Error reading {label_path}: {e}")

    if total_pixels == 0:
        print(f"[{set_name}] No data found.")
        return

    percent_ones = (total_ones / total_pixels) * 100
    percent_zeros = (total_zeros / total_pixels) * 100
    
    print(f"\n=== {set_name} Results ===")
    print(f"Total Pixels Checked: {total_pixels}")
    print(f"Count 0 (No Change): {total_zeros} ({percent_zeros:.2f}%)")
    print(f"Count 1 (Changed)  : {total_ones} ({percent_ones:.2f}%)")
    if total_ones > 0:
        print(f"Ratio 0:1          : {total_zeros/total_ones:.2f} : 1")
    else:
        print(f"Ratio 0:1          : N/A (No 1s found)")
    print("==========================\n")

def main():
    parser = argparse.ArgumentParser(description="Check label balance (0 vs 1) in dataset.")
    parser.add_argument("--label_dir", type=str, default="/data/cliang/mineworld/misalignment_dataset_labels", help="Path to label directory")
    parser.add_argument("--max_episodes", type=int, default=-1, help="Limit number of episodes to check (for speed)")
    args = parser.parse_args()

    if not os.path.exists(args.label_dir):
        print(f"Error: Label directory {args.label_dir} does not exist.")
        return

    # 1. Get all episodes
    all_label_files = sorted(glob.glob(os.path.join(args.label_dir, "*_labels.npy")))
    all_ep_names = [os.path.basename(f).replace("_labels.npy", "") for f in all_label_files]
    
    if len(all_ep_names) == 0:
        print("No label files found.")
        return

    # 2. Shuffle and Split (Same logic as training script)
    # Using seed 42 to match train_pred_with_attn.py
    random.seed(42)
    random.shuffle(all_ep_names)
    
    if args.max_episodes > 0:
        all_ep_names = all_ep_names[:args.max_episodes]
        
    # 90/10 Split
    split_idx = int(0.9 * len(all_ep_names))
    train_ep_names = all_ep_names[:split_idx]
    val_ep_names = all_ep_names[split_idx:]
    
    print(f"Total Episodes Found: {len(all_ep_names)}")
    print(f"Train Split: {len(train_ep_names)}")
    print(f"Val Split  : {len(val_ep_names)}\n")

    # 3. Calculate stats
    calculate_balance(args.label_dir, train_ep_names, "Train")
    calculate_balance(args.label_dir, val_ep_names, "Validation")

if __name__ == "__main__":
    main()