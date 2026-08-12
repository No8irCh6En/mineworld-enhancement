#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import csv
import re
import ast
import math
import argparse
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

# 常量（与训练侧保持一致）
IMAGE_TOKEN_LENGTH = 336
ACTION_TOKEN_LENGTH = 11
STRIDE = IMAGE_TOKEN_LENGTH + ACTION_TOKEN_LENGTH  # 347
# 由 224x384 的 16x16 patch 得到 14x24 的网格
GRID_H = 14
GRID_W = 24


def parse_items(cell) -> List[int]:
    """
    将一格单元的字符串/列表解析成 int 列表。
    支持格式：
      - Python 字面量: "[1,2,3]"、"(1,2)"、"set([...])"
      - 纯数字: "5"
      - 以逗号/空格/分号/竖线分隔: "1, 2, 3" / "1 2 3" / "1;2|3"
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if s == "":
        return []
    # 1) 优先尝试 Python 字面量
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, set)):
            out = []
            for x in val:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out
        # 单个数字
        try:
            return [int(val)]
        except Exception:
            pass
    except Exception:
        pass
    # 2) 分隔符解析
    s = s.strip("[](){}")
    parts = re.split(r'[\s,;|]+', s)
    out = []
    for p in parts:
        p = p.strip().strip("'\"")
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out


def find_column_name(fieldnames: List[str]) -> Optional[str]:
    """
    自动寻找最可能存放“相对差 r 列表”的列名。
    """
    # 优先精确匹配
    for key in ('rel_positions', 'real_positions'):
        if key in fieldnames:
            return key
    # 次选：包含 rel 或 pos 的列（不区分大小写）
    for fn in fieldnames:
        low = fn.lower()
        if ('rel' in low) or ('pos' in low):
            return fn
    return None


def _dx_dy_from_indices(q_img_idx: int, t_img_idx: int) -> Tuple[int, int]:
    """
    基于查询与目标在图像内的绝对索引，计算 2D 相对位置 (dx, dy)。
    - 假设索引范围: 0-335 (image tokens)，对应 14x24 网格 (GRID_H=14, GRID_W=24)。
    - 不处理 action tokens (336-346)，因为它们没有 2D 位置。
    - dx = t_x - q_x（列差），dy = t_y - q_y（行差）。
    - 如果索引超出范围，返回 (0, 0) 并打印警告。
    """
    if not (0 <= q_img_idx < IMAGE_TOKEN_LENGTH) or not (0 <= t_img_idx < IMAGE_TOKEN_LENGTH):
        print(f"[WARNING] _dx_dy_from_indices: 索引超出 image 范围 (0-{IMAGE_TOKEN_LENGTH-1})，q={q_img_idx}, t={t_img_idx}，返回 (0,0)")
        return 0, 0

    # 手动计算行/列 (行优先展开: idx = y * GRID_W + x)
    q_y = q_img_idx // GRID_W
    q_x = q_img_idx % GRID_W
    t_y = t_img_idx // GRID_W
    t_x = t_img_idx % GRID_W

    dx = t_x - q_x
    dy = t_y - q_y
    return dx, dy


def specific_analysis(csv_path: str, column: str, top: int = 60, encoding: str = 'utf-8', verbose: bool = False):
    """
    细粒度统计（假设查询 token 固定为 image token 0，即 q_mod=0, q_type='image'）：
      - 从 r 计算 tgt_mod = r % STRIDE，判断目标类型（image 或 action）
      - img_total: 总查询样本数（所有行）
      - act_total: 0（无 action 查询）
      - unknown_total: 0
      - 计数时对每行 items 做去重(set)，避免单行重复多计
      - ImageQ -> ImageK: 统计 1D r 分布 + 3D (frame_delta, dx, dy)
      - ImageQ -> ActionK: 统计 1D r 分布
    """
    try:
        f = open(csv_path, encoding=encoding, newline='')
    except Exception as e:
        print(f"[specific_analysis] 无法打开文件: {e}", file=sys.stderr)
        return

    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        print("[specific_analysis] CSV 无表头，无法解析。", file=sys.stderr)
        f.close()
        return

    # 计数器
    imgq_imgk_1d = Counter()
    imgq_actk_1d = Counter()
    imgq_imgk_3d = Counter()  # (frame_delta, dx, dy)
    actq_actk_1d = Counter()
    actq_imgk_1d = Counter()

    # 分母（所有行都是 image 查询）
    img_total = 0
    act_total = 0  # 无 action 查询
    unknown_total = 0  # 无未知

    # 调试信息
    dbg_total_rows = 0
    dbg_no_items = 0

    for row_index, row in enumerate(reader):  # 新增 row_index
        dbg_total_rows += 1

        items = parse_items(row.get(column))
        # print(f"[DEBUG] items:{items}")
        if not items:
            dbg_no_items += 1
            continue
        # 去重：按“每条查询是否出现该 r”计数
        r_set = set(items)

        # 从行号推断 q_mod 和 q_type
        q_mod = row_index % STRIDE
        q_type = 'image' if q_mod < IMAGE_TOKEN_LENGTH else 'action'

        # 更新分母
        if q_type == 'image':
            img_total += 1
        elif q_type == 'action':
            act_total += 1

        # 对每个 r 计算目标类型
        for r in r_set:
            tgt_linear = q_mod + r
            tgt_mod = tgt_linear % STRIDE
            frame_delta = math.floor(tgt_linear / STRIDE)

            if tgt_mod < IMAGE_TOKEN_LENGTH:
                # 目标是 image
                if q_type == 'image':
                    imgq_imgk_1d[r] += 1
                    # 计算 3D (q_img_idx 只在 q_type == 'image' 时有效)
                    dx, dy = _dx_dy_from_indices(q_mod, tgt_mod)
                    imgq_imgk_3d[(frame_delta, dx, dy)] += 1
                elif q_type == 'action':
                    actq_imgk_1d[r] += 1
            else:
                # 目标是 action
                if q_type == 'image':
                    imgq_actk_1d[r] += 1
                elif q_type == 'action':
                    actq_actk_1d[r] += 1

    f.close()

    def _print_top(counter: Counter, name: str, denom_queries: int, topn: int):
        """
        - denom_queries: 该桶的查询样本数（逐行计一次）
        - 计数使用“每行去重后”的 r 出现次数，故百分比 ≤ 100%
        """
        total_items = sum(counter.values())
        print(f"\n[{name}]")
        print(f"- 查询样本数(denom): {denom_queries}")
        print(f"- 计数项总和(items): {total_items}")
        if denom_queries <= 0:
            denom = 1  # 仅用于显示，避免除零
        else:
            denom = denom_queries
        for i, (k, v) in enumerate(counter.most_common(topn), start=1):
            pct = 100.0 * v / denom
            print(f"{i:>3d}. {k}: {v} ({pct:.2f}% per-query)")

    print("\n================ 细粒度统计（specific_analysis） ================")
    print(f"CSV: {csv_path}, 列: {column}, top N: {top}")

    if verbose:
        print("\n[DEBUG] 扫描统计")
        print(f"- 总行数: {dbg_total_rows}")
        print(f"- 空 items 行数: {dbg_no_items}")
        print(f"- img_total: {img_total}  act_total: {act_total}  unknown_total: {unknown_total}")

    # ImageQ 的各类统计（分母用 img_total）
    _print_top(imgq_imgk_1d, "ImageQ -> ImageK (1D 相对差 r)", denom_queries=img_total, topn=top)
    _print_top(imgq_actk_1d, "ImageQ -> ActionK (1D 相对差 r)", denom_queries=img_total, topn=top)
    _print_top(imgq_imgk_3d, "ImageQ -> ImageK (3D: frame_delta, dx, dy)", denom_queries=img_total, topn=top)

    # ActionQ 的各类统计（分母用 act_total）
    _print_top(actq_imgk_1d, "ActionQ -> ImageK (1D 相对差 r)", denom_queries=act_total, topn=top)
    _print_top(actq_actk_1d, "ActionQ -> ActionK (1D 相对差 r)", denom_queries=act_total, topn=top)


def main():
    p = argparse.ArgumentParser(description="统计 CSV 中相对位置 r 的出现频率，并做细粒度划分")
    p.add_argument("--csv_file", default="attention_weights/analysis.csv", help="CSV 文件路径")
    p.add_argument("--column", "-c", default=None, help="包含相对差 r 列表的列名，默认自动检测")
    p.add_argument("--top", "-n", type=int, default=60, help="输出前 N 个（默认 60）")
    p.add_argument("--encoding", default="utf-8", help="文件编码")
    p.add_argument("--verbose", action="store_true", help="输出调试信息")
    args = p.parse_args()

    # 先做简单整体统计（不区分查询/目标类型）
    try:
        f = open(args.csv_file, encoding=args.encoding, newline='')
    except Exception as e:
        print("无法打开文件:", e, file=sys.stderr)
        sys.exit(2)

    try:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("CSV 没有表头，无法解析。", file=sys.stderr)
            f.close()
            sys.exit(3)

        fieldnames = reader.fieldnames
        col = args.column or find_column_name(fieldnames)
        if col is None:
            print(f"未找到合适的列名，请用 --column 指定。可选列: {fieldnames}", file=sys.stderr)
            f.close()
            sys.exit(4)

        counts = Counter()
        total_rows = 0
        for row in reader:
            items = parse_items(row.get(col))
            if not items:
                continue
            counts.update(items)
            total_rows += 1

    except csv.Error as e:
        print("CSV 解析错误:", e, file=sys.stderr)
        f.close()
        sys.exit(5)
    finally:
        f.close()

    print(f"统计总计行数(非空 items 的查询样本数): {total_rows}")
    print(f"出现频次前 {args.top}：")
    for i, (k, v) in enumerate(counts.most_common(args.top), start=1):
        # 这里的百分比按“每个样本平均出现次数”近似：v / total_rows
        pct = 100.0 * v / max(1, total_rows)
        print(f"{i:>3d}. {k}: {v} ({pct:.2f}% per-query)")

    # 细粒度分析（区分查询/目标类型 + 3D）
    specific_analysis(args.csv_file, column=col, top=args.top, encoding=args.encoding, verbose=args.verbose)


if __name__ == "__main__":
    main()