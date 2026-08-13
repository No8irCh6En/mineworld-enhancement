#!/usr/bin/env python3
"""
修复 JSON 文件中的孤立 UTF-16 代理项（lone surrogates）。

原因：Python 无法用 UTF-8 编码孤立代理项（如 \\ude80），导致
    UnicodeEncodeError: 'utf-8' codec can't encode character '\\ude80' ... surrogates not allowed

处理策略（递归遍历所有字符串字段）：
1. 若高位代理(0xD800-0xDBFF)后面紧跟低位代理(0xDC00-0xDFFF) -> 合并为完整 emoji
2. 若高位代理后没有低位代理 -> 高位代理本身无法确定原字符，替换为占位
3. 孤立低位代理（如 \\ude80，是 🚀 U+1F680 丢失了高位部分）-> 尝试按已知映射补全，
   否则替换为占位说明

用法: python fix_surrogates.py input.json output.json
"""

import json
import sys
import argparse

# 已知的"丢失高位代理"的常见 emoji 低位部分 -> 完整 emoji 映射
# 这里 \ude80 对应 U+1F680 (🚀 ROCKET)，高位为 \ud83d
LOW_TO_FULL = {
    '\ude80': '\ud83d\ude80',  # 🚀 (U+1F680)
}


def fix_string(s: str) -> str:
    """修复单个字符串中的孤立代理项。

    策略：
    - 完整代理对(\\ud83d\\ude80) -> 用 utf-16 编解码转成真正字符 (🚀)
    - 孤立高位代理(后无低位) -> 替换为 [??] 占位
    - 孤立低位代理(如 \\ude80) -> 按 LOW_TO_FULL 补全，否则 [??]
    """
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDBFF:
            # 高位代理
            if i + 1 < n:
                nxt = s[i + 1]
                ncp = ord(nxt)
                if 0xDC00 <= ncp <= 0xDFFF:
                    # 完整代理对 -> 转成真正字符
                    combined = ch + nxt
                    real_char = combined.encode('utf-16', 'surrogatepass').decode('utf-16')
                    out.append(real_char)
                    i += 2
                    continue
            # 高位代理后没有低位代理 -> 无法确定原字符
            out.append('[??]')
            i += 1
            continue
        elif 0xDC00 <= cp <= 0xDFFF:
            # 孤立低位代理
            if ch in LOW_TO_FULL:
                pair = LOW_TO_FULL[ch]
                real_char = pair.encode('utf-16', 'surrogatepass').decode('utf-16')
                out.append(real_char)
            else:
                out.append('[??]')
            i += 1
            continue
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def fix_value(obj):
    """递归修复 dict/list/str 中的孤立代理项。"""
    if isinstance(obj, str):
        return fix_string(obj)
    elif isinstance(obj, dict):
        return {k: fix_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_value(v) for v in obj]
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description='Fix lone surrogates in a JSON file')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('output_file', help='Output JSON file (fixed)')
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed = fix_value(data)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f'Fixed {args.input_file} -> {args.output_file}')


if __name__ == '__main__':
    main()
