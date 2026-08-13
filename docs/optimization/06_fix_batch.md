# 突破 5：修复 spec row list 空时仍 packed forward（1.62 → 2.22 fps）

## 日期
2026-08-13

## 关键 bug

异步流重构后，spec 流的 row list 立即置空（draft 帧已 prefill），但 `spec_active=True` 保持。导致主循环里 `if spec_active:` 分支仍然走 packed forward，把 Main 从 batch=1 扩展到 batch=K（即使 spec 没有 token 要解码），浪费 40% 算力。

## 修复

```python
# 修复前
if spec_active:
    input_0 = input_0.expand(input_1.shape[0], -1)  # 即使 input_1 是空 [K,0]
    packed_input = torch.cat([input_0, input_1], dim=1)

# 修复后
spec_has_tokens = (len1 > 0)  # len1 = input_1.shape[1]
if spec_has_tokens:
    ... packed forward
else:
    packed_input = input_0  # batch=1
```

## 速度

| 配置 | 第二 demo FPS |
|------|------|
| baseline | 2.79 |
| 投机（merge=False） | 1.62 |
| **投机（修复 batch）+** | **2.22（+37%）** |

## 当前状态

- 2.22 vs 2.79，只差 20%
- HIT 60%（9/15），确实在跳帧
- draft(0.05s) + prefill(0.02s) + action(0.003s) × 12 MISS ≈ 0.87s 额外开销
- restore_kv_cache 很快（0.002s）

## 剩余瓶颈

1. draft_func 的 VAE decode prev + depth（~0.05s/帧）
2. HIT 率 60%（action predictor 准确率限制）
3. 要达到 baseline 以上，需要 draft 更快或 HIT 率更高
