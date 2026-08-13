# 突破 4：异步流重构（spec 不重解码，HIT 直接接受 draft 帧）

## 日期
2026-08-13

## 核心改动

按用户想法实现异步流：**draft 小模型预测整帧（1 步前向）→ prefill 到 spec KV cache → spec 流不再用大模型逐 token 对角线解码 336 步 → 帧边界 action 验证 → HIT 直接接受 draft 整帧**。

### 具体修改
1. `draft_full_frames`：保存 draft 的完整 336 token（`[K, 336]`）。
2. spec 激活后：row list 立即置空（`state_row_lists[1] = []`），`state_1_len = pixnum`（标记完成），**不再进入对角线解码**。
3. HIT 分支：直接 `all_tokens_main.append(draft_full_frames[hit].tolist())`，不再用 spec 流解码的 token 重建。

## 速度变化

| 配置 | 第二 demo FPS |
|------|------|
| baseline | 2.79 |
| 投机（修复 token 计数后） | 1.17 |
| **投机（异步流重构后）** | **1.43（+22%）** |

- HIT 9/15 = 60%
- 视频正确生成（15帧）

## 剩余瓶颈分析

异步流后，HIT 不再重复解码，但仍有开销：
1. **draft_func ~0.23s/帧**（VAE decode + depth + draft forward + VAE merge）× 12 MISS ≈ 2.8s
2. **spec prefill ~0.025s/帧** × 12 ≈ 0.3s
3. **action prediction ~0.004s/帧** × 12 ≈ 0.05s

总计 ~3.1s 额外开销，而 HIT 省下的是 Main 的解码时间（9 次 HIT × 0.36s = 3.2s）。

**当前接近打平**：省下的 ≈ 额外开销。要真正加速，需要：
1. draft_func 更快（去掉 VAE merge，直接用 raw draft token）
2. 或提高 HIT 率（更好的 action predictor）
3. 或 HIT 时 Main 也跳过（更深流水线）
