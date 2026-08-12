# 突破 1：投机解码流程首次完整跑通

## 日期
2026-08-13

## 状态
✅ 投机解码（Main/Spec 双流 + HIT/MISS 验证 + draft/action 调用）首次端到端跑通，无报错。

## 修复的关键 Bug

### Bug 1：投机逻辑从未触发
**现象**：修复前 speculative 推理能跑，但 HIT/MISS/draft/action 全部 0 次——投机逻辑从未执行，只是"背着 spec 开销的纯对角线解码"。

**根因**：`main_frame_done` 的判断条件 `len(state_row_lists[0]) == 0`，但循环末尾的 restart 逻辑在每轮立即把 row list 填回 `[0]`，导致循环顶部检查时 row list 永远非空 → `main_frame_done` 永远 False。

**修复**：restart 逻辑从"循环末尾立即执行"改为"verification 之后执行"，让空 row list 能持续到下一轮循环顶部被正确检测。

### Bug 2：draft_frame_done 判断错误导致死循环
**现象**：修复 Bug 1 后陷入死循环，`state_1_len` 卡在 335（不是 336）。

**根因**：spec 流从 action 的最后一个 token 开始（`state_row_tokens_lists[1][0] += 1`），导致 spec 的 `state_1_len` 永远差 1 到不了 336 的整数倍。`draft_frame_done = (state_1_len % pixnum == 0)` 永远 False → Main 被永久暂停等 spec。

**修复**：`draft_frame_done` 改用 `len(state_row_lists[1]) == 0`（spec schedule 走完即完成），而非 `state_1_len % pixnum == 0`。

### Bug 3：draft_func 拿到 335 token 而非 336
**现象**：`shape '[1, 14, 24]' is invalid for input of size 335`。

**根因**：每帧的第一个像素 token 由 prefill 产出，存在 `all_tokens_main[1]`，而 `new_tokens_main` 只积累剩余 335 个。flush 时只 flush 了 `new_tokens_main`，丢了 prefill token。

**修复**：flush 时把 prefill token 补回帧首，拼接成完整 336 token。

### Bug 4：VAE token2image 不支持 batch
**现象**：`shape '[1, 14, 24, 64]' is invalid for input of size 107520`。

**根因**：`draft_func` 的 merge 逻辑把 `[K, 14, 24]`（K 个候选）直接传给 `token2image`，但它只支持单帧 `[1, 14, 24]`。

**修复**：改用 `token2image_gpu`（支持 batch `(-1,14,24,64)`，直接返回 `[K,3,H,W]` 的 [-1,1] GPU tensor），同时去掉了 uint8/numpy 转换开销。

### Bug 5：torch.compile 编译卡死
**现象**：`action_pred_func = torch.compile(..., fullgraph=True)` 和 `decode_some_token` 的 `max-autotune` 在投机逻辑生效后卡死数分钟。

**根因**：
1. `action_pred_func` 内含 CPU numpy 转换、Python 字典循环、beam search——`fullgraph=True` 无法编译。
2. `decode_some_token` 的 `max-autotune` 在 Main/Spec 不同 batch 形状下反复编译。

**修复**：
1. `action_pred_func` 和 `draft_func` 不编译。
2. `decode_some_token` 暂时不编译（正确性优先，优化后置）。

## 当前速度

| 模式 | 15帧耗时 | FPS |
|------|---------|-----|
| baseline 对角线解码 | 5.3s | 2.81 |
| speculative（本突破后） | 33.4s | 0.42 |

- HIT 率：9/15 = 60%（投机确实在跳帧）
- 但 draft + action + spec prefill 开销 >> 跳帧收益

## 下一步优化方向

1. draft_func 缓存：上一帧 token→image 的 VAE decode 可以复用
2. merge 的 VAE re-encode 是最大开销（一次 VAE 前向）
3. action predictor 的 CPU numpy 转换
4. spec prefill 与 Main 生成重叠
