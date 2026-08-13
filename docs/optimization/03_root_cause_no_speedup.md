# 突破 3：定位投机未加速的架构级根因

## 日期
2026-08-13

## 核心洞察：当前投机设计为何不加速

### 当前实现
1. **MISS 时**：draft 模型（小模型）一次性预测整帧 336 token → prefill 到 Spec KV cache。
2. **Spec 流激活**：用**大模型**（target model）从第一个像素开始**对角线解码 336 步**，重新生成整帧。
3. **帧边界验证**：对比 action 预测 → HIT 则 Main 跳帧。

### 问题：Spec 流重复了大模型解码
- Spec 流的对角线解码用**大模型**（不是 draft 小模型），336 步，成本 = Main 正常解码。
- draft 小模型已经预测了整帧，但 spec 流又用大模型重新解码一遍。
- **净收益 = 0**：spec 解码（大模型 336 步）≈ Main 解码（大模型 336 步），HIT 跳帧省下的 Main 时间被 spec 解码抵消。

### 正确的投机设计
1. draft 小模型提议整帧（1 步，快）。
2. 大模型 **prefill 验证**（1 步，一次性算整帧 logits，对比 draft 提议）。
3. 接受匹配 token，拒绝则不匹配部分由大模型补全。

**关键**：验证应该用"一次性 prefill"而非"逐 token 对角线解码"。当前代码把验证做成了 336 步对角线解码，失去投机意义。

## 当前速度数据

| 配置 | 第二 demo FPS |
|------|------|
| baseline | 2.81 |
| speculative（当前） | 1.20 |
| speculative + batch 优化 | 1.18（无变化，证明 spec 未激活时间占比小） |

## 已修复的 Bug（本 session）
1. 投机逻辑从未触发（frame_completed 时序）
2. draft_frame_done 死循环（state_1_len 差 1）
3. draft_func 335 token（prefill token 缺失）
4. VAE token2image batch（改 token2image_gpu）
5. torch.compile 卡死（action_pred_func 不编译，去 max-autotune）
6. SDPBackend.MATH 慢（去除，+58%）

## 未修复的 Bug
- **token 计数**：HIT 时 spec 帧少 1 个像素 token（`5037 not multiple of 336`），导致视频无法保存。

## 下一步建议
1. 重构验证逻辑：draft 提议 → 大模型 prefill 一次性验证 → 接受/拒绝。
2. 或保持现状（投机跑通但不加速），先把 token 计数 bug 修好，让视频能生成。
