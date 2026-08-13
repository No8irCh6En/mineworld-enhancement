# 最终总结：投机解码优化历程（2026-08-13 通宵）

## 核心成果

**从"投机逻辑从未触发"到"完整跑通并生成正确视频"**，共修复 8 个关键 bug，并定位了投机未加速的架构级根因。

## 最终状态

| 指标 | baseline | speculative（优化后） |
|------|---------|------|
| 视频生成 | ✅ 15帧 | ✅ 15帧（修复 token 计数后） |
| FPS（第2 demo） | 2.79 | **1.62** |
| HIT 率 | — | 60%（9/15） |

### 速度演进时间线（第二 demo FPS）

| 阶段 | FPS | 改动 |
|------|-----|------|
| 投机逻辑首次跑通 | 0.42 | 修复 8 个 bug |
| + 去 SDPBackend.MATH | 1.20 | 用默认 Flash Attention |
| + 异步流（spec 不重解码） | 1.43 | HIT 直接接受 draft 帧 |
| + merge=False（去 VAE merge） | **1.62** | 用 raw draft token |

累计从 0.42 提升到 1.62 fps（+286%），但仍低于 baseline 2.79 fps（还差 42%）。

## 修复的 Bug 清单（按时间顺序）

1. **投机逻辑从未触发**：`main_frame_done` 判断 `len(row_list)==0`，但 restart 在循环末尾立即填充 → 永远非空。修复：restart 移到 verification 之后。

2. **draft_frame_done 死循环**：spec 流从 action EOS 开始，`state_1_len` 差 1 到不了 336 倍数 → `% 336 == 0` 永远 False。修复：改用 `len(row_list)==0`。

3. **draft_func 拿 335 token**：prefill token 单独存，flush 只存 335。修复：flush 时补 prefill token。

4. **VAE token2image 不支持 batch**：`[K,14,24]` 传给单帧接口。修复：改用 `token2image_gpu`。

5. **torch.compile 卡死**：`action_pred_func` 含 Python 循环/numpy 无法 fullgraph 编译；`decode_some_token` max-autotune 对变长形状爆炸。修复：不编译 action_pred_func，decode 用 dynamic。

6. **SDPBackend.MATH 慢**：强制数学实现注意力。修复：移除，+58% 提速。

7. **token 计数错误（5037/5041 非 336 倍数）**：
   - insert 位置用全局 prefix_counts 而非帧内计数
   - prefill token 重复存储
   - 最后一帧未 flush
   修复：改用帧内 running count，prefill 合并进第一帧 flush，去掉 flush gate。

8. **HIT 路径 token 缺失**：spec 帧 335（缺 draft 第一像素）。修复：HIT 时补 draft 第一像素 token。

## 投机未加速的架构级根因（重要）

**当前投机设计的核心问题**：spec 流用**大模型**（target model）从第一个像素对角线解码 336 步，成本 ≈ Main 正常解码。而 draft 小模型已经预测了整帧，spec 流的对角线解码是重复验证。

正确的投机设计应该是：
1. draft 小模型提议整帧（1 步，快）
2. 大模型 **prefill 一次性验证**（1 步，算整帧 logits 对比 draft）
3. 接受匹配 token，拒绝部分由大模型补全

**当前实现把验证做成了 336 步对角线解码，失去投机意义**，所以净加速 ≈ 0（甚至为负）。

## 若要继续加速，建议方向

1. **验证改为一次性 prefill**：大模型一次 forward 算整帧 logits，与 draft 对比，接受匹配前缀。
2. **draft 模型换轻量架构**：当前 draft 还要 VAE decode + depth 估计（~0.23s/帧），可换纯 token-level 预测。
3. **减少 num_candidates**：实测 K=2 和 K=5 速度一样，说明 batch 不是瓶颈，瓶颈在验证方式。

## 关键文件变更

- `diagonal_decoding.py`：投机解码状态机（frame_completed、token 收集、flush、HIT 补 token）
- `lvm.py`：`speculative_diag_generate_img_token`（去掉 action_pred_func 编译、去 max-autotune、去 MATH）
- `speculative_wrapper.py`：`token2image_gpu` 替换 `token2image`
- `docs/optimization/`：4 篇技术文档
