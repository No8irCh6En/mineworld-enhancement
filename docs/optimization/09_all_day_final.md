# 最终完整总结（02:00 → 16:00 全天迭代）

## 最终结果

| 指标 | baseline | 投机（最终） |
|------|---------|-------------|
| FPS（第2 demo，GPU 7 RTX 3090） | 2.79 | **2.66** |
| 视频生成 | ✅ 15帧 | ✅ 15帧 |
| HIT 率 | — | 67%（10/15） |
| 无报错 | ✅ | ✅ |
| PSNR vs GT | 18.58 dB | 18.14 dB（仅差 0.44 dB） |

**投机从 0.42 fps（无法运行）→ 2.66 fps（baseline 95%），累计 +533%，且画面质量几乎无损（PSNR 仅降 0.44 dB）。**

## 速度演进完整时间线

| 阶段 | FPS | 关键改动 | commit |
|------|-----|---------|--------|
| 起点（投机从未触发） | 0.42 | — | — |
| +修复 8 bug | 0.84 | 投机真正运行 | cf158c5 |
| +去 SDPBackend.MATH | 1.20 | Flash Attention | cf158c5 |
| +异步流 | 1.43 | spec 不重解码 | 9c830b9 |
| +merge=False | 1.62 | 去 VAE merge | 9c830b9 |
| +batch 修复 | 2.22 | spec 空时 batch=1 | 8849d63 |
| +max-autotune | 2.36 | 匹配 baseline 步速 | 63d8c89 |
| +容错匹配 | 2.38 | camera ±2 bin | f77b3e0 |
| +CPU 计数 | 2.56 | 消除 .item() 同步 | ec7b84f |
| +pos buffer | 2.66 | 消除 torch.tensor(list) | 0b19dc6 |

## 10 个关键 Bug

1. 投机逻辑从未触发（frame_completed 时序错误）
2. draft_frame_done 死循环（state_1_len 差 1）
3. draft_func 拿 335 token（prefill token 缺失）
4. VAE token2image 不支持 batch
5. torch.compile 卡死（action_pred_func 含 Python 循环）
6. SDPBackend.MATH 慢 58%
7. token 计数错误（5037 非 336 倍数）
8. HIT 路径 token 缺失
9. .item() GPU 同步 8000 次
10. torch.tensor(list, cuda) 9s CUDA malloc

## 性能瓶颈定位（profile 证据）

### 瓶颈 1：update 9.9s → .item() 同步
`_update_stream_state` 里每个 token 调 `.item()`，8000 次 GPU→CPU 同步。

### 瓶颈 2：pos 9.1s → torch.tensor(list, cuda)
每步 `torch.tensor(pos_ids, device="cuda")` 触发 CUDA malloc + H2D + sync。

### 瓶颈 3：decode 慢 7.7 倍 → dynamic vs max-autotune
异步流后 batch 固定，max-autotune 恢复 baseline 的每步速度。

### 剩余 5%：投机循环固有开销
- 每步 10.5ms vs baseline 6.4ms（多 4ms GPU 同步）
- draft/action/prefill 0.055s/帧
- verification Python 开销
- HIT 跳帧不彻底（省 34 步而非 60 步）

## 架构级结论

### 投机未超越 baseline 的数学

- HIT 67%：10 帧跳帧
- draft 开销 0.045s × 15 = 0.68s
- 主循环 Python 额外 4ms × 500 步 = 2s
- 净效果：跳帧收益 ≈ 开销，打平

### 要超越 baseline 的路径

1. **HIT 率 > 85%**：更好的 action predictor（模型质量）
2. **draft < 0.02s**：token-level draft（不用 VAE decode）
3. **HIT 完全跳帧**：Main 在 HIT 后完全不解码被跳帧

## 你的两个想法实现情况

1. **异步流** ✅ 已实现：spec 不再用大模型重解码，HIT 直接接受 draft 帧。
2. **尾部容错** ✅ 已实现：camera ±2 bin 容错匹配 + draft 帧直接接受（draft 前面的对角线准，尾部误差容忍）。

## 已知限制与未来优化方向

### 1. HIT 跳帧不彻底（已诊断，未修复）

HIT 时 `state_0_len += pixnum` 计数跳帧，但 `imagenum_main += 1` 让 Main 仍解码被接受的帧（frame_offset 指向被跳帧）。正确应为 `imagenum_main += 2`，但该改动触发 CUDA assert（KV cache 位置冲突），需要进一步梳理投机状态机的语义后才能安全修复。

### 2. 要超越 baseline 的路径

1. **修复 HIT 跳帧**：`imagenum_main += 2`（需解决 KV cache 语义）
2. **HIT 率 > 85%**：更好的 action predictor
3. **draft < 0.02s**：token-level draft（省 VAE decode + depth）

## 文档

`docs/optimization/` 9 篇技术文档，完整记录优化历程。
