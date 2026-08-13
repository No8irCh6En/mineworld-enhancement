# 完整优化总结（2026-08-13 02:00 → 11:15）

## 最终结果

| 指标 | baseline | 投机（最终） |
|------|---------|-------------|
| FPS（第2 demo，GPU 7 RTX 3090） | 2.79 | **2.66** |
| 视频生成 | ✅ 15帧 | ✅ 15帧（验证帧数一致） |
| HIT 率 | — | 67%（10/15） |
| 无报错 | ✅ | ✅ |

**投机从 0.42 fps（无法运行）优化到 2.66 fps（baseline 的 95%），累计 +533%。**

## 速度演进时间线

| 阶段 | FPS | 关键改动 |
|------|-----|---------|
| 起点 | 0.42 | 投机逻辑从未触发 |
| +修复 8 bug | 0.84 | 让投机真正运行 |
| +去 MATH | 1.20 | Flash Attention |
| +异步流 | 1.43 | spec 不重解码 |
| +merge=False | 1.62 | 去 VAE merge |
| +batch 修复 | 2.22 | spec 空时 batch=1 |
| +max-autotune | 2.36 | 匹配 baseline 步速 |
| +容错匹配 | 2.38 | camera ±2 bin |
| +CPU 计数 | 2.56 | 消除 8000 次 .item() |
| +pos buffer | **2.66** | 消除 torch.tensor(list) 9s |

## 10 个关键 Bug 修复

1. 投机逻辑从未触发（frame_completed 时序）
2. draft_frame_done 死循环（state_1_len 差 1）
3. draft_func 拿 335 token（prefill token 缺失）
4. VAE token2image 不支持 batch（改 token2image_gpu）
5. torch.compile 卡死（action_pred_func 不编译）
6. SDPBackend.MATH 慢 58%
7. token 计数错误（5037 非 336 倍数）
8. HIT 路径 token 缺失
9. .item() GPU 同步 8000 次（改 CPU 计数）
10. torch.tensor(list, cuda) 9s（改 buffer 复用）

## 性能瓶颈定位全过程

1. decode 慢 7.7 倍 → dynamic vs max-autotune 编译
2. update 9.9s → .item() GPU 同步
3. pos 9.1s → torch.tensor CUDA malloc + sync
4. 剩余 5% → 投机循环固有 Python 开销（每步 10.5ms vs baseline 6.4ms）

## 架构级结论

### 投机未超越 baseline 的原因

1. **HIT 67%**：10/15 帧跳帧，省 ~3.9s Main 解码
2. **draft 0.045s/帧**（VAE 0.032 + depth 0.011 + forward 0.002）× 15 = 0.68s
3. **主循环 Python 开销**：每步 4ms 额外（spec 状态管理）× 500 步 = 2s

净效果：跳帧收益 ≈ draft + Python 开销，打平。

### 投机理论上限

如果 HIT 100% 且 draft 开销 <0.02s，投机理论上能到 4+ fps（Main 只解码 1 帧）。当前受限：
- HIT 率：action predictor 准确率（模型质量，非代码问题）
- draft 开销：VAE decode（深度图输入必需）

## 技术文档

`docs/optimization/` 下 7 篇：
- 00_baseline.md
- 01_first_full_pipeline.md
- 02_remove_math_backend.md
- 03_root_cause_no_speedup.md
- 04_final_summary.md（早期版）
- 05_async_stream.md
- 06_fix_batch.md
- 07_final_16pm.md
