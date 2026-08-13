# 最终总结：投机解码优化全历程（2026-08-13 02:00 → 16:00）

## 核心成果

从"投机逻辑从未触发"到"**接近 baseline 速度**"，共修复 10+ bug，实现 6 项优化。

## 速度演进（第二 demo FPS，GPU 7 RTX 3090）

| 阶段 | FPS | 改动 |
|------|-----|------|
| 起点（投机从未触发） | 0.42 | — |
| + 修复 8 个 bug | 0.84 | 让投机真正运行 |
| + 去 SDPBackend.MATH | 1.20 | Flash Attention |
| + 异步流（spec 不重解码） | 1.43 | HIT 直接接受 draft 帧 |
| + merge=False | 1.62 | 去 VAE 图像 merge |
| + 修复 batch=1（spec 空时不 packed） | 2.22 | 消除 40% 浪费 |
| + max-autotune decode | 2.36 | 匹配 baseline 每步速度 |
| + 容错匹配（camera ±2） | 2.38 | HIT 9→10 |
| + 消除 GPU→CPU 同步（CPU 计数） | 2.56 | 消除 8000 次 .item() |
| + pos buffer 复用 | **2.66** | 消除 torch.tensor(list) 9s |

**累计：0.42 → 2.66 fps（+533%）**，baseline 2.79，只差 5%。

## 关键瓶颈定位过程

1. **投机逻辑从未触发** → frame_completed 时序
2. **decode 慢 7.7 倍** → dynamic 编译 vs max-autotune
3. **update 9.9s** → .item() GPU 同步（8000 次）
4. **pos 9.1s** → torch.tensor(list, device='cuda') 的 CUDA malloc + sync
5. **剩余 3.9s** → 主循环 Python 开销（500 次迭代基础成本）

## 架构级结论

### 投机为何最终接近 baseline 而非超越

1. **HIT 60%**（action predictor 准确率）：10/14 帧跳帧，省 ~3.9s Main 解码
2. **draft 开销 0.78s**：14 次 × (draft 0.046s + action 0.003s + prefill 0.007s)
3. **主循环 Python 开销 ~3s**：500 次迭代的基础成本（投机循环比 baseline 多 spec 状态管理）

净效果：跳帧收益 ≈ draft 开销 + Python 开销，打平。

### 要超越 baseline 需要

1. **HIT 率 > 70%**：更好的 action predictor，或更多候选
2. **draft 开销 < 0.02s**：去掉 VAE decode prev（用 token-level draft）
3. **减少 Python 开销**：合并投机循环的 spec 状态管理

## 关键文件

- `diagonal_decoding.py`：投机状态机 + 所有性能优化
- `lvm.py`：decode 编译策略
- `speculative_wrapper.py`：token2image_gpu
- `docs/optimization/`：6 篇技术文档
