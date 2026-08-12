# 突破 2：去除 SDPBackend.MATH 提速 58%

## 日期
2026-08-13

## 速度变化

| 配置 | 第二 demo FPS | 提升 |
|------|------|------|
| 投机逻辑首次跑通（dynamic 编译） | 0.42 | 基线 |
| + decode_some_token 编译（dynamic=True） | 0.76 | +81% |
| + 去除 SDPBackend.MATH | **1.20** | +58% |
| baseline 对角线解码 | 2.81 | 目标 |

## 关键发现

1. **SDPBackend.MATH 是速度杀手**：代码用 `sdpa_kernel(SDPBackend.MATH)` 强制数学实现注意力（最慢），torch.compile 在 dynamic=True 下无法优化它。去除后让 SDPA 用默认后端（Flash Attention），提速 58%。

2. **投机逻辑正确性已确认**：HIT 9/15 = 60%，draft/action/spec prefill 全部正常工作。

## 剩余瓶颈（按耗时排序）

1. **packed batch forward**：Main+Spec 合并 batch=6，每步 forward 比 batch=1 慢。
2. **draft_func 的 VAE 操作**：~0.23s/帧（VAE decode prev + depth + draft forward + VAE decode draft + VAE re-encode），12 次 MISS ≈ 2.8s。
3. **spec prefill**：~0.025s/帧。

## 已知 Bug

- HIT 跳帧时 token 计数错误：`code_list length 5037 is not multiple of 336`（少 3 个 token）。
