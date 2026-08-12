# Speculative Decoding 优化记录

## 基准（GPU 7，RTX 3090）

环境：torch 2.6.0+cu124，mineworld conda 环境，300M_16f 模型，15 帧，top_p=0.8，window_size=2，small_validation（clip_13 + clip_25）。

### 速度基准（第 2 个 demo 为真实速度，第 1 个含 torch.compile 预热）

| 模式 | 第 1 个 demo | 第 2 个 demo | 相对 baseline |
|------|------|------|------|
| baseline 对角线解码（`image_diagd`） | 0.296 fps | **2.81 fps** | 1.0× |
| speculative（含 debug print） | 0.05 fps | 0.79 fps | 0.28× |
| speculative（清理 print 后） | 0.21 fps | **0.84 fps** | 0.30× |

## 瓶颈分析（初步）

1. **draft_func 开销**（每帧 MISS 调用一次）：
   - VAE `token2image`（decode 上一帧）→ 深度估计输入
   - DepthAnything 深度估计
   - AttentionTokenPredictor 前向
   - `merge=True` 时：VAE `token2image`（decode draft）+ VAE `tokenize_images`（re-encode merged）
   - **合计约 4 次重模型前向（VAE×2 + Depth + Draft）**

2. **action_pred_func 开销**（每帧一次）：GRU 前向，较小。

3. **spec prefill**（每帧 MISS 一次）：对 Main 注入 GT action，并 prefill Spec。

## 优化方向

- [ ] draft_func 缓存：上一帧 token → image 的 decode 可以复用（Main 已经生成过上一帧）
- [ ] merge 的 VAE re-encode 可以跳过（直接用 raw draft token，牺牲部分质量换速度）
- [ ] 减少 action 候选数 K
- [ ] spec prefill 与 Main 生成并行
