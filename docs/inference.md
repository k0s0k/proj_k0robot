# 推理与部署

> **项目**: RDL-Dreams — 基于 Cosmos GR00T-Dreams 的商超零售机器人抓取轨迹生成  
> **主方案**: [../README.md](../README.md) | **官方菜谱**: [../../Dev_book.md](../../Dev_book.md)

---

## 1. 推理管线概览

```
推理管线
═══════════════════════════════════════════════════════

  ┌─────────────┐     ┌───────────────┐     ┌───────────────┐
  │ 输入          │     │ Predict 2.5    │     │ 输出           │
  │              │     │ 推理           │     │               │
  │ 📷 初始帧     │────→│               │────→│ 🎬 轨迹视频     │
  │ 📝 JSON(L)   │     │ 单条 / 批量    │     │ (N 条候选)     │
  │ 🏋️ PT 检查点 │     │               │     │               │
  └─────────────┘     └───────────────┘     └───────┬───────┘
                                                     │
                                                     ▼
                                          ┌───────────────┐
                                          │ Reason 2       │
                                          │ Critic 评分     │
                                          │               │
                                          │ 逐视频: 1-5 分  │
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ 筛选           │
                                          │               │
                                          │ Score ≥ 4.0   │
                                          │ → 高质量轨迹   │
                                          └───────────────┘
```

---

## 2. 前置步骤：检查点转换

训练输出的 DCP 格式必须转换为 PyTorch `.pt` 格式才能用于推理：

```bash
# DCP → PT 转换
python scripts/convert_distcp_to_pt.py \
    $CHECKPOINT_DIR/model \
    $CHECKPOINT_DIR

# 检查输出文件
ls -lh $CHECKPOINT_DIR/model_ema_bf16.pt
# 预期: model_ema_bf16.pt (EMA 平滑版本，推理推荐)
#       model_bf16.pt (直接版本)
```

---

## 3. 单视频推理

### 3.1 命令

```bash
# 单条轨迹生成 (Dev_book.md 标准命令)
torchrun --nproc_per_node=8 \
    examples/inference.py \
    -i gr00t_image2world.json \
    -o outputs/trajectories/ \
    --checkpoint-path model_ema_bf16.pt \
    --experiment predict2_video2world_training_2b_groot_gr1_480
```

### 3.2 输入 JSON 格式

参见 [`inference/input_jsonl/example_retail_grasp.json`](../inference/input_jsonl/example_retail_grasp.json)：

```json
{
    "name": "retail_grasp_cereal_top_shelf",
    "inference_type": "image2world",
    "prompt": "The robot arm is performing a task. Use the right hand to pick up cereal box from top level of supermarket shelf to shopping cart.",
    "input_path": "inference/input_images/shelf_top_001.png",
    "num_output_frames": 93,
    "resolution": "432,768",
    "seed": 0,
    "guidance": 7
}
```

### 3.3 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | 场景名称，用于命名输出文件 |
| `inference_type` | string | ✅ | 固定 `"image2world"` |
| `prompt` | string | ✅ | 任务描述 |
| `input_path` | string | ✅ | 初始帧图像路径 (PNG/JPEG) |
| `num_output_frames` | int | ✅ | 生成帧数，推荐 93 (约 3 秒 @ 30fps) |
| `resolution` | string | ✅ | `"高度,宽度"`，如 `"480,960"` |
| `seed` | int | ✅ | 随机种子 (0-4294967295)，不同种子产生不同轨迹 |
| `guidance` | float | 否 | 引导强度，默认 7，越大越贴近 Prompt |
| `input_video_path` | string | 否 | Video2World 模式输入视频路径 |
| `negative_prompt` | string | 否 | 负向 Prompt |

---

## 4. 批量推理与 Best-of-N

### 4.1 JSONL 批量格式

```jsonl
{"name": "scene_001", "inference_type": "image2world", "prompt": "...", "input_path": "frame_001.png", "num_output_frames": 93, "resolution": "480,960", "seed": 0, "guidance": 7}
{"name": "scene_002", "inference_type": "image2world", "prompt": "...", "input_path": "frame_002.png", "num_output_frames": 93, "resolution": "480,960", "seed": 1, "guidance": 7}
```

### 4.2 批量推理命令

```bash
# 批量生成 (--num-generations 控制每条 Prompt 的候选数)
torchrun --nproc_per_node=1 \
    inference.py \
    -i inputs/retail_batch.jsonl \
    -o outputs/trajectories/ \
    --num-generations 5 \
    --checkpoint-path model_ema_bf16.pt \
    --experiment predict2_video2world_training_2b_groot_gr1_480

# 输出结构:
# outputs/trajectories/
# ├── scene_001_seed_0.mp4
# ├── scene_001_seed_1.mp4
# ├── scene_001_seed_2.mp4
# ├── scene_001_seed_3.mp4
# ├── scene_001_seed_4.mp4
# └── ...
```

### 4.3 Best-of-N 参数调优

| N 值 | 候选数 | 理论最优选中率 | GPU 时间 | 建议场景 |
|------|--------|---------------|---------|---------|
| 3 | 3 | ~87% | 3× 单次 | 快速迭代 |
| **5** | 5 | **~95%** | 5× 单次 | **默认推荐** ✅ |
| 10 | 10 | ~99% | 10× 单次 | 高精度场景 |

---

## 5. Critic 评分

### 5.1 零样本评分

```bash
# Reason 2 对生成视频进行物理合理性评分
uv run examples/gr00t-dreams/inference_videophy2.py \
    --video-dir ./outputs/trajectories \
    --output-dir outputs/critic_scores
```

### 5.2 SFT 后评分

```bash
# 使用微调后的 Critic 评分
uv run examples/gr00t-dreams/inference_videophy2.py \
    --video-dir ./outputs/trajectories \
    --output-dir outputs/critic_scores \
    --model outputs/videophy2_sft_cosmos_rl
```

### 5.3 评分输出格式

```json
{
    "video_id": "scene_001_seed_0",
    "score": 4,
    "video_path": "outputs/trajectories/scene_001_seed_0.mp4",
    "timestamp": "2026-01-15T10:30:00Z"
}
```

---

## 6. 高质量轨迹筛选

### 6.1 自动筛选脚本

```bash
# 使用 filter_retail_grasp.sh 筛选 Score ≥ 4.0 的视频
bash scripts/filter_retail_grasp.sh \
    outputs/critic_scores \
    outputs/trajectories \
    4.0

# 输出: outputs/trajectories/filtered/
#   scene_001_seed_0.mp4  (score: 4)
#   scene_001_seed_3.mp4  (score: 5)
```

### 6.2 筛选逻辑

```
for each video in outputs/trajectories/:
    score = critic_scores[video_name]
    if score >= threshold (default 4.0):
        cp video → outputs/trajectories/filtered/

筛选结果统计:
  - 总候选数: N × M  (N=场景数, M=每场景生成数)
  - 通过数: P
  - 通过率: P / (N × M) × 100%
```

---

## 7. Linux 服务器部署

### 7.1 目录结构

```
/opt/rdl-dreams/
├── cosmos-predict2.5/          # Predict 2.5 仓库
├── cosmos-reason2/             # Reason 2 仓库
├── models/
│   ├── predict_checkpoints/    # PT 检查点
│   └── critic/                 # Critic 模型
├── data/
│   └── prompts/                # 推理输入 JSONL
├── outputs/
│   ├── trajectories/           # 生成轨迹
│   ├── critic_scores/          # 评分结果
│   └── filtered/               # 筛选后高质量轨迹
└── logs/                       # 运行日志
```

### 7.2 环境变量

```bash
# 添加到 ~/.bashrc 或启动脚本
export IMAGINAIRE_OUTPUT_ROOT=/opt/rdl-dreams/models/predict_checkpoints
export HF_HOME=/opt/rdl-dreams/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
```

### 7.3 一键推理脚本

```bash
#!/bin/bash
# run_inference.sh — RDL-Dreams 推理一键脚本

set -e

INPUT_JSONL=${1:-"inputs/retail_batch.jsonl"}
OUTPUT_DIR=${2:-"outputs/trajectories"}
CHECKPOINT=${3:-"models/predict_checkpoints/model_ema_bf16.pt"}
NUM_GEN=${4:-5}

echo "=== 阶段 1: 轨迹生成 ==="
torchrun --nproc_per_node=1 \
    inference.py \
    -i "$INPUT_JSONL" \
    -o "$OUTPUT_DIR" \
    --num-generations "$NUM_GEN" \
    --checkpoint-path "$CHECKPOINT" \
    --experiment predict2_video2world_training_2b_groot_gr1_480

echo "=== 阶段 2: Critic 评分 ==="
uv run examples/gr00t-dreams/inference_videophy2.py \
    --video-dir "$OUTPUT_DIR" \
    --output-dir outputs/critic_scores

echo "=== 阶段 3: 筛选高质量轨迹 ==="
bash scripts/filter_retail_grasp.sh \
    outputs/critic_scores \
    "$OUTPUT_DIR" \
    4.0

echo "=== 完成 ==="
echo "高质量轨迹: outputs/trajectories/filtered/"
```

### 7.4 显存规划

| 操作 | GPU 数量 | 每卡显存 | 总显存 |
|------|---------|---------|--------|
| 单条推理 (2B) | 1 | ~40 GB | 40 GB |
| 单条推理 (14B) | 1 | ~80 GB | 80 GB |
| 批量推理 (2B, 8 GPU) | 8 | ~40 GB | 320 GB |
| Critic 评分 (8B) | 1 | ~60 GB | 60 GB |

---

## 8. 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `model_ema_bf16.pt not found` | 未运行 DCP→PT 转换 | 执行 `convert_distcp_to_pt.py` |
| `CUDA out of memory` | 显存不足 | 降分辨率、减帧数、用更小模型 |
| `--experiment not found` | 配置名拼写错误 | 确认 config.py 中的 experiment 名称 |
| JSONL 解析失败 | JSON 格式错误 | 逐行检查 JSON 有效性 |
| Critic 返回异常分 | Prompt 模板不匹配 | 使用 `video_reward.yaml` 模板 |
