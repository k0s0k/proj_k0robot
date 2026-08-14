# 训练方案

> **项目**: RDL-Dreams — 基于 Cosmos GR00T-Dreams 的商超零售机器人抓取轨迹生成  
> **主方案**: [../README.md](../README.md) | **官方菜谱**: [../../Dev_book.md](../../Dev_book.md)

---

## 1. 训练概览

RDL-Dreams 涉及两类训练任务：

| 训练任务 | 模型 | 方式 | 数据 | 是否必选 |
|---------|------|------|------|---------|
| **Predict 2.5 后训练** | Video2World | Full / LoRA | GR1 + 零售合成 | ✅ 核心 |
| **Reason 2 Critic SFT** | 8B VLM | SFT / RL | VideoPhy-2 + 零售标注 | 可选 |

---

## 2. Predict 2.5 后训练

### 2.1 训练命令

```bash
# Predict 2.5 Post-Training (Dev_book.md 标准命令)
torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2_video2world_training_2b_groot_gr1_480

# 多卡训练
torchrun --nproc_per_node=8 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2_video2world_training_2b_groot_gr1_480
```

### 2.2 关键配置项

```python
# config.py — predict2_video2world_training_2b_groot_gr1_480

experiment = dict(
    # === 视频配置 ===
    video2world=dict(
        resolution=[480, 960],     # 训练分辨率 (H,W)
        num_input_frames=1,        # Video2World 输入帧
        num_output_frames=93,      # 生成帧数
    ),

    # === 模型配置 ===
    model=dict(
        predict2_2b_initialized=False,  # 使用官方预训练权重
        checkpoint_path="nvidia/Cosmos-Predict2.5-Video2World-2B",
    ),

    # === 数据配置 ===
    dataset=dict(
        batch_size=1,              # 单卡 batch size
        workers=4,                 # DataLoader workers
    ),
)
```

### 2.3 训练策略对比

| 策略 | 数据量需求 | 训练时间 | 显存需求 | 效果 | 适用场景 |
|------|----------|---------|---------|------|---------|
| **Full Post-Training** | ≥ 100 条 | 数小时~数天 | 60+ GB/卡 | 最优 | 数据充分时首选 |
| **LoRA** | 20-50 条 | 1-2 小时 | 30-40 GB/卡 | 良好 | 数据有限时 |

### 2.4 零售数据训练策略

```
零售场景推荐策略
═══════════════════════════════════

数据量 ≥ 100 条 → Full Post-Training
                  torchrun --nproc_per_node=8 ...
                  experiment=predict2_video2world_training_2b_retail_grasp_480

数据量 20-50 条 → LoRA 微调
                  rank=64, alpha=128
                  只训练 attention 层权重

混合策略 (推荐):
  Step 1: Full PT 在 GR1-100 上 (100 条)
  Step 2: LoRA 微调在零售数据上 (20-50 条)
  → 兼顾通用操作先验 + 零售领域特化
```

### 2.5 检查点格式

| 格式 | 用途 | 说明 |
|------|------|------|
| **DCP** (Distributed Checkpoint) | 训练保存 | torch.distributed.checkpoint 格式 |
| **PT** (PyTorch) | 推理使用 | 需通过 `convert_distcp_to_pt.py` 转换 |

DCP → PT 转换命令：

```bash
python scripts/convert_distcp_to_pt.py \
    $CHECKPOINT_DIR/model \
    $CHECKPOINT_DIR

# 输出: $CHECKPOINT_DIR/model_ema_bf16.pt
```

### 2.6 超参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **学习率** | 1e-5 (Full) / 5e-4 (LoRA) | LoRA 用更大学习率 |
| **Batch Size** | 1 / GPU | 视频模型显存限制 |
| **梯度累积** | 4-8 | 等效增大有效 batch |
| **训练步数** | 500-2000 | 根据数据量调整 |
| **精度** | BF16 | 减少显存，加速训练 |
| **优化器** | AdamW | weight_decay=0.01 |
| **LR Schedule** | Cosine | warmup=3% |

### 2.7 环境变量

```bash
export IMAGINAIRE_OUTPUT_ROOT=/path/to/checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/path/to/huggingface_cache
```

---

## 3. Reason 2 Critic SFT

### 3.1 为什么做 SFT

- 零样本 Reason 2 在零售场景的物理评估可能不够精确
- SFT 使模型理解零售领域的特定物理约束（商品变形、抓取稳定性等）
- 训练数据：VideoPhy-2（3.4k 视频 + 人工物理评分 1-5）

### 3.2 VideoPhy-2 数据集

| 属性 | 说明 |
|------|------|
| **训练集** | 3,400 条视频 |
| **评估集** | 3,300 条视频 |
| **标注** | 人工标注物理合理性 1-5 分 |
| **来源** | HuggingFace: `videophysics/videophy2_train` `videophysics/videophy2_test` |

### 3.3 SFT 训练命令

```bash
# Step 1: 零样本基线评估
uv run examples/video_critic/inference_videophy2.py \
    --model nvidia/Cosmos-Reason2-8B \
    --output-dir results/videophy2_zero_shot \
    --dataset videophysics/videophy2_test \
    --split test \
    --input-file prompts/video_reward.yaml

# Step 2: SFT 训练
cd examples/cosmos_rl
uv run cosmos-rl \
    --config ../video_critic/configs/videophy2_sft.toml \
    scripts/hf_sft.py

# Step 3: SFT 后评估
uv run examples/video_critic/inference_videophy2.py \
    --model outputs/videophy2_sft_cosmos_rl \
    --output-dir results/videophy2_sft \
    --dataset videophysics/videophy2_test \
    --split test
```

### 3.4 SFT 训练配置

参见 [`configs/critic/videophy2_sft.toml`](../configs/critic/videophy2_sft.toml)：

| 参数 | 值 | 说明 |
|------|-----|------|
| **学习率** | 1e-6 | 低学习率避免灾难性遗忘 |
| **Epochs** | 10 | |
| **Batch Size** | 32 (全局) | dp_shard_size=8, mini_batch=4 |
| **模型** | nvidia/Cosmos-Reason2-8B | |
| **Max Length** | 4096 | |
| **输出** | outputs/videophy2_sft_cosmos_rl | |

### 3.5 RL 训练 (高级)

当高质量标注数据稀缺时，RL 训练提供更灵活的替代方案：

```bash
uv run cosmos-rl \
    --config configs/videophy2_rl.toml \
    scripts/hf_rl.py
```

RL 通过奖励信号引导模型，无需精确的人工评分，更适合主观性强的物理合理性判断。

---

## 4. 训练管线完整流程

```
Step 1  ───→ Step 2  ───→ Step 3  ───→ Step 4 (可选) ───→ Step 5 (可选)
数据准备     Predict       Critic       Critic SFT          Critic RL
             Post-Train    零样本评估
```

### 4.1 执行顺序

```bash
# === Phase 1: 数据 ===
hf download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/
python -m scripts.create_prompts_for_gr1_dataset --dataset_path datasets/benchmark_train/gr1

# === Phase 2: Predict 2.5 后训练 ===
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2_video2world_training_2b_groot_gr1_480

# === Phase 3: DCP → PT 转换 ===
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

# === Phase 4 (可选): Critic SFT ===
uv run cosmos-rl --config configs/critic/videophy2_sft.toml scripts/hf_sft.py
```

---

## 5. 训练监控

### 5.1 W&B 日志

```bash
# 启用 W&B
export WANDB_API_KEY="your_key"
export WANDB_PROJECT="rdl-dreams"

# 禁用 W&B (离线训练)
export WANDB_MODE=disabled
```

### 5.2 关键监控指标

| 指标 | 说明 | 正常趋势 |
|------|------|---------|
| **Training Loss** | 扩散去噪损失 | 持续下降 |
| **Validation Loss** | 验证集损失 | 先降后平 |
| **GPU Memory** | 显存使用率 | 稳定在 80-95% |
| **Frames/sec** | 训练吞吐量 | 稳定 |
| **Checkpoint Size** | DCP 文件大小 | 稳定 |

---

## 6. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| OOM (Out of Memory) | batch_size 过大 | 设 batch_size=1, 增大梯度累积 |
| CUDA 版本不匹配 | Python/CUDA 不对齐 | 使用 uv sync --extra=cu128 |
| DCP 加载失败 | 版本不兼容 | 确认 cosmos-predict2.5 版本一致 |
| 训练 Loss 不下降 | 学习率过大/过小 | 从 1e-5 开始调整 |
| 多卡训练卡死 | NCCL 通信问题 | 检查 `NCCL_DEBUG=INFO` 输出 |
