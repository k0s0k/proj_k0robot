#  基于 GR00T-Dreams 的商超零售场景机器人抓取轨迹生成项目

> **路线**: Cosmos Predict 2.5 → 后训练 → 轨迹生成 → Cosmos Reason 2 Critic 拒绝采样  
> **参考菜谱**: [Leveraging World Foundation Models for Synthetic Trajectory Generation in Robot Learning](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/end2end/gr00t-dreams/post-training.html)  
> **原始来源**: NVIDIA Cosmos Cookbook — gr00t-dreams/post-training  
> **目标场景**: 商超零售货架商品抓取（机械臂从货架取商品 → 放入购物篮/购物车）

---

## 📋 目录

- [1. 概述](#1-概述)
- [2. 前提条件](#2-前提条件)
  - [2.1 硬件要求](#21-硬件要求)
  - [2.2 软件环境初始化](#22-软件环境初始化)
  - [2.3 HuggingFace 认证与权限](#23-huggingface-模型数据集访问权限)
  - [2.4 环境变量配置](#24-环境变量配置)
  - [2.5 W&B 日志配置](#25-weights--biases-日志配置)
- [3. 数据准备](#3-数据准备)
  - [3.1 下载 GR1 基线数据](#31-下载-gr1-训练数据集作为基线)
  - [3.2 零售抓取数据生成](#32--生成零售抓取数据关键适配步骤)
  - [3.3 Prompt 预处理](#33-预处理为每个视频生成-prompt-文本)
  - [3.4 零售 Prompt 格式](#34--零售场景-prompt-格式参考)
- [4. 后训练 Cosmos Predict 2.5](#4-后训练-cosmos-predict-25)
- [5. 推理：轨迹生成](#5-推理轨迹生成)
- [6. 策略评估：批量生成](#6-策略评估批量轨迹生成与-critic-筛选)
- [7. Cosmos Reason 2 Critic 拒绝采样](#7-cosmos-reason-2-视频-critic-拒绝采样)
- [8. Critic 微调（可选）](#8-cosmos-reason-2-critic-微调可选但推荐)
- [9. 计算评估指标](#9-计算评估指标)
- [10. 局限性与注意事项](#10-局限性与注意事项)
- [11. 完整执行清单](#11-完整执行清单)
- [12. 引用](#12-引用)

---

## 1. 概述

本指南基于 NVIDIA 官方 GR00T-Dreams 菜谱，将 Cosmos Predict 2.5 在 **PhysicalAI-Robotics-GR00T-GR1** 开放数据集上进行后训练，生成合成机器人轨迹，**并适配到商超零售商品抓取场景**。

### 管线架构

```
──────────────────     ──────────────────     ──────────────────
  Cosmos Predict         Cosmos Reason 2        输出：高质量     
  2.5 Video2World  ->     Video Critic     ->    零售抓取轨迹    
  (轨迹生成)              (物理合理性评分)        (Score ≥ 4.0)  
──────────────────     ──────────────────     ──────────────────
```

### 涉及模型

| 模型 | 工作负载 | 用途 |
|------|---------|------|
| Cosmos Predict 2.5 (2B | 14B) | 后训练 + 推理 | 合成轨迹生成（Video2World） |
| Cosmos Reason 2 | 推理 | 零样本视频 Critic，评估物理合理性 |

---

## 2. 前提条件

### 2.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU 架构 | NVIDIA Ampere 及以上 (A100, H100, RTX Pro 6000) | H100 (80GB) × 8 |
| CUDA | 12.8+ (cu128) 或 12.9 (cu129) | 12.8+ |
| 系统内存 | 64 GB | 256 GB |
| 磁盘空间 | 200 GB | 500 GB+ (存放训练数据、检查点、HF 缓存) |
| 操作系统 | Ubuntu 22.04+ (glibc ≥ 2.35) | Ubuntu 24.04 |
| Python | 3.10 | 3.10 |

> ⚠️ **关键限制**: 不支持旧架构 GPU (Volta/Turing)，必须 Ampere 及以上。Cosmos Predict 2.5 和 Reason 2 均通过 `uv` 和 `uv sync` 管理依赖，CUDA 版本通过 `--extra=cu128` 或 `--extra=cu129` 指定。

### 2.2 软件环境初始化

> ⚠️ **重要**: Cosmos Predict 2.5 和 Reason 2 都通过 **`uv`** 管理依赖（官方推荐），不要使用 `pip install`。区别在于：**Predict 2.5 直接 `uv sync`**（自动解析 CUDA 依赖），**Reason 2 需用 `--extra=cu128`**（或 `--extra=cu129`）显式指定 CUDA 变体。

```bash
# ============================================
# 0. 系统依赖
# ============================================
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    ffmpeg

# ============================================
# 1. 安装 uv 包管理器
# ============================================
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# ============================================
# 2. 克隆仓库
# ============================================
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git

# ============================================
# 3. 安装 Cosmos Predict 2.5（使用 uv）
# ============================================
cd cosmos-predict2.5

# 安装到新的虚拟环境（官方推荐）
uv sync
source .venv/bin/activate

# 或者：安装到活动环境（如 conda）
conda create -n RLD_cosmos python=3.13 -y
conda activate RDL_cosmos
uv sync --active # uv sync --active --inexact，用--inexact同步保留 conda 环境中已有的无关包，有时候会有猫病

# 以后再次同步依赖建议直接运行：
cd 到 /cosmos-predict2.5
/root/.local/bin/uv sync --frozen

# 可选：安装全部可选依赖组
# uv sync --all-groups

# ============================================
# 4. 安装 Cosmos Reason 2（使用 uv）
# ============================================
cd ../cosmos-reason2
uv sync --extra=cu128
source .venv/bin/activate

# 如果使用 CUDA 12.9，替换为:
# uv sync --extra=cu129

# ============================================
# 5. HuggingFace CLI 安装与认证
# ============================================
# 安装 HuggingFace CLI
uv tool install -U "huggingface_hub[cli]"

# 登录（交互式，输入 Access Token）
hf auth login

# 验证 CUDA 可用
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2.3 HuggingFace 模型/数据集访问权限

在开始之前，请确保已完成以下步骤：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | [获取 HuggingFace Access Token](https://huggingface.co/settings/tokens) | 权限设为 `Read` |
| 2 | `hf auth login` | 输入 Token 完成认证 |
| 3 | 接受 [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) | 在模型页面点击 "Agree and access repository" |
| 4 | 接受各数据集的使用条款 | GR1-100、EVAL-175 同样需要点击同意 |

| 资源 | 类型 | 地址 |
|------|------|------|
| Cosmos-Predict2.5-2B | 模型 | `nvidia/Cosmos-Predict2.5-2B-Video2World` |
| GR1-100 | 数据集 | `nvidia/GR1-100` |
| EVAL-175 | 数据集 | `nvidia/EVAL-175` |
| Cosmos-Reason2-8B | 模型 | `nvidia/Cosmos-Reason2-8B` |

> ⚠️ **检查点自动下载**: 训练和推理时检查点会自动从 HuggingFace 下载。如需更改缓存位置，设置环境变量 `export HF_HOME=/path/to/cache`。确保 `HF_HOME` 有足够磁盘空间。

### 2.4 环境变量配置

```bash
# ============================================
# 训练输出目录（检查点、日志、工件）
# 默认: /tmp/imaginaire4-output
# ============================================
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory

# ============================================
# HuggingFace 缓存目录（模型和数据集下载位置）
# 默认: ~/.cache/huggingface
# ============================================
export HF_HOME=/path/to/huggingface/cache

# 确保磁盘空间充足
df -h $IMAGINAIRE_OUTPUT_ROOT
df -h $HF_HOME
```

### 2.5 Weights & Biases 日志配置

训练默认会尝试记录指标到 W&B，你有以下选择：

**启用 W&B**:

```bash
# 1. 在 https://wandb.ai 创建免费账户
# 2. 在 https://wandb.ai/authorize 获取 API Key
# 3. 设置环境变量
export WANDB_API_KEY=your_api_key_here
```

**禁用 W&B** (在训练命令中添加):

```bash
# 在训练命令末尾添加:
job.wandb_mode=disabled
```

> ⚠️ **安全警告**: 不要将 API Key 硬编码到脚本或提交到 git，始终通过环境变量管理。

---

## 3. 数据准备

> ⚠️ **零售场景适配要点**: 此步骤是最大的改造点。原始管线使用 GR1（桌面物体操作）数据集。  
> 对于零售抓取场景，你需要准备或生成**商超货架商品抓取**的视频数据集。  
> 推荐路线：**仿真器批量生成 → Cosmos Transfer 2.5 Sim2Real 增强 → Cosmos Curator 策展**。

### 3.1 下载 GR1 训练数据集（作为基线）

```bash
cd cosmos-predict2.5

# 下载 GR1-100 数据集
huggingface-cli download nvidia/GR1-100 \
    --repo-type dataset \
    --local-dir datasets/benchmark_train/hf_gr1/

# 整理目录结构
mkdir -p datasets/benchmark_train/gr1/videos
mv datasets/benchmark_train/hf_gr1/gr1/*.mp4 datasets/benchmark_train/gr1/videos/
mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/
```

### 3.2  生成零售抓取数据（关键适配步骤）

> **说明**: GR1 数据涵盖桌面物体操作（如"从桌子上拿起香蕉放入篮子"），与商超零售场景（货架→购物篮）的动作模式类似但场景不同。你需要额外准备零售场景视频来微调模型。

#### 方案 A: 仿真器生成 + Cosmos Transfer 2.5 增强（推荐）

```bash
# ============================================
# Step A1: 使用 Isaac Sim / MuJoCo 生成合成抓取视频
# ============================================
# (此处为示意，具体取决于你使用的仿真器)
# 生成内容：
#   - 商超货架场景（多层货架、不同商品排列）
#   - 机械臂抓取动作（reach → grasp → lift → place）
#   - 多视角录制（正视图 + 顶视图 + 机械臂第一人称视角）
# 输出目录: ./retail_sim_videos/

# ============================================
# Step A2: 用 Cosmos Transfer 2.5 做 Sim2Real 增强
# ============================================
# 参考 cosmos-cookbook Gallery: Robotics Domain Adaptation (Scene 1b)
# 用 edge control 保持机器人动作不变，转换为照片级商超场景

# 伪命令示例（具体参数见 Transfer 2.5 文档）:
# python cosmos_transfer2/inference.py \
#   --input_video ./retail_sim_videos/grasp_001.mp4 \
#   --control edge \
#   --prompt "A photorealistic supermarket aisle with white shelves, a robotic arm grasping a red bottle from the middle shelf, bright fluorescent lighting, clean floor" \
#   --guidance 7 \
#   --output ./retail_realistic_videos/
```

#### 方案 B: 直接录制真实商超抓取视频

```bash
# 在商超环境中用摄像头+机械臂录制真实抓取操作
# 录制要求：
#   - 分辨率 ≥ 480p
#   - 帧率 ≥ 24fps
#   - 每个视频长度 3-10 秒
#   - 包含完整抓取动作（reach → grasp → lift → place）
# 输出目录: ./retail_real_videos/
```

#### 数据整合

```bash
# ============================================
# Step A3: 合并 GR1 基线数据 + 零售场景数据
# ============================================
# 将零售抓取视频合并到训练数据集目录

# 复制零售视频到训练集
cp ./retail_realistic_videos/*.mp4 datasets/benchmark_train/gr1/videos/

# ============================================
# Step A4 (可选): 用 Cosmos Curator 策展零售视频
# ============================================
# 参考 cosmos-cookbook 第 10/11 章
#   - 分割 (Shot-Aware Splitting)
#   - 转码 (GPU Transcoding): 统一 480p, 24fps
#   - 字幕 (Captioning): 用 VLM 标注商品类别、动作描述
#   - 过滤 (VLM Filter): 去除低质量/模糊片段
#   - 去重 (Deduplication)
```

### 3.3 预处理：为每个视频生成 Prompt 文本

```bash
# 为每个训练视频生成对应的 txt prompt 文件
# 此脚本会读取 metadata.csv 中的标注信息，为每个视频生成 prompt
python -m scripts.create_prompts_for_gr1_dataset \
    --dataset_path datasets/benchmark_train/gr1
```

**预处理后的目录结构**:

```
datasets/benchmark_train/gr1/
├── metas/
│   ├── <video_name>.txt      # 每个视频的 Prompt 文本
├── videos/
│   ├── <video_name>.mp4       # 原始 GR1 视频 + 零售抓取视频
├── metadata.csv                # 元数据标注
```

### 3.4  零售场景 Prompt 格式参考

原始 GR1 Prompt 格式：

```
The robot arm is performing a task. Use the right hand to pick up
banana from teal plate to wooden table.
```

**零售场景 Prompt 格式**（按此格式标注你的零售抓取视频）：

```
The robot arm is performing a task. Use the right hand to pick up
<cereal box> from <top level of supermarket shelf> to <shopping cart>.

The robot arm is performing a task. Use the right hand to pick up
<bottled water> from <middle shelf of convenience store aisle> to <plastic basket>.
```

---

## 4. 后训练 Cosmos Predict 2.5

> 本节所有命令在 `cosmos-predict2.5` 仓库根目录下运行。

### 4.1 启动后训练

```bash
cd cosmos-predict2.5

# 后训练 Predict 2.5 2B Video2World 模型
# 使用 GR1 数据（已包含零售抓取数据）
torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2_video2world_training_2b_groot_gr1_480
```

> **说明**: 如果使用 14B 模型，将配置名中的 `2b` 替换为 `14b`，训练流程完全相同，仅 GPU 显存需求更大。

### 4.2 训练配置解析

使用的关键配置（位于 `cosmos_predict2/_src/predict2/configs/video2world/config.py`）:

```python
predict2_video2world_training_2b_groot_gr1_480 = dict(
    ...,
    dataloader_train=dataloader_train_gr1,
    ...,
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_groot_gr1_480",
    ),
    ...,
)
```

### 4.3 检查点保存位置

```
${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/
└── cosmos_predict_v2p5/          # PROJECT
    └── video2world/               # GROUP
        └── 2b_groot_gr1_480/      # NAME
            └── checkpoints/
                ├── iter_<NUMBER>/
                │   ├── model/
                │   │   ├── .metadata
                │   │   └── __0_0.distcp
                │   ├── optim/
                │   ├── scheduler/
                │   └── trainer/
                └── latest_checkpoint.txt
```

### 4.4  零售适配的微调策略选择

| 场景 | 策略 | 命令修改 |
|------|------|---------|
| 零售数据 < 1000 条 | LoRA 微调 | 添加 LoRA 配置到 config.py |
| 零售数据 1000-10000 条 | Full Post-Training (2B) | 使用默认命令 |
| 零售数据 > 10000 条 | Full Post-Training (14B) | 切换到 14B 配置 |

> 原始 GR1-100 含约 100 条视频。加入零售数据后，按总数据量选择策略。

### 4.5 禁用 W&B 日志（可选）

```bash
# 如果不需要 Weights & Biases 日志记录
torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2_video2world_training_2b_groot_gr1_480 \
    job.wandb_mode=disabled
```

---

## 5. 推理：轨迹生成

> 本节所有命令在 `cosmos-predict2.5` 仓库根目录下运行。

### 5.1 DCP 检查点转为 PyTorch 格式

训练保存的检查点是分布式检查点（DCP）格式，推理前需转换：

```bash
# 设置检查点路径
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# 转换 DCP → PyTorch 格式
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

**转换后生成的文件**:

| 文件 | 说明 |
|------|------|
| `model.pt` | 完整检查点（含常规权重 + EMA 权重） |
| `model_ema_fp32.pt` | EMA 权重 (float32) |
| `model_ema_bf16.pt` | EMA 权重 (bfloat16)，**推理推荐用此文件** ✅ |

### 5.2 单视频轨迹生成

```bash
# 使用商店抓取 Prompt 生成单条轨迹
torchrun --nproc_per_node=1 examples/inference.py \
    -i assets/sample_gr00t_dreams_gr1/gr00t_image2world.json \
    -o outputs/retail_grasp_sample \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment predict2_video2world_training_2b_groot_gr1_480
```

### 5.3  零售抓取推理输入示例

创建自定义推理输入 JSON（`retail_grasp_input.json`）:

```json
{
    "name": "retail_grasp_cereal_box",
    "inference_type": "image2world",
    "prompt": "The robot arm is performing a task. Use the right hand to pick up cereal box from top shelf of supermarket aisle to shopping cart.",
    "input_path": "images/retail_shelf_001.png",
    "num_output_frames": 93,
    "resolution": "432,768",
    "seed": 0,
    "guidance": 7
}
```

```bash
# 使用自定义输入
torchrun --nproc_per_node=1 examples/inference.py \
    -i retail_grasp_input.json \
    -o outputs/retail_grasp_single \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment predict2_video2world_training_2b_groot_gr1_480
```

---

## 6. 策略评估：批量轨迹生成与 Critic 筛选

> 本节前半部分在 `cosmos-predict2.5` 目录下，后半部分（Reason 2 Critic）在 `cosmos-reason2` 目录下。

### 6.1 下载评估数据集

```bash
cd cosmos-predict2.5

# 下载 EVAL-175 评估数据集
huggingface-cli download nvidia/EVAL-175 \
    --repo-type dataset \
    --local-dir dream_gen_benchmark
```

### 6.2 准备批量推理输入

```bash
# Step 1: 生成批量输入 JSON
python -m scripts.prepare_batch_input_json \
    --dataset_path dream_gen_benchmark/gr1_object/ \
    --save_path output/dream_gen_benchmark/cosmos_predict2_14b_gr1_object/ \
    --output_path dream_gen_benchmark/gr1_object/batch_input.json

# Step 2: 从 cosmos-cookbook 复制辅助脚本到 cosmos-predict2.5
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/gr1_batch_to_jsonl.py scripts/

# Step 3: 将 batch_input.json 转为 JSONL 格式
python scripts/gr1_batch_to_jsonl.py
```

**生成的 JSONL 格式** (每行一条推理请求):

```json
{"inference_type": "image2world", "name": "000_11_grasp_pepper_shelf", "prompt": "Use the right hand to pick up green pepper from black shelf to inside brown paper bag.", "input_path": "11_grasp_pepper.png", "num_output_frames": 93, "resolution": "432,768", "seed": 0, "guidance": 7}
```

### 6.3 批量多视频展开生成 (Multiple Rollouts)

```bash
# 复制推理脚本
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/inference.py .
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/config.py .

# 批量生成：每个 Prompt 生成 5 个候选轨迹
torchrun --nproc_per_node=1 inference.py \
    -i dream_gen_benchmark/gr1_object/gr1_batch.jsonl \
    -o outputs/retail_grasp_rollouts \
    --num-generations 5 \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment predict2_video2world_training_2b_groot_gr1_480
```

> `--num-generations 5` 表示每个输入生成 5 个候选视频，后续用 Reason 2 做 Best-of-5 采样。

---

## 7. Cosmos Reason 2 视频 Critic 拒绝采样

> 本节所有命令在 `cosmos-reason2` 仓库根目录下运行。

### 7.1 评估标准

每个生成的视频按物理合理性 1-5 分评分：

| 分数 | 描述 | 物理合理性 | 处理策略 |
|------|------|-----------|---------|
| 1 | 完全违反物理定律 | 完全不可信 | ❌ 丢弃 |
| 2 | 大部分违反物理定律 | 多数不真实 | ❌ 丢弃 |
| 3 | 物理合理性中等 | 真实/不真实混杂 | ⚠️ 人工复核 |
| 4 | 物理合理性良好 | 大部分真实 | ✅ 保留 |
| 5 | 完美遵循物理定律 | 完全可信 | ✅ 保留 |

> **核心原则**: 仅保留分数 **≥ 4.0** 的视频，丢弃 1.0 和 2.0 的视频。

### 7.2 零样本推理

```bash
cd cosmos-reason2

# 复制 Critic 脚本和评分 Prompt 模板
mkdir -p examples/gr00t-dreams
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/inference_videophy2.py examples/gr00t-dreams/
cp ../cosmos-cookbook/docs/recipes/end2end/gr00t-dreams/assets/video_reward.yaml prompts/

# 批量评估所有生成视频
uv run examples/gr00t-dreams/inference_videophy2.py \
    --video-dir ../cosmos-predict2.5/outputs/retail_grasp_rollouts \
    --output-dir outputs/retail_grasp_critic
```

**输出**: 每个视频生成一个 JSON，包含评分和解释：

```json
{
    "video": "retail_grasp_001.mp4",
    "score": 4.0,
    "explanation": "The robot arm smoothly reaches toward the cereal box on the supermarket shelf, grasps it firmly, lifts it without slippage, and deposits it into the shopping cart. The motion follows gravitational constraints, the gripper maintains contact throughout the grasp, and no objects clip or deform unnaturally."
}
```

### 7.3  零售场景的 Critic Prompt 定制

默认评分 Prompt（来自 `video_reward.yaml`）评估 4 个维度：

| 维度 | 零售场景对应含义 |
|------|----------------|
| **Object Behavior** | 商品是否按预期物理属性运动（刚性包装不变形、液体瓶自然晃动） |
| **Motion and Forces** | 机械臂抓取动作是否符合重力、惯性、动量守恒 |
| **Interactions** | 夹爪与商品之间无穿透、商品放入购物篮时碰撞自然 |
| **Consistency Over Time** | 视频帧间一致性，无突变的商品位置跳跃或消失 |

如需定制评分标准（例如增加"商品标签识别正确性"），可编辑 `prompts/video_reward.yaml` 中的 `user_prompt`。

---

### 7.4  零售抓取数据过滤脚本

根据 Reason 2 Critic 评分筛选高质量轨迹：

```bash
#!/bin/bash
# filter_retail_grasp_trajectories.sh
# 从 Critic 输出中筛选 score ≥ 4.0 的轨迹

CRITIC_DIR="./outputs/retail_grasp_critic"
FILTERED_DIR="./outputs/retail_grasp_filtered"
mkdir -p $FILTERED_DIR

echo "=== 筛选零售抓取高质量轨迹 ==="
echo "阈值: score >= 4.0"
echo ""

COUNT=0
TOTAL=0

for json_file in $CRITIC_DIR/*.json; do
    TOTAL=$((TOTAL + 1))
    score=$(python -c "import json; print(json.load(open('$json_file'))['score'])")
    video_name=$(basename "$json_file" .json)

    if (( $(echo "$score >= 4.0" | bc -l) )); then
        cp "../cosmos-predict2.5/outputs/retail_grasp_rollouts/${video_name}.mp4" \
           "$FILTERED_DIR/"
        echo "✅ $video_name | Score: $score | 保留"
        COUNT=$((COUNT + 1))
    else
        echo "❌ $video_name | Score: $score | 丢弃"
    fi
done

echo ""
echo "=== 筛选完成 ==="
echo "总共: $TOTAL 条, 保留: $COUNT 条, 丢弃: $((TOTAL - COUNT)) 条"
echo "保留率: $(echo "scale=1; $COUNT * 100 / $TOTAL" | bc)%"
```

```bash
chmod +x filter_retail_grasp_trajectories.sh
./filter_retail_grasp_trajectories.sh
```

---

## 8.  Cosmos Reason 2 Critic 微调（可选但推荐）

> 来源: Dev_book.md 新增的 Physical Plausibility Prediction with Cosmos Reason 2 章节。  
> **核心价值**: 零样本 Critic 虽可直接使用，但在零售场景通过微调 VideoPhy-2 可显著提升对商品抓取物理合理性的判断准确率。

### 8.1 为什么需要微调 Critic

| 对比维度 | 零样本 (Zero-Shot) | 微调后 (SFT) |
|---------|-------------------|-------------|
| Reason 2 准确率 | 基线 | **+25%** (vs Reason 1 零样本) |
| Reason 2 相关性 | 基线 | Pearson 0.419 (SFT Step 80) |
| 零售场景适用性 | 通用物理理解 | 可针对商品抓取特化 |
| 计算成本 | 无额外训练 | 8×GPU ~数小时 |

### 8.2 VideoPhy-2 数据集

VideoPhy-2 是以动作为中心的物理常识评估基准，采用 1-5 分制人工标注：

| 划分 | 大小 | HuggingFace 地址 |
|------|------|-----------------|
| 训练集 | 3.4k 视频 | `videophysics/videophy2_train` |
| 评估集 | 3.3k 视频 | `videophysics/videophy2_test` |

**六大物理评估维度**（与零售抓取高度相关）:

| 维度 | 零售场景示例 |
|------|-------------|
| 守恒定律 | 商品被抓取后质量和动量是否合理 |
| 重力效应 | 商品从货架抓起后下落轨迹是否自然 |
| 碰撞动力学 | 夹爪与商品、商品与货架之间的碰撞检测 |
| 时间因果 | 抓取→提起→移动→放置的因果链条 |
| 空间约束 | 商品不穿透货架、夹爪边界合理 |
| 弹性形变 | 软包装商品（如袋装零食）的形变是否合理 |

### 8.3 零样本评估（不微调，快速验证）

```bash
cd cosmos-reason2

# 复制 video_critic 文件夹
mkdir -p examples/video_critic
cp -r ../cosmos-cookbook/scripts/examples/reason2/physical-plausibility-check/video_critic examples/

# 复制评分 Prompt 模板
cp ../cosmos-cookbook/docs/recipes/post_training/reason2/physical-plausibility-check/assets/video_reward.yaml prompts/

# 对 VideoPhy-2 测试集运行零样本推理
uv run examples/video_critic/inference_videophy2.py \
    --model nvidia/Cosmos-Reason2-8B \
    --output-dir results/videophy2_test \
    --dataset videophysics/videophy2_test \
    --split test \
    --input-file prompts/video_reward.yaml

# 计算评估指标（准确率 + Pearson 相关性）
python3 examples/video_critic/compute_metrics.py results/videophy2_test
```

### 8.4 监督微调 (SFT)

```bash
cd cosmos-reason2

# Step 1: 下载并预处理 VideoPhy-2 训练数据
uv run examples/video_critic/download_videophy2_train.py \
    --dataset videophysics/videophy2_train \
    --split train \
    --output data/videophy2_train \
    --prompt_path prompts/video_reward.yaml

# Step 2: 启动微调训练
# 配置文件位于 examples/video_critic/configs/videophy2_sft.toml
# 关键超参数:
#   - learning rate: 1e-6 (经过 1e-5 和 2e-7 超参搜索，1e-6 最优)
#   - epochs: 10
#   - batch_size: 32 per replica
#   - dp_shard_size: 设为你的 GPU 数量

cd examples/cosmos_rl
uv run cosmos-rl \
    --config ../video_critic/configs/videophy2_sft.toml \
    scripts/hf_sft.py

# Step 3: 评估微调后的检查点
# 替换 {timestamp} 和 {step_number}:
uv run examples/video_critic/inference_videophy2.py \
    --model outputs/videophy2_sft/{timestamp}/safetensors/step_{step_number} \
    --output-dir results/videophy2_test_sft_step_{step_number} \
    --dataset videophysics/videophy2_test \
    --split test \
    --input-file prompts/video_reward.yaml

python3 examples/video_critic/compute_metrics.py results/videophy2_test_sft_step_{step_number}
```

### 8.5 训练配置参考 (TOML)

```toml
[custom.dataset]
path = "../../data/videophy2_train"

[train]
seed = 42
train_batch_per_replica = 32
epoch = 10
optm_lr = 1e-6
optm_weight_decay = 0.01
optm_warmup_steps = 0.03
optm_decay_type = "cosine"
optm_grad_norm_clip = 1.0
output_dir = "../../outputs/videophy2_sft_cosmos_rl"
compile = false

[policy]
model_name_or_path = "nvidia/Cosmos-Reason2-8B"
model_max_length = 4096

[logging]
logger = ['console', 'wandb']
project_name = "cosmos_reason2_physical_plausibility"
experiment_name = "cosmos_rl/videophy2_sft"

[train.train_policy]
type = "sft"
conversation_column_name = "conversations"
mini_batch = 4

[train.ckpt]
enable_checkpoint = true
max_keep = 20

[policy.parallelism]
tp_size = 1
cp_size = 1
dp_shard_size = 8    # 设为你的 GPU 数量
pp_size = 1
```

> **注意**: 如果 GPU 显存较小，增加 `tp_size` 启用张量并行。上述配置假设单 GPU 可容纳模型（A100/H100），仅使用数据并行。

### 8.6  零售场景适配建议

微调 Critic 后，可以用零售抓取视频 + 人工标注的物理合理性分数构建自己的微调数据集，替代 VideoPhy-2 做二阶段 SFT：

```
阶段一: VideoPhy-2 通用物理合理性微调 (3.4k 视频)
         ↓
阶段二: 零售抓取特化微调 (用你自己的 ~500-1000 条标注视频)
         ↓
结果: Critic 对零售抓取场景的评分准确率显著提升
```

## 9. 计算评估指标

> 两类指标: (A) 生成视频质量指标 (FID/FVD) + (B) Critic 自身性能指标 (准确率/相关性)

### 9.1 生成视频质量指标 (FID/FVD)

```bash
cd cosmos-predict2.5

# 安装指标依赖
pip install decord torchmetrics[image] torch-fidelity cd-fvd einops scipy

# 计算 FID (Fréchet Inception Distance) - 图像级质量
python scripts/metrics/compute_fid_single_view.py \
    --pred_video_paths "./outputs/retail_grasp_filtered/*.mp4" \
    --gt_video_paths "./path/to/ground_truth_videos/*.mp4" \
    --num_frames 57 \
    --output_file fid_retail_results.json

# 计算 FVD (Fréchet Video Distance) - 视频级时空质量
python scripts/metrics/compute_fvd.py \
    --pred_video_paths "./outputs/retail_grasp_filtered/*.mp4" \
    --gt_video_paths "./path/to/ground_truth_videos/*.mp4" \
    --num_frames 57 \
    --output_file fvd_retail_results.json
```

**分数参考**:

| 评级 | FID | FVD | 建议 |
|------|-----|-----|------|
| 优秀 | < 30 | < 100 | 可直接用于下游训练 |
| 良好 | 30-50 | 100-200 | 可接受 |
| 一般 | 50-100 | 200-400 | 需要改进 |

### 9.2 Critic 自身性能指标

评估 Reason 2 Critic 打分与人工标注的一致性（参考 Dev_book Physical Plausibility 章节）:

```bash
cd cosmos-reason2

# 计算准确率 (exact integer match) 和 Pearson 相关性
python3 examples/video_critic/compute_metrics.py results/videophy2_test
```

**输出**:
- `accuracy`: 预测分数与人工标注的精确整数匹配百分比
- `pearson_correlation`: 预测分数与人工标注的 Pearson 相关系数
- `summary.json`: 汇总结果文件

**Dev_book 参考结果**:

| 模型 | 零样本准确率 | 零样本相关性 | SFT 后最佳准确率 | SFT 后最佳相关性 |
|------|------------|------------|----------------|----------------|
| Cosmos Reason 1 | 基线 | 基线 | 0.395 (Step 60) | 0.395 (Step 60) |
| Cosmos Reason 2 | **+25%** | **+12%** | 0.401 (Step 40) | 0.419 (Step 80) |

---

## 10. 局限性与注意事项

1. **合成 vs. 真实数据**：生成的轨迹来自学习到的分布，物理上可能合理但仍可能与真实机器人数据有差异，纯合成数据训练的下游策略存在 sim-to-real gap。

2. **Critic 是代理而非真值**：Reason 2 视频 Critic 是物理合理性的代理评估，可能遗漏细粒度物理问题，Best-of-N 采样提升质量但不能保证正确性。

3. **领域泛化**：模型在 GR1/GR00T 风格操作数据上表现最强，泛化到不同机器人构型或任务可能需要更多数据或调优。

4. **计算成本**：Best-of-N 和多次 Rollout 会增加计算成本，需按需平衡质量与吞吐量。

5. **零售场景的特殊挑战**：
   - 透明包装商品（如塑料瓶装水）的视觉识别和物理建模更困难
   - 拥挤货架中多商品间的碰撞检测更复杂
   - 不同商超光照条件（荧光灯 vs. 自然光）影响生成质量

---

## 11. 完整执行清单

按顺序执行以下命令即可跑通完整管线（含零售适配）：

```bash
# ==================== Phase 1: 环境初始化 ====================
# 系统依赖
sudo apt update && sudo apt install -y build-essential git curl ffmpeg

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 克隆仓库
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git

# 安装 Cosmos Predict 2.5
cd cosmos-predict2.5
uv sync
source .venv/bin/activate

# 安装 Cosmos Reason 2
cd ../cosmos-reason2
uv sync --extra=cu128
source .venv/bin/activate

# 安装 HuggingFace CLI 并登录
uv tool install -U "huggingface_hub[cli]"
hf auth login

# 环境变量
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output
export HF_HOME=/path/to/huggingface/cache
export WANDB_API_KEY=your_key   # 或用 job.wandb_mode=disabled

# ==================== Phase 2: 数据准备 ====================
cd ../cosmos-predict2.5

# 下载 GR1-100
huggingface-cli download nvidia/GR1-100 --repo-type dataset \
    --local-dir datasets/benchmark_train/hf_gr1/
mkdir -p datasets/benchmark_train/gr1/videos
mv datasets/benchmark_train/hf_gr1/gr1/*.mp4 datasets/benchmark_train/gr1/videos/
mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/

#  零售适配: 加入零售抓取视频和 metadata
# cp ./retail_videos/*.mp4 datasets/benchmark_train/gr1/videos/

# 生成 Prompt 文件
python -m scripts.create_prompts_for_gr1_dataset \
    --dataset_path datasets/benchmark_train/gr1

# ==================== Phase 3: 后训练 ====================
torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2_video2world_training_2b_groot_gr1_480 \
    job.wandb_mode=disabled

# ==================== Phase 4: 检查点转换 ====================
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

# ==================== Phase 5: 单轨迹生成验证 ====================
torchrun --nproc_per_node=1 examples/inference.py \
    -i assets/sample_gr00t_dreams_gr1/gr00t_image2world.json \
    -o outputs/retail_grasp_sample \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment predict2_video2world_training_2b_groot_gr1_480

# ==================== Phase 6: 批量 Rollout 生成 ====================
huggingface-cli download nvidia/EVAL-175 --repo-type dataset \
    --local-dir dream_gen_benchmark
python -m scripts.prepare_batch_input_json \
    --dataset_path dream_gen_benchmark/gr1_object/ \
    --save_path output/dream_gen_benchmark/cosmos_predict2_14b_gr1_object/ \
    --output_path dream_gen_benchmark/gr1_object/batch_input.json
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/gr1_batch_to_jsonl.py scripts/
python scripts/gr1_batch_to_jsonl.py
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/inference.py .
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/config.py .
torchrun --nproc_per_node=1 inference.py \
    -i dream_gen_benchmark/gr1_object/gr1_batch.jsonl \
    -o outputs/retail_grasp_rollouts \
    --num-generations 5 \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment predict2_video2world_training_2b_groot_gr1_480

# ==================== Phase 7: Reason 2 Critic 零样本筛选 ====================
cd ../cosmos-reason2
mkdir -p examples/gr00t-dreams
cp -r ../cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/inference_videophy2.py examples/gr00t-dreams/
cp ../cosmos-cookbook/docs/recipes/end2end/gr00t-dreams/assets/video_reward.yaml prompts/
uv run examples/gr00t-dreams/inference_videophy2.py \
    --video-dir ../cosmos-predict2.5/outputs/retail_grasp_rollouts \
    --output-dir outputs/retail_grasp_critic

# ==================== Phase 7b (可选): Reason 2 Critic SFT 微调 ====================
# 在 VideoPhy-2 上微调 Critic 提升物理判断准确率
cp -r ../cosmos-cookbook/scripts/examples/reason2/physical-plausibility-check/video_critic examples/
cp ../cosmos-cookbook/docs/recipes/post_training/reason2/physical-plausibility-check/assets/video_reward.yaml prompts/
uv run examples/video_critic/download_videophy2_train.py \
    --dataset videophysics/videophy2_train \
    --split train \
    --output data/videophy2_train \
    --prompt_path prompts/video_reward.yaml
cd examples/cosmos_rl
uv run cosmos-rl --config ../video_critic/configs/videophy2_sft.toml scripts/hf_sft.py
cd ../..

# ==================== Phase 8: 筛选高质量轨迹 ====================
cd ..
./filter_retail_grasp_trajectories.sh

echo "✅ GR00T-Dreams 零售抓取管线执行完毕！"
echo "高质量轨迹保存在: cosmos-predict2.5/outputs/retail_grasp_filtered/"
echo "Critic 评分结果: cosmos-reason2/outputs/retail_grasp_critic/"
```

---

## 12. 引用

```
@misc{cosmos_cookbook_gr00t_2026,
  title={Leveraging World Foundation Models for Synthetic Trajectory Generation in Robot Learning},
  author={Apte, Rucha and Jin, Jingyi and Nanda, Saurav},
  organization={NVIDIA},
  year={2026},
  month={March},
  howpublished={\url{https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/end2end/gr00t-dreams/post-training.html}},
  note={NVIDIA Cosmos Cookbook}
}
```
