# 数据处理流程

> **项目**: RDL-Dreams — 基于 Cosmos GR00T-Dreams 的商超零售机器人抓取轨迹生成  
> **主方案**: [../README.md](../README.md) | **官方菜谱**: [../../Dev_book.md](../../Dev_book.md) | **架构**: [architecture.md](architecture.md)

---

## 1. 数据管线全景

零售抓取场景的数据挑战在于：没有现成的"商超机器人抓取"公开数据集。RDL-Dreams 采用 **仿真合成 + Sim2Real 增强 + 策展过滤** 的三阶段管线。

```
数据管线全景
═══════════════════════════════════════════════════════════════════

 阶段 A: 基线数据                         阶段 B: 零售合成数据
 ┌──────────────────┐                    ┌──────────────────────┐
 │ GR1-100           │                    │ Isaac Sim / MuJoCo   │
 │ (HuggingFace)     │                    │                      │
 │                    │                    │ 场景: 商超货架+机械臂  │
 │ 100 条桌面操作     │                    │ 任务: 抓取-放置       │
 │ 训练轨迹          │                    │ 输出: 合成视频序列    │
 └────────┬─────────┘                    └──────────┬───────────┘
          │                                         │
          │                                         ▼
          │                               ┌──────────────────────┐
          │                               │ Cosmos Transfer 2.5  │
          │                               │ Sim2Real (Edge Ctrl) │
          │                               │                      │
          │                               │ 合成 → 真实风格增强   │
          │                               └──────────┬───────────┘
          │                                          │
          ▼                                          ▼
 ┌─────────────────────────────────────────────────────────────┐
 │              阶段 C: 策展 & 合并                               │
 │                                                              │
 │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
 │  │ 分割     │ → │ 转码     │ → │ 过滤     │ → │ 字幕     │  │
 │  │ Splitter │   │ Transcoder│   │ Filter   │   │ Caption  │  │
 │  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
 │                                                     │        │
 │                                          ┌──────────┘        │
 │                                          ▼                    │
 │                                    ┌──────────┐              │
 │                                    │ 去重     │              │
 │                                    │ Dedup    │              │
 │                                    └────┬─────┘              │
 └─────────────────────────────────────────┼────────────────────┘
                                           │
                                           ▼
                              ┌──────────────────────┐
                              │ 预处理: 生成 Prompt    │
                              │ create_prompts_for_   │
                              │ gr1_dataset.py       │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ 最终训练集             │
                              │ datasets/benchmark_   │
                              │ train/retail_grasp/   │
                              └──────────────────────┘
```

---

## 2. 阶段 A: GR1-100 基线数据

### 2.1 数据集概述

| 属性 | 说明 |
|------|------|
| **名称** | GR1-100 |
| **来源** | NVIDIA (HuggingFace: `nvidia/GR1-100`) |
| **规模** | 100 条桌面操作训练轨迹 |
| **内容** | 机器人桌面操作：拾取、放置、推、拉等基础动作 |
| **格式** | 视频 + 语言描述 Prompt |
| **许可** | 需签署 NVIDIA Open Model License |

### 2.2 下载与预处理

```bash
# Step 1: 登录 HuggingFace
hf auth login

# Step 2: 签署许可 https://huggingface.co/nvidia/GR1-100

# Step 3: 下载数据集
hf download nvidia/GR1-100 --repo-type dataset \
    --local-dir datasets/benchmark_train/hf_gr1/

# Step 4: 预处理 — 生成 JSONL Prompt 文件
python -m scripts.create_prompts_for_gr1_dataset \
    --dataset_path datasets/benchmark_train/gr1

# 输出: datasets/benchmark_train/gr1/gr1_batch.jsonl
```

### 2.3 JSONL 格式示例

```json
{
    "name": "gr1_task_001",
    "inference_type": "image2world",
    "prompt": "The robot arm is performing a task. Use the right hand to pick up the block on the table and place it in the bowl.",
    "input_path": "datasets/benchmark_train/gr1/videos/001/frame_0.png",
    "num_output_frames": 93,
    "resolution": "480,960",
    "seed": 0,
    "guidance": 7
}
```

---

## 3. 阶段 B: 零售合成数据

### 3.1 仿真环境

```
Isaac Sim / MuJoCo 仿真
═══════════════════════════════════
环境:
  - 商超货架模型（多层，含商品 3D 模型）
  - 机械臂模型（Franka / UR5）
  - 相机视角（正面/侧面/俯视）

任务配置:
  - 商品: 麦片盒、瓶装水、罐装汤、洗发水瓶等
  - 起始: 机械臂末端执行器位于待抓取商品上方
  - 目标: 将商品放入购物车/购物篮/传送带

输出:
  - 视频序列: 1280×720, 30fps, RGB
  - 元数据: 商品类别、货架层、成功/失败标签
  - 数量: 建议 ≥ 50 条高质量轨迹
```

### 3.2 Cosmos Transfer 2.5 Sim2Real

合成视频 → 真实风格迁移，提升下游模型泛化能力：

```bash
# Transfer 2.5 推理 (Edge Control 模式)
# 参考: Dev_book.md 的 Transfer 2.5 文档
python -m cosmos_transfer.inference \
    --input-dir data/synthetic_videos/ \
    --output-dir data/sim2real_videos/ \
    --control-type edge \
    --checkpoint nvidia/Cosmos-Transfer2.5
```

| 控制模式 | 保持结构 | 改变风格 |
|---------|---------|---------|
| **Edge** | 商品轮廓、货架边缘 | 材质、光照、色彩 → 照片级真实感 |
| Canny | 更细的边缘信息 | 同上 |
| Depth | 深度结构 | 纹理、外观 |
| Sketch | 素描线条 | 完整视觉风格 |

### 3.3 零售数据要求

| 要求 | 详细 |
|------|------|
| **视频长度** | ≥ 93 帧 (~3 秒 @ 30fps)，与 Predict 2.5 训练配置一致 |
| **分辨率** | 建议 480×960 或以上 |
| **商品多样性** | 至少覆盖 4 类（包装食品/饮料/日用品/生鲜），每类 ≥ 3 种 |
| **场景多样性** | 不同货架排布、不同光照、不同相机角度 |
| **动作多样性** | 不同抓取角度、不同放置位置、不同运动速度 |

---

## 4. 阶段 C: Cosmos Curator 策展

### 4.1 七阶段管线

来自 Cosmos Curator 官方工作流（参见 `11.Cosmos-Curator.md`）：

| 阶段 | 名称 | 功能 | 配置 |
|------|------|------|------|
| 1 | **Splitter** | 长视频切分为短片段 | `segment_duration=3s` |
| 2 | **Transcoder** | 统一编码格式 | 输出 mp4/H.264 |
| 3 | **Motion Filter** | 筛除静态/无运动帧 | 光度流阈值 |
| 4 | **Quality Filter** | 筛除模糊/低质量片段 | 清晰度/光照检测 |
| 5 | **Aesthetic Filter** | 美学评分过滤 | 评分阈值 |
| 6 | **Caption** | 自动生成视频字幕 | VLM-based captioner |
| 7 | **Deduplicate** | 相似片段去重 | 嵌入相似度 |

### 4.2 策展命令参考

```bash
# 安装 Cosmos Curator
cd cosmos-curator
uv sync --extra=cu128

# 运行策展流水线
python -m curator.pipeline \
    --input-dir data/sim2real_videos/ \
    --output-dir data/curated_videos/ \
    --config curator/configs/retail_grasp_curation.yaml
```

---

## 5. 数据格式与预处理

### 5.1 最终训练数据格式

合并 GR1 + 零售数据后，需统一下列格式：

```
datasets/benchmark_train/retail_grasp/
├── gr1/                      # GR1-100 预处理后数据
│   ├── gr1_batch.jsonl       # Prompt 文件
│   └── videos/               # 原始视频
├── retail/                   # 零售合成数据
│   ├── retail_batch.jsonl    # Prompt 文件
│   └── videos/               # 策展后视频
└── combined_batch.jsonl      # 合并后的完整训练集
```

### 5.2 Prompt 模板

见 [`data/prompts/retail_prompt_template.txt`](../data/prompts/retail_prompt_template.txt)，核心格式：

```
"The robot arm is performing a task. Use the right hand to pick up
 <商品名> from <货架位置> to <目标位置>."
```

### 5.3 数据量建议

| 数据组成 | 最少数量 | 推荐数量 | 说明 |
|---------|---------|---------|------|
| GR1-100 (基线) | 100 | 100 | 通用操作先验知识 |
| 零售合成数据 | 20 | 50-100 | 领域特定数据 |
| **总计** | **120** | **150-200** | 混合训练 |

---

## 6. 工具链总结

| 工具 | 版本 | 安装方式 | 用途 |
|------|------|---------|------|
| Isaac Sim | 4.x+ | NVIDIA Omniverse | 3D 仿真，生成合成视频 |
| MuJoCo | 3.x | `pip install mujoco` | 轻量级物理仿真替代方案 |
| Cosmos Transfer 2.5 | 2.5 | `uv sync` | Sim2Real 风格迁移 |
| Cosmos Curator | latest | `uv sync` | 视频策展（分割/过滤/去重） |
| HuggingFace CLI | latest | `pip install huggingface_hub` | 下载 GR1-100 数据集 |
