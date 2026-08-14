# 接口文档

> **项目**: RDL-Dreams — 基于 Cosmos GR00T-Dreams 的商超零售机器人抓取轨迹生成  
> **主方案**: [../README.md](../README.md) | **官方菜谱**: [../../Dev_book.md](../../Dev_book.md)

---

## 1. CLI 接口总览

| 命令 | 入口 | 输入 | 输出 | 功能 |
|------|------|------|------|------|
| `scripts.train` | Python 模块 | 训练数据 + 配置 | DCP 检查点 | Predict 2.5 后训练 |
| `inference.py` | Python 脚本 | JSON/JSONL + 初始帧 | 轨迹视频 MP4 | 视频轨迹生成 |
| `convert_distcp_to_pt.py` | Python 脚本 | DCP 目录 | `.pt` 文件 | 检查点格式转换 |
| `inference_videophy2.py` | Python 脚本 | 视频目录 + Prompt YAML | 评分 JSON | Critic 物理合理性评分 |
| `cosmos-rl` | CLI (uv) | SFT/RL TOML 配置 | 微调后模型 | Reason 2 SFT/RL 训练 |
| `filter_retail_grasp.sh` | Bash 脚本 | 评分 + 视频目录 | 筛选后视频 | 阈值筛选 |

---

## 2. Predict 2.5 训练接口

### 2.1 `scripts.train`

```bash
torchrun --nproc_per_node=<N_GPU> --master_port=<PORT> \
    -m scripts.train \
    --config=<CONFIG_PATH> \
    -- experiment=<EXPERIMENT_NAME>
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--nproc_per_node` | int | ✅ | GPU 数量 |
| `--master_port` | int | ✅ | 分布式通信端口 |
| `--config` | path | ✅ | 配置文件路径 |
| `experiment` | string | ✅ | 实验名称 (与 config.py 中对应) |

### 2.2 配置结构

```python
# cosmos_predict2/_src/predict2/configs/video2world/config.py

experiment = dict(
    video2world=dict(
        resolution=[480, 960],        # [H, W]
        num_input_frames=1,           # 输入帧数
        num_output_frames=93,         # 输出帧数
    ),
    model=dict(
        predict2_2b_initialized=False,
        checkpoint_path="nvidia/Cosmos-Predict2.5-Video2World-2B",
    ),
    dataset=dict(
        batch_size=1,
        workers=4,
    ),
    optimizer=dict(
        lr=1e-5,
        weight_decay=0.01,
    ),
)
```

---

## 3. Predict 2.5 推理接口

### 3.1 `inference.py`

```bash
torchrun --nproc_per_node=<N_GPU> \
    inference.py \
    -i <INPUT_JSON_OR_JSONL> \
    -o <OUTPUT_DIR> \
    --checkpoint-path <PT_FILE> \
    --experiment <EXPERIMENT> \
    [--num-generations <N>]
```

| 参数 | 短名 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| `--input` | `-i` | path | ✅ | 单条 JSON 或批量 JSONL |
| `--output-dir` | `-o` | path | ✅ | 输出目录 |
| `--checkpoint-path` | | path | ✅ | PT 格式检查点路径 |
| `--experiment` | | string | ✅ | 实验配置名 |
| `--num-generations` | | int | 否 | 每 Prompt 生成候选数 (默认 1) |
| `--seed-offset` | | int | 否 | 种子偏移量 |

### 3.2 输入 JSON Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["name", "inference_type", "prompt", "input_path", "num_output_frames", "resolution", "seed"],
    "properties": {
        "name": {
            "type": "string",
            "description": "场景唯一标识符"
        },
        "inference_type": {
            "type": "string",
            "enum": ["image2world", "video2world"],
            "description": "推理模式"
        },
        "prompt": {
            "type": "string",
            "description": "任务语言描述"
        },
        "input_path": {
            "type": "string",
            "description": "初始帧图像文件路径 (PNG/JPEG)"
        },
        "input_video_path": {
            "type": "string",
            "description": "Video2World 模式下的历史视频路径"
        },
        "num_output_frames": {
            "type": "integer",
            "minimum": 1,
            "maximum": 256,
            "description": "生成帧数"
        },
        "resolution": {
            "type": "string",
            "pattern": "^\\d+,\\d+$",
            "description": "格式 \"H,W\"，如 \"480,960\""
        },
        "seed": {
            "type": "integer",
            "minimum": 0,
            "maximum": 4294967295,
            "description": "随机种子"
        },
        "guidance": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 20.0,
            "default": 7.0,
            "description": "引导强度"
        },
        "negative_prompt": {
            "type": "string",
            "description": "负向提示词"
        }
    }
}
```

### 3.3 输出格式

```
输出目录/
├── {name}_seed_{seed}.mp4          # 生成的轨迹视频
└── {name}_seed_{seed}_meta.json    # 生成元数据
```

元数据示例：
```json
{
    "name": "retail_grasp_cereal_top_shelf",
    "resolution": [432, 768],
    "num_frames": 93,
    "fps": 30,
    "seed": 0,
    "guidance": 7.0,
    "prompt": "The robot arm is performing a task...",
    "checkpoint": "model_ema_bf16.pt",
    "experiment": "predict2_video2world_training_2b_groot_gr1_480"
}
```

---

## 4. Checkpoint 转换接口

### 4.1 `convert_distcp_to_pt.py`

```bash
python scripts/convert_distcp_to_pt.py \
    <DCP_DIR> \
    <OUTPUT_DIR>
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `DCP_DIR` | path | ✅ | DCP 检查点目录 (包含 __0_0.distcp 等) |
| `OUTPUT_DIR` | path | ✅ | PT 文件输出目录 |

输出：
- `model_bf16.pt` — 直接版本
- `model_ema_bf16.pt` — **EMA 平滑版本 (推理推荐)**

---

## 5. Critic 评分接口

### 5.1 `inference_videophy2.py`

```bash
uv run examples/gr00t-dreams/inference_videophy2.py \
    --video-dir <VIDEO_DIR> \
    --output-dir <OUTPUT_DIR> \
    [--model <MODEL_NAME_OR_PATH>] \
    [--input-file <PROMPT_YAML>]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--video-dir` | path | ✅ | 待评分视频所在目录 |
| `--output-dir` | path | ✅ | 评分结果输出目录 |
| `--model` | string | 否 | 模型名称或路径，默认 `nvidia/Cosmos-Reason2-8B` |
| `--input-file` | path | 否 | Prompt YAML 配置路径 |
| `--dataset` | string | 否 | VideoPhy-2 评估模式：`videophysics/videophy2_test` |
| `--split` | string | 否 | 数据集 split |

### 5.2 Prompt YAML Schema

```yaml
# configs/critic/video_reward.yaml
system_prompt: "You are a helpful assistant."

user_prompt: |
  You are a helpful video analyzer. Evaluate whether the video follows
  physical commonsense.

  Evaluation Criteria:
  1. **Object Behavior:** ...
  2. **Motion and Forces:** ...
  3. **Interactions:** ...
  4. **Consistency Over Time:** ...

  Instructions for Scoring:
  - **1:** No adherence to physical commonsense.
  - **2:** Poor adherence.
  - **3:** Moderate adherence.
  - **4:** Good adherence.
  - **5:** Perfect adherence.

  Response Template:
  [Score between 1 and 5.]

  Does this video adhere to the physical laws?
```

### 5.3 评分输出 Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["video_id", "score", "video_path"],
        "properties": {
            "video_id": {
                "type": "string",
                "description": "视频唯一标识符 (与生成时的 name 对应)"
            },
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "物理合理性评分"
            },
            "video_path": {
                "type": "string",
                "description": "视频文件绝对路径"
            },
            "reasoning": {
                "type": "string",
                "description": "评分理由 (可选，取决于 Critic 配置)"
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "评分时间戳"
            }
        }
    }
}
```

---

## 6. Cosmos RL 训练接口

### 6.1 `cosmos-rl`

```bash
uv run cosmos-rl \
    --config <TOML_PATH> \
    scripts/hf_sft.py
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--config` | path | ✅ | TOML 训练配置文件 |
| `scripts/hf_sft.py` | — | ✅ | SFT 训练脚本 (固定) |
| `scripts/hf_rl.py` | — | ✅ | RL 训练脚本 (替代) |

### 6.2 TOML 配置 Schema

```toml
# configs/critic/videophy2_sft.toml

[custom.dataset]
path = "../../data/videophy2_train"      # 数据集路径

[train]
seed = 42                                 # 随机种子
train_batch_per_replica = 32              # 全局 batch size
epoch = 10                                # 训练轮数
optm_lr = 1e-6                            # 学习率
optm_weight_decay = 0.01                  # 权重衰减
optm_warmup_steps = 0.03                  # warmup 比例
optm_decay_type = "cosine"                # LR 衰减策略
optm_grad_norm_clip = 1.0                # 梯度裁剪
output_dir = "../../outputs/videophy2_sft_cosmos_rl"
compile = false                           # torch.compile

[policy]
model_name_or_path = "nvidia/Cosmos-Reason2-8B"
model_max_length = 4096

[policy.parallelism]
tp_size = 1                               # 张量并行
cp_size = 1                               # 上下文并行
dp_shard_size = 8                         # 数据并行 (设 GPU 数)
pp_size = 1                               # 流水线并行

[logging]
logger = ['console', 'wandb']
project_name = "cosmos_reason2_physical_plausibility"
experiment_name = "cosmos_rl/videophy2_sft"

[train.train_policy]
type = "sft"                              # "sft" 或 "rl"
conversation_column_name = "conversations"
mini_batch = 4

[train.ckpt]
enable_checkpoint = true
max_keep = 20
```

---

## 7. 筛选脚本接口

### 7.1 `filter_retail_grasp.sh`

```bash
bash scripts/filter_retail_grasp.sh \
    <CRITIC_DIR> \
    <VIDEO_DIR> \
    [THRESHOLD]
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `CRITIC_DIR` | path | ✅ | — | Critic 评分 JSON 文件目录 |
| `VIDEO_DIR` | path | ✅ | — | 轨迹视频目录 |
| `THRESHOLD` | number | 否 | `4.0` | 筛选阈值 (≥ 此值保留) |

输出：
- `<VIDEO_DIR>/filtered/` — 合格视频的副本
- 终端输出筛选统计 (总数、通过数、通过率)

---

## 8. 零售场景 Prompt 模板

### 8.1 商品类别枚举

| 类别 | 可用的商品名称 |
|------|--------------|
| **包装食品** | `cereal box`, `canned soup`, `bag of chips`, `box of cookies` |
| **饮料** | `bottled water`, `soda can`, `milk carton`, `juice bottle` |
| **日用品** | `bottle of shampoo`, `box of detergent`, `roll of paper towels` |
| **生鲜** | `bag of apples`, `head of lettuce`, `bunch of bananas` |

### 8.2 货架位置枚举

| 英文 | 中文 |
|------|------|
| `top level of supermarket shelf` | 货架顶层 |
| `middle shelf of convenience store aisle` | 便利店中层货架 |
| `bottom shelf of grocery aisle` | 杂货通道底层货架 |
| `eye-level shelf of retail display` | 视线高度货架 |

### 8.3 目标位置枚举

| 英文 | 中文 |
|------|------|
| `shopping cart` | 购物车 |
| `plastic basket` | 塑料购物篮 |
| `conveyor belt at checkout` | 收银台传送带 |
| `paper bag` | 纸袋 |

### 8.4 完整 Prompt 格式

```
"The robot arm is performing a task. Use the right hand to pick up
 {商品名} from {货架位置} to {目标位置}."
```

---

## 9. 错误码与含义

| 退出码 | 含义 | 解决 |
|--------|------|------|
| 0 | 成功 | — |
| 1 | 通用错误 | 检查日志 |
| 137 | OOM (被系统 KILL) | 减少 GPU 数量或降低分辨率 |
| 139 | 段错误 (Segfault) | 检查 CUDA 版本兼容性 |
| `CUDA_ERROR_OUT_OF_MEMORY` | 显存不足 | 降 batch size 或分辨率 |
| `NCCL_ERROR` | 分布式通信故障 | 检查网络和 NCCL 配置 |
