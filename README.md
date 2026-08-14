# proj_RDL-Dreams

> **Retail DreamGrasp — 基于 Cosmos GR00T-Dreams 的商超零售机器人抓取轨迹生成项目**
>


---

## 目录结构与管线对应

```
proj_RDL-Dreams/
│
├── data/                           #  Phase 2: 数据准备
│   ├── gr1_baseline/               #   GR1-100 基线数据集
│   │   ├── videos/                 #     原始 GR1 操作视频 (.mp4)
│   │   └── metas/                  #     每个视频对应的 Prompt 文本 (.txt)
│   ├── retail/                     #    零售场景数据
│   │   ├── sim_videos/             #     仿真器生成 (Isaac Sim / MuJoCo)
│   │   ├── real_videos/            #     真实商超录制视频
│   │   ├── transfer_enhanced/      #     Cosmos Transfer 2.5 Sim2Real 增强后
│   │   └── curated/                #     Cosmos Curator 策展后最终数据
│   ├── metadata/                   #   数据集标注 (metadata.csv)
│   └── prompts/                    #   零售场景 Prompt 模板
│
├── models/                         #  Phase 3-4: 模型与检查点
│   ├── predict_checkpoints/        #   Cosmos Predict 2.5 后训练检查点
│   ├── reason_checkpoints/         #   Cosmos Reason 2 模型缓存
│   └── critic_sft/                 #   Critic SFT 微调后检查点 (Phase 8)
│
├── inference/                      #  Phase 5-6: 推理输入
│   ├── input_images/               #   初始帧图像 (.png)
│   ├── input_jsonl/                #   批量推理 JSONL 文件
│   └── configs/                    #   推理配置 (inference.py config)
│
├── outputs/                        #  Phase 5-8: 所有输出
│   ├── trajectories/
│   │   ├── single/                 #   单视频推演输出
│   │   ├── rollouts/               #   批量多候选推演 (--num-generations 5)
│   │   └── filtered/               #    筛选后高质量轨迹 (Score ≥ 4.0)
│   ├── critic_scores/              #   Reason 2 Critic 评分 JSON
│   └── videophy2_eval/             #   VideoPhy-2 Critic 评估结果
│
├── metrics/                        #  Phase 9: 评估指标
│   ├── fid_fvd/                    #   FID/FVD 视频质量指标
│   └── critic_performance/         #   Critic 准确率/相关性
│
├── configs/                        #  配置文件
│   ├── training/                   #   Predict 2.5 训练配置 (config.py 导出)
│   └── critic/                     #   video_reward.yaml, videophy2_sft.toml
│
├── scripts/                        #  自定义脚本
│   └── filter_retail_grasp.sh      #   Critic 评分筛选脚本
│
├── logs/                           #  日志
│   ├── training/                   #   训练日志
│   └── inference/                  #   推理日志
│
└── docs/                           #  项目文档
```

---

## 快速开始

### 1. 准备零售抓取数据

```bash
# 方案 A: 仿真器生成 → 放入 data/retail/sim_videos/
# 方案 B: 真实录制   → 放入 data/retail/real_videos/

# Sim2Real 增强后 → data/retail/transfer_enhanced/
# Curator 策展后   → data/retail/curated/
```

### 2. 合并训练数据

```bash
# 将 GR1 基线 + 零售数据合并
cp data/gr1_baseline/videos/*.mp4   <cosmos-predict2.5>/datasets/benchmark_train/gr1/videos/
cp data/retail/curated/*.mp4        <cosmos-predict2.5>/datasets/benchmark_train/gr1/videos/
cp data/retail/curated/metadata.csv <cosmos-predict2.5>/datasets/benchmark_train/gr1/
```


---

## 关键配置文件说明

| 文件 | 用途 | 对应 README 章节 |
|------|------|-----------------|
| `configs/critic/video_reward.yaml` | Critic 物理合理性评分 Prompt | 7.2 |
| `configs/critic/videophy2_sft.toml` | Critic SFT 训练配置 | 8.5 |
| `data/prompts/retail_prompt_template.txt` | 零售抓取 Prompt 模板 | 3.4 |
| `inference/input_jsonl/example.json` | 推理输入示例 | 5.3 |
| `scripts/filter_retail_grasp.sh` | 轨迹筛选脚本 | 7.4 |
