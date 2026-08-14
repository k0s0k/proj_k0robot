#!/bin/bash
# ============================================================================
# filter_retail_grasp.sh
# 根据 Cosmos Reason 2 Critic 评分筛选高质量零售抓取轨迹
# 用法: ./filter_retail_grasp.sh [CRITIC_DIR] [VIDEO_DIR] [THRESHOLD]
#
# 对应 README.md 第 7.4 节
# ============================================================================

set -e

# 默认参数
CRITIC_DIR="${1:-./outputs/critic_scores}"
VIDEO_DIR="${2:-./outputs/trajectories/rollouts}"
FILTERED_DIR="./outputs/trajectories/filtered"
THRESHOLD="${3:-4.0}"

mkdir -p "$FILTERED_DIR"

echo "============================================"
echo "  RDL-Dreams 零售抓取轨迹筛选"
echo "============================================"
echo "  Critic 评分目录 : $CRITIC_DIR"
echo "  轨迹视频目录    : $VIDEO_DIR"
echo "  筛选阈值        : Score >= $THRESHOLD"
echo "  输出目录        : $FILTERED_DIR"
echo "============================================"
echo ""

COUNT=0
TOTAL=0

for json_file in "$CRITIC_DIR"/*.json; do
    [ -e "$json_file" ] || continue
    TOTAL=$((TOTAL + 1))

    score=$(python3 -c "import json; print(json.load(open('$json_file'))['score'])" 2>/dev/null || echo "0")
    video_name=$(basename "$json_file" .json)

    if (( $(echo "$score >= $THRESHOLD" | bc -l) )); then
        if [ -f "$VIDEO_DIR/${video_name}.mp4" ]; then
            cp "$VIDEO_DIR/${video_name}.mp4" "$FILTERED_DIR/"
            echo "✅ $video_name | Score: $score | 保留"
        else
            echo "⚠️  $video_name | Score: $score | 视频文件缺失"
        fi
        COUNT=$((COUNT + 1))
    else
        echo "❌ $video_name | Score: $score | 丢弃"
    fi
done

echo ""
echo "============================================"
echo "  筛选完成"
echo "============================================"
echo "  总共    : $TOTAL 条"
echo "  保留    : $COUNT 条"
echo "  丢弃    : $((TOTAL - COUNT)) 条"

if [ $TOTAL -gt 0 ]; then
    RATE=$(echo "scale=1; $COUNT * 100 / $TOTAL" | bc)
    echo "  保留率  : ${RATE}%"
fi

echo "  输出目录: $FILTERED_DIR"
echo "============================================"
