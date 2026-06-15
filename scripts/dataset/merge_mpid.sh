#!/bin/bash
# =============================================================================
# merge_mpid.sh
# 将三个 MPID 子集（glass / porcelain / composite）合并，并切分为
# train / val / test（80% / 10% / 10%），供 Ultralytics 训练使用。
#
# 实际目录结构（已确认）：
#   data/mpid/glass/glass_MPID/train/{images,labels}/
#   data/mpid/porcelain/porcelain_MPID/train/{images,labels}/
#   data/mpid/composite/composite_MPID/train/{images,labels}/
#
# 三个子集均为单类（nc=1），class id = 0，类别名不同但 id 一致，
# 合并后统一命名为 insulator（见 configs/datasets/mpid.yaml）。
#
# 运行方式（在服务器项目根目录下）：
#   bash scripts/dataset/merge_mpid.sh
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
MPID_BASE="${PROJECT_ROOT}/data/mpid"

# 三个子集的 train 目录
SRCS=(
    "${MPID_BASE}/glass/glass_MPID/train"
    "${MPID_BASE}/porcelain/porcelain_MPID/train"
    "${MPID_BASE}/composite/composite_MPID/train"
)

# 合并后目标目录
DST="${MPID_BASE}/merged"
POOL_IMG="${DST}/_pool/images"
POOL_LBL="${DST}/_pool/labels"

# 分割比例（整数，总和=10）
TRAIN_RATIO=8
VAL_RATIO=1
TEST_RATIO=1

echo "===== MPID 合并 & 切分脚本 ====="
echo "目标目录: ${DST}"
echo ""

# ---- Step 1：汇总所有图片/标签到 pool ----
mkdir -p "$POOL_IMG" "$POOL_LBL"

for src in "${SRCS[@]}"; do
    img_dir="${src}/images"
    lbl_dir="${src}/labels"
    if [ ! -d "$img_dir" ]; then
        echo "[WARN] 跳过不存在的目录: $img_dir"
        continue
    fi
    echo "[INFO] 合并: $src"
    cp -n "${img_dir}"/* "$POOL_IMG/" 2>/dev/null || true
    cp -n "${lbl_dir}"/* "$POOL_LBL/" 2>/dev/null || true
done

TOTAL=$(ls "$POOL_IMG" | wc -l)
echo ""
echo "[INFO] pool 总图片数: ${TOTAL}"

# ---- Step 2：随机切分 ----
# 创建目标目录
for split in train val test; do
    mkdir -p "${DST}/images/${split}"
    mkdir -p "${DST}/labels/${split}"
done

# 生成随机排列的文件名列表（去掉扩展名作为 stem）
STEMS=( $(ls "$POOL_IMG" | sed 's/\.[^.]*$//' | shuf) )
N=${#STEMS[@]}

N_VAL=$(( N * VAL_RATIO / 10 ))
N_TEST=$(( N * TEST_RATIO / 10 ))
N_TRAIN=$(( N - N_VAL - N_TEST ))

echo "[INFO] 切分: train=${N_TRAIN}, val=${N_VAL}, test=${N_TEST}"
echo ""

copy_split() {
    local split=$1
    local start=$2
    local count=$3
    local moved=0

    for (( i=start; i<start+count && i<N; i++ )); do
        stem="${STEMS[$i]}"
        # 找到对应图片文件（可能是 .jpg/.jpeg/.png）
        img_file=$(ls "${POOL_IMG}/${stem}".* 2>/dev/null | head -1)
        lbl_file="${POOL_LBL}/${stem}.txt"

        if [ -n "$img_file" ]; then
            cp "$img_file" "${DST}/images/${split}/"
            moved=$(( moved + 1 ))
        fi
        if [ -f "$lbl_file" ]; then
            cp "$lbl_file" "${DST}/labels/${split}/"
        fi
    done
    echo "[INFO] ${split}: 已复制 ${moved} 张图片"
}

copy_split "train" 0 "$N_TRAIN"
copy_split "val"   "$N_TRAIN" "$N_VAL"
copy_split "test"  "$(( N_TRAIN + N_VAL ))" "$N_TEST"

# ---- Step 3：统计 ----
echo ""
echo "===== 合并切分完成 ====="
for split in train val test; do
    img_cnt=$(ls "${DST}/images/${split}/" | wc -l)
    lbl_cnt=$(ls "${DST}/labels/${split}/" | wc -l)
    echo "  ${split}: images=${img_cnt}, labels=${lbl_cnt}"
done
echo ""
echo "合并目录: ${DST}"
echo "请将统计数据填入 logs/EXP-08_mpid_transfer.md"
