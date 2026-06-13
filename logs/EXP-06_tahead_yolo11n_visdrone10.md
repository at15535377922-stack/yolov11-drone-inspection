# EXP-06：轻量任务对齐检测头（TADetect）消融实验

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-06 |
| 实验类型 | 消融实验 — 轻量任务对齐检测头（TADetect，P2/P3 双向 cls/reg 交互） |
| 对照基线 | EXP-05（P2 + NWD 损失，mAP50=0.336） |
| 模型配置 | `ultralytics/cfg/models/11/yolo11-p2-tahead.yaml` |
| 训练脚本 | `scripts/train/run_visdrone10_exp06_tahead.sh` |
| 预训练权重 | `weights/pretrained/yolo11n.pt` |
| 数据集 | VisDrone2019-DET，10 类 |
| 数据配置 | `configs/datasets/visdrone10.yaml` |
| 输入尺寸 | 640 × 640 |
| Batch / Epochs | 16 / 100 |
| Device / Workers | GPU 0（RTX 4090）/ 8 |
| **核心变量** | **将 P2/P3 检测头由 `Detect` 换为 `TADetect`（`ta_layers=2`），NWD 损失保持不变（同 EXP-05）** |
| 实验日期 | （训练完成后回填） |
| 训练时长 | （训练完成后回填） |

> **选择 EXP-05 作为对照**：EXP-05 已验证 NWD 损失有效（awning-tricycle +16%），EXP-06 在相同损失函数下叠加 TADetect，单独验证检测头对分类-定位一致性的贡献，保持单变量原则。

---

## 设计方案（论文 §6.6）

### 问题定位

EXP-05 表明，NWD 损失改善了极小目标（awning-tricycle/tricycle）的定位平滑性，但 mAP50 整体仅提升 0.001，说明还存在另一个制约因素：**分类置信度与定位质量的不一致性**。

在 VisDrone 低空巡检场景中：
- 极小目标的边界框经常有轻微位移，但分类分支无法感知这种位移信息；
- 回归分支预测出精准框后，分类分支可能给出错误的低置信度，导致 NMS 抑制正确检测结果；
- P2/P3 层（处理 < 32px 目标）特别容易出现此问题，P4/P5 相对影响较小。

### 改动内容

**新建模块**：`TADetect`（`ultralytics/nn/modules/head.py`）

**核心思想**：在 `Detect.forward_head` 里，对前 `ta_layers`（=2）个尺度（P2/P3），在中间特征层做双向任务交互：

$$F_\text{cls}' = F_\text{cls} \otimes S(F_\text{reg})$$

$$F_\text{reg}' = F_\text{reg} \otimes C(F_\text{cls})$$

- $S(\cdot)$：空间注意力，`Conv(c2→1, 3×3) + Sigmoid`，输出 `(B, 1, H, W)` 权重图；
- $C(\cdot)$：通道注意力，`AvgPool → Conv(c3→c2, 1×1) + Sigmoid`，输出 `(B, c2, 1, 1)` 权重向量；
- `c2`：回归分支中间通道（`max(16, ch//4, reg_max*4)`）；
- `c3`：分类分支中间通道（`max(ch[0], min(nc, 100))`）。

**新建文件**：`ultralytics/cfg/models/11/yolo11-p2-tahead.yaml`

与 `yolo11-p2.yaml` 结构完全相同，仅最后一行：

```yaml
# 原（EXP-02/05）:
- [[19, 22, 25, 28], 1, Detect,   [nc]]
# 改（EXP-06）:
- [[19, 22, 25, 28], 1, TADetect, [nc, 16, 2]]
```

**P4/P5 不变**：`ta_layers=2` 表示只有 P2（索引 0）和 P3（索引 1）做交互，P4/P5 走原始 `Detect` 路径，避免大目标头引入不必要的参数。

### 参数量估算

TADetect 在每个 TA 尺度新增：
- 空间注意力：`c2 × 1 × 3 × 3 = 9c2` 参数（n 尺度 c2≈16，约 144 参数/尺度）
- 通道注意力：`c3 × c2 × 1 × 1 = c2 × c3` 参数（n 尺度约 `16 × 16 = 256` 参数/尺度）
- 两层合计约 800 参数，**参数量几乎不变**（基准 2.66M）

---

## 预期结果

| 指标 | EXP-05 (NWD) | EXP-06 预期方向 | 依据 |
|---|---|---|---|
| mAP50（整体） | 0.336 | **↑ +0.005~+0.015** | cls/reg 一致性提升，NMS 保留更多正确框 |
| mAP50-95（整体） | 0.193 | **↑ 小幅提升** | 精确框（IOU≥0.75）更多 |
| Precision | 0.462 | **↑ 明显** | 高质量候选框被正确赋予高分类置信度 |
| Recall | 0.354 | **↑ 小幅** | 位置正确但置信度低的框被激活 |
| bicycle AP50 | 0.0914 | **↑ 明显** | P2 层交互直接改善极小目标 |
| awning-tricycle AP50 | 0.108 | **↑ 明显** | 同上 |
| 参数量 | 2.66M | **≈不变（+<1K）** | TA 模块极轻量 |
| 推理速度 | 0.4ms | **≈不变** | 额外 Conv 操作可忽略不计 |

### 判定标准

- 若 mAP50 ≥ 0.340，或 bicycle/awning-tricycle 任一类 AP50 改善 ≥ 0.005，则认为 TADetect 有效。
- 若结果与 EXP-05 持平但无下降，可保留 TADetect 进入 EXP-07 全量集成。
- 若有明显下降（mAP50 < 0.332），排查 YAML args 解析或中间层 shape 不匹配问题。

---

## 实验结果

> ⚠️ **占位区域** — 训练完成后回填真实数据，禁止填入估计值或目标值。

### 总体指标（val split，best.pt）

| 指标 | EXP-06 (TADetect) | EXP-05 (NWD) | Delta |
|---|---|---|---|
| mAP50 | — | 0.336 | — |
| mAP50-95 | — | 0.193 | — |
| Precision | — | 0.462 | — |
| Recall | — | 0.354 | — |

### 分类 AP50（val split）

| 类别 | EXP-06 | EXP-05 | Delta |
|---|---|---|---|
| pedestrian | — | 0.390 | — |
| people | — | 0.302 | — |
| bicycle | — | 0.0914 | — |
| car | — | 0.776 | — |
| van | — | 0.365 | — |
| truck | — | 0.291 | — |
| tricycle | — | 0.199 | — |
| awning-tricycle | — | 0.108 | — |
| bus | — | 0.451 | — |
| motor | — | 0.390 | — |

### 训练输出路径

```
runs/detect/visdrone10_exp06_tahead/
├── results.csv
├── weights/best.pt
└── weights/last.pt
```

---

## 结论（训练完成后填写）

- **整体结论**：（回填）
- **分类-定位一致性改善**：（回填）
- **与 EXP-05 的增量**：（回填）
- **是否进入最终集成（EXP-07）**：（回填）
- **下一步建议**：（回填；EXP-07 为全量集成：P2 + TADetect + NWD 损失，验证各模块叠加效果）

---

## 超参快照

```yaml
model:       yolo11-p2-tahead.yaml
pretrained:  weights/pretrained/yolo11n.pt
data:        configs/datasets/visdrone10.yaml
epochs:      100
imgsz:       640
batch:       16
device:      0
workers:     8
# NWD 损失（同 EXP-05）
nwd_weight:     0.4
nwd_constant:   12.8
nwd_small_rho:  0.5
# TADetect 参数（写在 YAML args 里，不通过 CLI 传入）
ta_layers: 2   # P2 + P3
```
