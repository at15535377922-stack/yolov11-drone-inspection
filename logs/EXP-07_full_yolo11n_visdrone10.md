# EXP-07：ASE-YOLOv11 全量集成实验

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-07 |
| 实验类型 | 全量集成验证 — 所有消融模块叠加 |
| 对照基线 | EXP-01（YOLOv11n 基线，mAP50=0.320） |
| 消融参考 | EXP-06（TADetect，mAP50=0.337，当前最优单模块叠加结果） |
| 模型配置 | `ultralytics/cfg/models/11/yolo11-p2-full.yaml` |
| 训练脚本 | `scripts/train/run_visdrone10_exp07_full.sh` |
| 预训练权重 | `weights/pretrained/yolo11n.pt` |
| 数据集 | VisDrone2019-DET，10 类 |
| 数据配置 | `configs/datasets/visdrone10.yaml` |
| 输入尺寸 | 640 × 640 |
| Batch / Epochs | 16 / 100 |
| Device / Workers | GPU 0（RTX 4090）/ 8 |
| **核心变量** | **P2 分支 + LWFusion（×6 节点） + CBAM（×4 尺度） + TADetect（ta_layers=2） + NWD 损失（alpha=0.6, beta=0.4）** |
| 实验日期 | 2026-06-14 |
| 训练时长 | 2.817 h（100 epochs） |

---

## 全量集成方案（论文 §6.7 / 主模型）

### 模块叠加逻辑

| 模块 | 来源实验 | mAP50 增益（单模块，相对 EXP-01） | 集成后预期协同效应 |
|---|---|---|---|
| P2 分支 | EXP-02 | +0.015（0.320→0.335） | 极小目标（<16px）的检测分辨率保证 |
| LWFusion | EXP-03 | ≈0（0.335→0.332，含噪声） | 多尺度特征加权融合，减少跨尺度噪声传递 |
| CBAM | EXP-04 | ≈0（持平 EXP-02） | 通道×空间双重注意力，压制背景噪声 |
| NWD 损失 | EXP-05 | +0.001（0.335→0.336） | 极小目标定位平滑性，提升 mAP50-95 |
| TADetect | EXP-06 | +0.001（0.336→0.337） | cls/reg 任务对齐，Recall 提升 +0.006 |

> **集成假设**：LWFusion + CBAM 单独消融时未体现正向收益，但在 P2 + NWD + TADetect 完整结构下，多尺度融合精度和注意力压噪有望产生**协同增益**，使总体效果超过各模块简单加和。

### YAML 关键变化（相对 EXP-04 `yolo11-p2-lwf-cbam.yaml`）

```yaml
# EXP-04（最后一行）:
  - [[29, 30, 31, 32], 1, Detect,   [nc]]

# EXP-07（最后一行）:
  - [[29, 30, 31, 32], 1, TADetect, [nc, 16, 2]]   # ta_layers=2 → P2+P3 双向交互
```

其余层定义（Backbone / LWFusion Neck / CBAM）与 EXP-04 完全一致。

### 超参快照

```yaml
model:       yolo11-p2-full.yaml
pretrained:  weights/pretrained/yolo11n.pt
data:        configs/datasets/visdrone10.yaml
epochs:      100
imgsz:       640
batch:       16
device:      0
workers:     8
# NWD 损失（同 EXP-05/06）
nwd_weight:     0.4
nwd_constant:   12.8
nwd_small_rho:  0.5
# TADetect（写在 YAML args 里）
ta_layers: 2   # P2 + P3 做双向 cls/reg 交互
```

---

## 预期结果

### 总体指标

| 指标 | EXP-01 (基线) | EXP-06 (当前最优) | EXP-07 预期 | 依据 |
|---|---|---|---|---|
| mAP50 | 0.320 | 0.337 | **≥ 0.345** | 协同增益；各模块正向类别覆盖不同 |
| mAP50-95 | 0.184 | 0.192 | **≥ 0.197** | NWD+TADetect 提升精确定位框比例 |
| Precision | 0.449 | 0.455 | **≥ 0.460** | CBAM 压噪 + TADetect 对齐 |
| Recall | 0.341 | 0.360 | **≥ 0.365** | TADetect 激活低置信度真正样本 |

### 重点关注类别

| 类别 | EXP-01 | EXP-06 | EXP-07 预期 | 关注原因 |
|---|---|---|---|---|
| bicycle | 0.0801 | 0.0962 | **≥ 0.100** | P2+CBAM+TADetect 联合 |
| awning-tricycle | 0.0876 | 0.104 | **≥ 0.110** | 极小密集目标，NWD+TADetect 联合 |
| tricycle | 0.183 | 0.204 | **≥ 0.210** | 小目标，同上 |
| pedestrian | 0.373 | 0.404 | **≥ 0.410** | 中型目标，TADetect 主导 |
| motor | 0.368 | 0.401 | **≥ 0.405** | 同上 |

### 判定标准

- ✅ **成功**：mAP50 ≥ 0.345 且 bicycle / awning-tricycle / tricycle 全部改善（相对 EXP-06）
- ⚠️ **部分成功**：mAP50 ∈ [0.340, 0.345)，或只有部分小目标类别改善
- ❌ **失败**：mAP50 < 0.337（不如 EXP-06 单模块叠加结果）

---

## 实验结果

> ⚠️ **占位区域** — 训练完成后回填真实数据，禁止填入估计值或目标值。

### 总体指标（val split，best.pt）

| 指标 | EXP-07 (Full) | EXP-06 (TADetect) | EXP-01 (Baseline) | Δ vs EXP-06 | Δ vs EXP-01 |
|---|---|---|---|---|---|
| mAP50 | **0.337** | 0.337 | 0.320 | 0.000 | **+0.017** |
| mAP50-95 | **0.191** | 0.192 | 0.184 | −0.001 | **+0.007** |
| Precision | **0.458** | 0.455 | 0.449 | **+0.003** | **+0.009** |
| Recall | **0.361** | 0.360 | 0.341 | **+0.001** | **+0.020** |

### 分类 AP50（val split）

| 类别 | EXP-07 | EXP-06 | EXP-01 | Δ vs EXP-06 |
|---|---|---|---|---|
| pedestrian | **0.400** | 0.404 | 0.373 | −0.004 |
| people | **0.310** | 0.309 | 0.291 | **+0.001** |
| bicycle | **0.104** | 0.0962 | 0.0801 | **+0.008** |
| car | **0.780** | 0.781 | 0.759 | −0.001 |
| van | **0.365** | 0.357 | 0.361 | **+0.008** |
| truck | **0.275** | 0.269 | 0.285 | **+0.006** |
| tricycle | 0.188 | 0.204 | 0.183 | −0.016 |
| awning-tricycle | 0.0899 | 0.104 | 0.0876 | −0.014 |
| bus | **0.457** | 0.443 | 0.449 | **+0.014** |
| motor | **0.405** | 0.401 | 0.368 | **+0.004** |

### 模型复杂度

| 项目 | EXP-07 (Full) | EXP-01 (Baseline) |
|---|---|---|
| 参数量 | **2.862M** | 2.59M |
| GFLOPs | **10.6** | 6.3 |
| 推理速度（ms/img，inference） | **1.5** | — |
| 训练时长 | **2.817 h** | 1.947 h |

### 训练输出路径

```
runs/detect/visdrone10_exp07_full/
├── results.csv
├── weights/best.pt
└── weights/last.pt
```

---

## 结论（训练完成后填写）

- **整体结论**：全量集成模型 mAP50=0.337，与 EXP-06（TADetect 单模块）持平，**未实现预期的协同增益（目标 ≥ 0.345）**。相对 EXP-01 基线总增益 +0.017，验证了各模块的累积有效性。
- **协同增益验证**：LWFusion 和 CBAM 在全量集成下**仍未体现额外协同增益**——全量结果与 EXP-06（仅叠加到 TADetect）持平，说明 LWFusion 和 CBAM 对本数据集/规模下的整体 mAP50 贡献有限，主要增益来源是 P2 分支（+0.015）、NWD 损失（+0.001）和 TADetect（+0.001）。
- **与 EXP-06 的增量**：mAP50 持平（0.000），bicycle（+0.008）、van（+0.008）、bus（+0.014）、truck（+0.006）、motor（+0.004）正向，但 tricycle（−0.016）和 awning-tricycle（−0.014）出现退步，可能因 LWFusion 加权融合在极密集小目标场景引入了特征融合噪声。
- **与 EXP-01 的总增益**：mAP50 +0.017（0.320→0.337），Recall +0.020，bicycle +0.024，awning-tricycle +0.002，motor +0.037，pedestrian +0.027，整体小目标检测能力显著提升。
- **模型是否满足论文核心指标**：❌ mAP50=0.337 < 0.345（目标未达成）。需进一步调优或分析退步类别（tricycle/awning-tricycle）的原因。
- **下一步建议**：建议进行 **EXP-07b**（增加训练轮数至 150 epochs，或调整 LWFusion 融合策略），或直接分析 tricycle/awning-tricycle 退步原因（可能需要调整 nwd_weight 或 ta_layers=1 仅保留 P2 交互），在得到 mAP50 ≥ 0.340 的稳定结果后再写入论文。
