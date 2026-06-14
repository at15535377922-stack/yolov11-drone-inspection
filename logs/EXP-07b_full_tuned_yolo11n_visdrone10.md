# EXP-07b：ASE-YOLOv11 全量集成调参实验

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-07b |
| 实验类型 | 全量集成调参 — 针对 EXP-07 tricycle/awning-tricycle 退步的定向优化 |
| 对照基线 | EXP-07（全量集成，mAP50=0.337，tricycle=0.188，awning-tricycle=0.0899） |
| 参考实验 | EXP-06（TADetect ta_layers=2，tricycle=0.204，awning-tricycle=0.104） |
| 模型配置 | `ultralytics/cfg/models/11/yolo11-p2-full-b.yaml` |
| 训练脚本 | `scripts/train/run_visdrone10_exp07b_full.sh` |
| 预训练权重 | `weights/pretrained/yolo11n.pt` |
| 数据集 | VisDrone2019-DET，10 类 |
| 数据配置 | `configs/datasets/visdrone10.yaml` |
| 输入尺寸 | 640 × 640 |
| Batch / Epochs | 16 / 100 |
| Device / Workers | GPU 0（RTX 4090）/ 8 |
| **变量 1** | **ta_layers: 2 → 1**（TADetect 仅在 P2 做双向交互，P3/P4/P5 走标准路径） |
| **变量 2** | **nwd_weight: 0.4 → 0.5**（加强 NWD 定位损失对极小目标的惩罚权重） |
| 实验日期 | （训练完成后回填） |
| 训练时长 | （训练完成后回填） |

---

## 设计方案

### EXP-07 退步根因分析

EXP-07 对比 EXP-06，tricycle 下降 −0.016，awning-tricycle 下降 −0.014，而这两个类在 EXP-05 NWD 消融时曾是增益最大的类别。分析可能原因：

1. **P3 TA 交互引入融合噪声**：ta_layers=2 时，P3 层也做 cls/reg 双向交互。P3（1/8 stride，处理 16-32px 目标）恰好是 tricycle 和 awning-tricycle 的主检测尺度。LWFusion 在 P3 融合节点（层 21）本身已引入加权求和，TADetect 再在 P3 做注意力重新加权，两层加权叠加可能导致这些密集小目标的特征分布偏移。
2. **NWD 权重不足**：nwd_weight=0.4 时，CIoU（0.6）仍占主导。对于极小目标（tricycle 平均约 18px，awning-tricycle 约 14px），CIoU 在 IOU 接近 0 时梯度消失，NWD 的高斯分布近似能提供更稳定梯度，适当上调权重可改善。

### 变量设置逻辑

| 变量 | EXP-07 | EXP-07b | 预期效果 |
|---|---|---|---|
| ta_layers | 2（P2+P3） | **1（仅 P2）** | 消除 P3 TA 噪声，保留 P2 最细粒度对齐 |
| nwd_weight | 0.4 | **0.5** | 增强极小目标定位平滑性，恢复 tricycle/awning-tricycle |
| 其余超参 | — | 不变 | 保持单实验双变量，两变量方向一致（均针对小目标） |

> **双变量说明**：ta_layers 和 nwd_weight 均直接针对同一问题（极小目标退步），方向一致，共同构成一次"小目标友好性"调优，不违反消融单变量原则。若 EXP-07b 成功，后续如需细分贡献，可补充 EXP-07c/d 单独验证。

---

## 预期结果

### 总体指标

| 指标 | EXP-07b 预期 | EXP-07 | EXP-06 | 依据 |
|---|---|---|---|---|
| mAP50 | **≥ 0.338** | 0.337 | 0.337 | 整体持平或微增 |
| mAP50-95 | **≥ 0.192** | 0.191 | 0.192 | NWD 提升精确定位比例 |
| Recall | **≥ 0.360** | 0.361 | 0.360 | 持平 |

### 重点关注类别

| 类别 | EXP-07b 预期 | EXP-07 | EXP-06 | 目标 |
|---|---|---|---|---|
| tricycle | **≥ 0.200** | 0.188 | 0.204 | 恢复至 EXP-06 水平 |
| awning-tricycle | **≥ 0.100** | 0.0899 | 0.104 | 恢复至 EXP-06 水平 |
| bicycle | **≥ 0.100** | 0.104 | 0.0962 | 维持 EXP-07 增益 |
| pedestrian | **≥ 0.400** | 0.400 | 0.404 | 维持水平 |

### 判定标准

- ✅ **成功**：mAP50 ≥ 0.338 **且** tricycle ≥ 0.200 **且** awning-tricycle ≥ 0.100
- ⚠️ **部分成功**：mAP50 ≥ 0.337 且小目标类别至少一项恢复
- ❌ **失败**：mAP50 < 0.335 或小目标类别未改善

---

## 实验结果

> ⚠️ **占位区域** — 训练完成后回填真实数据，禁止填入估计值或目标值。

### 总体指标（val split，best.pt）

| 指标 | EXP-07b (Tuned) | EXP-07 (Full) | EXP-06 (TADetect) | Δ vs EXP-07 |
|---|---|---|---|---|
| mAP50 | — | 0.337 | 0.337 | — |
| mAP50-95 | — | 0.191 | 0.192 | — |
| Precision | — | 0.458 | 0.455 | — |
| Recall | — | 0.361 | 0.360 | — |

### 分类 AP50（val split，重点类别）

| 类别 | EXP-07b | EXP-07 | EXP-06 | Δ vs EXP-07 |
|---|---|---|---|---|
| pedestrian | — | 0.400 | 0.404 | — |
| people | — | 0.310 | 0.309 | — |
| bicycle | — | 0.104 | 0.0962 | — |
| car | — | 0.780 | 0.781 | — |
| van | — | 0.365 | 0.357 | — |
| truck | — | 0.275 | 0.269 | — |
| tricycle | — | 0.188 | 0.204 | — |
| awning-tricycle | — | 0.0899 | 0.104 | — |
| bus | — | 0.457 | 0.443 | — |
| motor | — | 0.405 | 0.401 | — |

### 训练输出路径

```
runs/detect/visdrone10_exp07b_full/
├── results.csv
├── weights/best.pt
└── weights/last.pt
```

---

## 结论（训练完成后填写）

- **整体结论**：（回填）
- **tricycle / awning-tricycle 是否恢复**：（回填）
- **ta_layers=1 vs ta_layers=2 的效果对比**：（回填）
- **nwd_weight=0.5 的影响**：（回填）
- **是否作为最终模型**：（回填；若 mAP50≥0.338 且小目标恢复，则 EXP-07b 为最终 ASE-YOLOv11；否则考虑接受 EXP-07/EXP-06 结果写论文）

---

## 超参快照

```yaml
model:       yolo11-p2-full-b.yaml
pretrained:  weights/pretrained/yolo11n.pt
data:        configs/datasets/visdrone10.yaml
epochs:      100
imgsz:       640
batch:       16
device:      0
workers:     8
# NWD 损失（调整 nwd_weight）
nwd_weight:     0.5    # ← 0.4 → 0.5
nwd_constant:   12.8
nwd_small_rho:  0.5
# TADetect 参数（写在 YAML args 里）
ta_layers: 1   # ← 2 → 1，仅 P2 做双向交互
```
