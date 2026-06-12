# EXP-05：面向小目标的边界框损失优化（CIoU + NWD）消融实验

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-05 |
| 实验类型 | 消融实验 — 组合定位损失（CIoU + NWD + 小目标权重） |
| 对照基线 | EXP-02（YOLOv11n + P2 分支，mAP50=0.335） |
| 模型配置 | `ultralytics/cfg/models/11/yolo11-p2.yaml`（与 EXP-02 相同，结构零修改） |
| 训练脚本 | `scripts/train/run_visdrone10_exp05_nwd.sh` |
| 预训练权重 | `weights/pretrained/yolo11n.pt` |
| 数据集 | VisDrone2019-DET，10 类 |
| 数据配置 | `configs/datasets/visdrone10.yaml` |
| 输入尺寸 | 640 × 640 |
| Batch / Epochs | 16 / 100 |
| Device / Workers | GPU 0（RTX 4090）/ 8 |
| **核心变量** | **仅修改定位损失函数（NWD 权重 β=0.4，C=12.8，小目标权重 ρ=0.5），网络结构与 EXP-02 完全相同** |
| 实验日期 | （训练完成后回填） |
| 训练时长 | （训练完成后回填） |

> **选择 EXP-02 而非 EXP-04 作为对照**：EXP-03/04 的 LWFusion 与 CBAM 对结果无正向贡献（与 EXP-02 持平或略退），保持单变量对照，将 EXP-05 的收益归因于损失函数设计。

---

## 设计方案（论文 §6.7）

### 问题定位

EXP-02~04 的消融表明，P2 分支引入是当前最有效的单项改进（mAP50 从 0.320 提升至 0.335），而轻量加权融合与通道-空间注意力的边际收益已基本耗尽。从结果的角度看：

- `bicycle` AP50 在 EXP-02~04 均在 0.085~0.090 附近，没有明显改善；
- `awning-tricycle` AP50 同样在 0.090~0.095 徘徊；
- 说明纯结构性改进（更多感受野层次、跨尺度融合权重、显著性增强）对这类 **极小目标的定位精度** 影响有限。

VisDrone 中的 `bicycle`、`awning-tricycle`、`van` 等类别目标尺寸常低于 16×16 像素。使用标准 CIoU 损失时，预测框相差 2~3 个像素即可导致 IoU 接近 0，训练信号极其稀疏，梯度不稳定。NWD 通过高斯分布 Wasserstein 距离度量两框的相似程度，在极小目标上提供平滑、稳定的训练信号。

### 改动内容

**改动文件**：`ultralytics/utils/loss.py`

在 `BboxLoss` 中加入可选的 NWD 项与小目标权重，通过训练命令行参数激活：

```
nwd_weight=0.4    # NWD 在定位损失中的比重 β
nwd_constant=12.8  # NWD 归一化常数 C（imgsz × 0.02）
nwd_small_rho=0.5  # 小目标额外上权重的缩放系数 ρ
```

**定位损失公式**（论文 §6.7）：

$$
d_N(B_p, B_g) = \sqrt{(x_p - x_g)^2 + (y_p - y_g)^2 + \frac{(w_p - w_g)^2 + (h_p - h_g)^2}{4}}
$$

$$
\text{NWD}(B_p, B_g) = \exp\left(-\frac{d_N(B_p, B_g)}{C}\right)
$$

$$
\omega_s = 1 + \rho \cdot \exp\left(-\frac{w_g h_g}{\overline{w_g h_g}}\right)
$$

$$
L_\text{box} = \omega_s \left[\alpha(1 - \text{CIoU}) + \beta(1 - \text{NWD})\right]
$$

其中 $\alpha = 1 - \beta = 0.6$，$\rho = 0.5$，$C = 12.8$（$640 \times 0.02$）。

**EXP-05 本次不引入长宽比差异项** $\Delta AR$：先验证 IoU + NWD + 小目标权重的基础组合，若效果显著再在 EXP-06 补充 $\Delta AR$ 项，保持消融粒度一致。

### 实现逻辑摘要

```python
# BboxLoss.forward 关键路径（nwd_weight > 0 时激活）
pb_px = pb * stride   # 预测框转回像素空间（N, 4）
tb_px = tb * stride   # 真实框转回像素空间（N, 4）
nwd   = NWD(pb_px, tb_px, C=12.8).unsqueeze(-1)     # (N, 1)

combined = 0.6 * (1 - CIoU) + 0.4 * (1 - nwd)
sw       = 1 + 0.5 * exp(-area / mean_area)          # 小目标权重 (N, 1)
loss_iou = (combined * sw * weight).sum() / target_scores_sum
```

### 超参选择依据

| 超参 | 值 | 依据 |
|---|---|---|
| `nwd_weight` (β) | 0.4 | 文献常用范围 [0.3, 0.5]；给 CIoU 保留主导地位（α=0.6） |
| `nwd_constant` (C) | 12.8 | imgsz=640 时，$C = 640 \times 0.02 = 12.8$，与 NWD 原论文推荐一致 |
| `nwd_small_rho` (ρ) | 0.5 | 中等上权重，避免过度放大噪声标注 |

---

## 预期结果

| 指标 | EXP-02 (P2 baseline) | EXP-05 预期方向 | 依据 |
|---|---|---|---|
| mAP50（整体） | 0.335 | **↑ +0.005~+0.015** | NWD 改善小目标框定位 |
| mAP50-95（整体） | 0.194 | **↑ 小幅提升** | 精确框 AP (0.75+) 最可能改善 |
| Recall | 0.352 | **↑ 小幅提升** | 更稳定梯度，小目标漏检减少 |
| bicycle AP50 | ≈0.088 | **↑ 明显** | 极小目标直接受益于 NWD 平滑 |
| awning-tricycle AP50 | ≈0.093 | **↑ 明显** | 同上 |
| pedestrian AP50 | ≈0.392 | **↑ 中等** | 框精度改善 |
| 推理速度 | 约 1.2 ms/img | **不变** | 仅改动训练损失，推理路径无修改 |
| 参数量 / GFLOPs | 2.86M / 10.6G | **不变** | 结构与 EXP-02 完全相同 |
| 训练稳定性 | 稳定 | 持平 | 损失函数改动温和，无新模块 |

### 判定标准

- 若 mAP50 相比 EXP-02 **提升 ≥ 0.005**，且 `bicycle` / `awning-tricycle` 至少有一类 AP50 改善 ≥ 0.005，则认为 NWD 组合损失有效。
- 若整体 mAP50 持平但小目标精类（bicycle/awning-tricycle）有稳定改善，仍保留该损失作为后续全模块集成的一部分。
- 若无改善，考虑调整 β 值（0.2 或 0.6）或增加 $\Delta AR$ 项重做 EXP-05b。

---

## 实验结果

> ⚠️ **占位区域** — 训练完成后回填真实数据，禁止填入估计值或目标值。

### 总体指标（val split，best.pt）

| 指标 | EXP-05 (NWD Loss) | EXP-02 (P2 baseline) | Delta |
|---|---|---|---|
| mAP50 | — | 0.335 | — |
| mAP50-95 | — | 0.194 | — |
| Precision | — | 0.472 | — |
| Recall | — | 0.352 | — |

### 分类 AP50（val split）

| 类别 | EXP-05 | EXP-02 | Delta |
|---|---|---|---|
| pedestrian | — | 0.392 | — |
| people | — | 0.232 | — |
| bicycle | — | 0.088 | — |
| car | — | 0.586 | — |
| van | — | 0.342 | — |
| truck | — | 0.285 | — |
| tricycle | — | 0.168 | — |
| awning-tricycle | — | 0.093 | — |
| bus | — | 0.469 | — |
| motor | — | 0.219 | — |

> 注：EXP-02 的分类 AP50 以实际 `val` 输出为准，数值为预估参考，回填时以真实日志为准。

### 训练曲线（回填路径）

```
runs/detect/visdrone10_exp05_nwd/
├── results.csv          # 各 epoch 损失 / mAP 曲线
├── weights/best.pt
└── weights/last.pt
```

---

## 结论（训练完成后填写）

- **整体结论**：（回填）
- **小目标类改善**：（回填）
- **与 EXP-02 的增量**：（回填）
- **是否进入最终集成**：（回填）
- **下一步建议**：（回填；若 EXP-05 有效，考虑在 EXP-06 叠加 $\Delta AR$ 项；若无效，考虑 Task-Aligned Head / 训练策略方向）

---

## 超参快照

```yaml
model:       yolo11-p2.yaml
pretrained:  weights/pretrained/yolo11n.pt
data:        configs/datasets/visdrone10.yaml
epochs:      100
imgsz:       640
batch:       16
device:      0
workers:     8
# NWD 损失激活参数
nwd_weight:     0.4
nwd_constant:   12.8
nwd_small_rho:  0.5
```
