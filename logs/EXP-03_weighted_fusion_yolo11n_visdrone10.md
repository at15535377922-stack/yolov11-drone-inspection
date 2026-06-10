# EXP-03：轻量加权跨尺度融合消融实验

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-03 |
| 实验类型 | 消融实验 — 轻量加权跨尺度融合（LWF） |
| 对照基线 | EXP-02（YOLOv11n + P2 分支） |
| 模型配置 | `ultralytics/cfg/models/11/yolo11-p2-lwf.yaml` |
| 训练脚本 | `scripts/train/run_visdrone10_exp03_lwf.sh` |
| 预训练权重 | `weights/pretrained/yolo11n.pt` |
| 数据集 | VisDrone2019-DET，10 类 |
| 数据配置 | `configs/datasets/visdrone10.yaml` |
| 输入尺寸 | 640 × 640 |
| Batch / Epochs | 16 / 100 |
| Device / Workers | GPU 0（RTX 4090）/ 8 |
| **核心变量** | **在 EXP-02 的 P2 结构基础上，用 LWFusion 替换关键融合节点的 Concat** |
| 实验日期 | （训练后填写） |
| 训练时长 | （训练后填写） |

---

## 设计方案（论文 §6.4）

### 问题定位

EXP-02 证明单独增加 P2 分支只能带来有限增益：整体 mAP50 从 0.320 提升到 0.335，但 bicycle 和 awning-tricycle 的提升很弱，同时带来了明显的计算与显存成本。一个直接原因是：EXP-02 在各尺度融合节点仍使用普通 `Concat`，默认不同尺度特征贡献相同，浅层噪声和深层语义被一并送入后续卷积，网络无法显式学习“当前任务更该依赖哪一层”。

### 改动内容

在 EXP-02 的 P2/P3/P4/P5 关键融合节点，将：

```text
Upsample/Downsample + Concat + C3k2
```

替换为：

```text
Upsample/Downsample + LWFusion + C3k2
```

其中 `LWFusion` 实现为：

$$
\hat{w}_i = \frac{\mathrm{ReLU}(w_i)}{\sum_j \mathrm{ReLU}(w_j) + \epsilon}
$$

$$
F_{out} = \mathrm{Conv}_{1\times1}\left(\sum_i \hat{w}_i X_i\right)
$$

模块特点：

- 每个输入分支只引入一个可学习标量权重，参数量极小；
- 先做归一化加权求和，再做 `1×1 Conv` 混合通道；
- 不改 Detect 头，不改 Backbone，不引入重型注意力或对齐卷积；
- 重点验证“仅靠轻量加权融合”是否能释放 P2 分支潜力。

### 接入节点

本次替换覆盖 6 个双路融合节点：

1. `P5↑ + F4`
2. `P4↑ + F3`
3. `P3↑ + F2`
4. `P2↓ + P3`
5. `P3↓ + P4`
6. `P4↓ + F5`

---

## 预期结果

| 指标 | EXP-02 | EXP-03 预期方向 | 依据 |
|---|---|---|---|
| mAP50（整体） | 0.335 | **↑ +0.01~+0.03** | 自适应抑制无效尺度噪声 |
| mAP50-95（整体） | 0.194 | **↑ 小幅提升** | 融合质量更稳定 |
| Recall | 0.348 | **↑ 小幅提升** | 小目标匹配更聚焦 |
| bicycle AP50 | 0.0967 | **↑ 明显优于 EXP-02** | P2/P3 权重应更偏向小目标有效尺度 |
| awning-tricycle AP50 | 0.0956 | **↑ 明显优于 EXP-02** | 同上 |
| 推理速度 | 12.7 ms | 持平或略慢 | 仅新增轻量加权和 1×1 混合 |
| 训练稳定性 | 存在 assigner OOM 回退 | **优于 EXP-02** | batch 已降到 16，降低显存压力 |

### 判定标准

- 若整体 mAP50 相比 EXP-02 再提升 ≥ 0.01，且 bicycle / awning-tricycle 至少有一类提升明显，则认为 LWFusion 有效。
- 若整体指标持平但速度代价更低或训练更稳定，也可认为该设计有保留价值。
- 若小目标类仍无改善，则说明仅做加权融合不足，后续需依赖注意力或损失函数继续补强。

---

## 实验结果（训练后填写）

### 总体指标（val split，best.pt）

| 指标 | EXP-03 (LWF) | EXP-02 (P2) | Delta |
|---|---|---|---|
| Precision | — | 0.466 | — |
| Recall | — | 0.348 | — |
| mAP50 | — | 0.335 | — |
| mAP50-95 | — | 0.194 | — |

### 各类别 mAP50（val split，best.pt）

| 类别 | EXP-03 (LWF) | EXP-02 (P2) | Delta |
|---|---|---|---|
| pedestrian | — | 0.391 | — |
| people | — | 0.301 | — |
| bicycle | — | 0.0967 | — |
| car | — | 0.777 | — |
| van | — | 0.361 | — |
| truck | — | 0.286 | — |
| tricycle | — | 0.200 | — |
| awning-tricycle | — | 0.0956 | — |
| bus | — | 0.457 | — |
| motor | — | 0.390 | — |
| **mAP50 均值** | — | **0.335** | — |

### 推理速度与模型规模

| 指标 | EXP-03 (LWF) | EXP-02 (P2) |
|---|---|---|
| 推理速度（ms/img，GPU） | — | 12.7 |
| 参数量（M） | — | 2.66 |
| FLOPs（G） | — | 10.2 |

---

## 主要观察（训练后填写）

- [ ] 小目标类是否较 EXP-02 进一步改善？
- [ ] LWFusion 是否在不明显增加开销的前提下提升整体指标？
- [ ] batch=16 后是否仍有 assigner OOM 回退？
- [ ] 是否为后续 EXP-04 注意力模块提供更好的融合底座？

---

## 结论与后续规划（训练后填写）

| EXP | 内容 | 状态 |
|---|---|---|
| EXP-01 | YOLOv11n 基线（P3/P4/P5） | ✅ 完成 |
| EXP-02 | P2 分支消融 | ✅ 完成 |
| **EXP-03** | **轻量加权跨尺度融合** | 🔄 进行中 |
| EXP-04 | 轻量通道-空间注意力 | ⬜ 待开始 |
| EXP-05 | 小目标损失函数（IoU+NWD） | ⬜ 待开始 |
| EXP-06 | Ghost/Rep 轻量化模块 | ⬜ 待开始 |
| EXP-07 | ASE-YOLOv11 全量模型 | ⬜ 待开始 |
| EXP-08 | MPID 迁移验证 | ⬜ 待数据 |
