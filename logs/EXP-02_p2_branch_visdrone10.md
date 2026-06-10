# EXP-02：P2 小目标检测分支消融实验

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-02 |
| 实验类型 | 消融实验 — P2 小目标检测分支 |
| 对照基线 | EXP-01（YOLOv11n 原始三检测头） |
| 模型配置 | `ultralytics/cfg/models/11/yolo11-p2.yaml` |
| 训练脚本 | `scripts/train/run_visdrone10_exp02_p2.sh` |
| 预训练权重 | `weights/pretrained/yolo11n.pt`（与 EXP-01 相同） |
| 数据集 | VisDrone2019-DET，10 类（与 EXP-01 相同） |
| 数据配置 | `configs/datasets/visdrone10.yaml` |
| 输入尺寸 | 640 × 640（与 EXP-01 相同） |
| Batch / Epochs | 32 / 100（与 EXP-01 相同） |
| Device / Workers | GPU 0（RTX 4090）/ 8（与 EXP-01 相同） |
| **唯一变量** | **Head 结构：三检测头 → 四检测头（新增 P2/4 分支）** |
| 实验日期 | （训练后填写） |
| 训练时长 | （训练后填写） |

---

## 设计方案（论文 §6.3）

### 问题定位

YOLOv11 原始架构输出 P3/P4/P5 三个检测头，最高分辨率特征图为 P3（stride=8）。对于输入 640×640 的图像，P3 特征图尺寸为 80×80。VisDrone 数据集中大量目标（bicycle、awning-tricycle 等）面积极小，在 P3 上可能只占 1~2 个 grid cell，边界和纹理信息在下采样过程中大量丢失，导致漏检率高。

**EXP-01 佐证**：

- bicycle mAP50 = **0.083**（10 类最低）
- awning-tricycle mAP50 = **0.091**（10 类第二低）
- 整体 Recall = 0.344，明显低于 Precision = 0.434，说明漏检是主要瓶颈

### 改动内容

在 Head 增加 **P2 分支（stride=4）**，具体结构（以 `n` scale 为例，实际通道数 = 配置通道 × width=0.25）：

```
Top-down FPN:
  F5(stride=32) → Upsample → Concat(F4) → C3k2 → P4_neck
  P4_neck       → Upsample → Concat(F3) → C3k2 → P3_neck
  P3_neck       → Upsample → Concat(F2) → C3k2 → P2_feat   ← 新增

Bottom-up PAN:
  P2_feat → Conv(s=2) → Concat(P3_neck) → C3k2 → P3_out    ← 新增，双向融合
  P3_out  → Conv(s=2) → Concat(P4_neck) → C3k2 → P4_out
  P4_out  → Conv(s=2) → Concat(F5)      → C3k2 → P5_out

Detect([P2_feat, P3_out, P4_out, P5_out])  ← 4 个检测头
```

**关键设计决策**：

1. **双向融合**：P2 不仅从 P3 获得自上而下的语义，还通过下采样路径将浅层细节反向传递给 P3，与论文 §6.3 "P2 进入后续双向融合网络" 一致。
2. **不改主干**：Backbone 层 0–10 完全不动，P2 浅层特征直接来自主干层 2（C3k2 输出，stride=4），无需修改任何 Python 代码，只修改 YAML 配置。
3. **控制通道**：P2 分支通道数设为 128（`n` scale 缩放后实际 32），比 P3（256）更轻量，避免过多计算负担。
4. **单一变量**：训练超参、数据、预训练权重全部与 EXP-01 相同，仅 Head 拓扑不同。

### 与 YOLO26-P2 的异同

| 对比项 | YOLO26-P2（官方） | 本方案（EXP-02） |
|---|---|---|
| P2 连接方式 | 单向 FPN（上采样到 P2，无回路） | 双向：P3→P2（FPN）+ P2→P3（PAN 下采样回流） |
| 主干修改 | SPPF 参数变化（YOLO26 特有） | **不改主干**，直接对接 yolo11.yaml 主干 |
| 模块类型 | YOLO26 专属 C3k2 参数 | 标准 YOLOv11 C3k2 |
| 代码修改量 | 需更换 backbone | **仅新增一个 YAML 文件** |

---

## 预期结果

### 预期指标方向

| 指标 | EXP-01 基线 | EXP-02 预期方向 | 依据 |
|---|---|---|---|
| mAP50（整体） | 0.320 | **↑ +0.02~+0.05** | P2 提升小目标召回 |
| mAP50-95（整体） | 0.181 | **↑ 小幅提升** | 精细定位略有改善 |
| Recall | 0.344 | **↑ 明显** | P2 减少漏检 |
| Precision | 0.434 | 持平或微降 | 新 anchor 可能引入少量误报 |
| bicycle AP50 | 0.083 | **↑ 较大幅度** | 极小目标最受益 |
| awning-tricycle AP50 | 0.091 | **↑ 较大幅度** | 同上 |
| pedestrian AP50 | 0.343 | **↑ 中等幅度** | 小目标行人也受益 |
| car AP50 | 0.748 | 基本持平 | 中大目标改善有限 |
| 推理速度（ms/img） | ~3.2 ms | **↑ 略慢** | P2 特征图 160×160，计算量增加 |
| 参数量（MB） | ~2.6M | **↑ 约增加 0.2~0.5M** | P2 分支新增卷积层 |

### 主要观察点

- **最关键**：bicycle 和 awning-tricycle 的 mAP50 是否有显著提升（目标 >0.12，即翻倍）
- 如果 mAP50 整体提升 ≥ 0.02 且推理速度降幅 < 20%，则认为 P2 分支有效，进入 EXP-03
- 如果 mAP50 提升 < 0.01，需分析原因（P2 通道数是否不足？是否需要加注意力？）

---

## 实验结果（训练后填写）

### 总体指标（val split，best.pt）

| 指标 | EXP-02 (P2) | EXP-01 (Baseline) | Delta |
|---|---|---|---|
| Precision | — | 0.434 | — |
| Recall | — | 0.344 | — |
| mAP50 | — | 0.320 | — |
| mAP50-95 | — | 0.181 | — |

### 各类别 mAP50（val split，best.pt）

| 类别 | EXP-02 (P2) | EXP-01 (Baseline) | Delta |
|---|---|---|---|
| pedestrian | — | 0.343 | — |
| people | — | 0.263 | — |
| bicycle | — | 0.083 | — |
| car | — | 0.748 | — |
| van | — | 0.349 | — |
| truck | — | 0.301 | — |
| tricycle | — | 0.190 | — |
| awning-tricycle | — | 0.091 | — |
| bus | — | 0.472 | — |
| motor | — | 0.359 | — |
| **mAP50 均值** | — | **0.320** | — |

### 推理速度与模型规模

| 指标 | EXP-02 (P2) | EXP-01 (Baseline) |
|---|---|---|
| 推理速度（ms/img，GPU） | — | ~3.2 |
| 参数量（M） | — | 2.6 |
| FLOPs（G） | — | 6.6 |

### 训练关键节点

| 节点 | Epoch | mAP50 | 说明 |
|---|---|---|---|
| best.pt 保存 | — | — | （填写实际 epoch） |
| 训练结束 | 100 | — | last.pt |

---

## 主要观察（训练后填写）

- [ ] bicycle / awning-tricycle AP 是否有显著提升？
- [ ] 整体 Recall 是否提升？
- [ ] 推理速度损耗是否在可接受范围（< 20%）？
- [ ] 是否存在过拟合迹象（val loss 曲线）？
- [ ] 与预期方向是否吻合？

---

## 结论与后续规划（训练后填写）

| EXP | 内容 | 状态 |
|---|---|---|
| EXP-01 | YOLOv11n 基线（P3/P4/P5） | ✅ 完成 |
| **EXP-02** | **P2 分支消融** | 🔄 进行中 |
| EXP-03 | 加权跨尺度融合（LWAF） | ⬜ 待开始 |
| EXP-04 | 轻量通道-空间注意力 | ⬜ 待开始 |
| EXP-05 | 小目标损失函数（IoU+NWD） | ⬜ 待开始 |
| EXP-06 | Ghost/Rep 轻量化模块 | ⬜ 待开始 |
| EXP-07 | ASE-YOLOv11 全量模型 | ⬜ 待开始 |
| EXP-08 | MPID 迁移验证 | ⬜ 待数据 |
