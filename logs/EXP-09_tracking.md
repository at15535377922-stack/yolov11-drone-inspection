# EXP-09：VisDrone-MOT 多目标追踪实验

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-09 |
| 实验目的 | 验证 ASE-YOLOv11 作为检测前端，结合追踪器在 VisDrone-MOT 上的多目标追踪性能 |
| 数据集 | VisDrone2019-MOT |
| 来源 | https://github.com/VisDrone/VisDrone-Dataset |
| 版本 | VisDrone2019-MOT（2018~2019 发布） |
| 下载日期 | 2026-06-15 |
| 许可 | 学术使用 |
| 服务器路径 | `/root/autodl-tmp/yolov11-drone-inspection/data/visdrone-mot/` |
| 状态 | 🔄 进行中（09a~09h ✅ / 09i ⬜ 隔离验证遮挡恢复机制）|

---

## 数据集结构（已确认 2026-06-15）

```
data/visdrone-mot/
  VisDrone2019-MOT-train/
    sequences/   # 56 个视频序列，每序列为独立目录，含逐帧 jpg
    annotations/ # 每序列对应一个 txt，格式见下方
  VisDrone2019-MOT-val/
    sequences/   # 待确认序列数
    annotations/
  VisDrone2019-MOT-test-dev/
    sequences/   # 无 GT，仅用于可选提交
```

**标注格式**（MOT Challenge 兼容）：
```
frame_id, target_id, bb_left, bb_top, bb_width, bb_height, score, category, truncation, occlusion
```
- `frame_id`：从 1 开始
- `score=0`：GT 标注固定为 0（非置信度）
- `category`：1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor（与 VisDrone-DET 一致）

---

## 实验设计

### EXP-09a：ASE-YOLOv11 + BoT-SORT 基线

| 参数 | 值 |
|---|---|
| 检测器 | ASE-YOLOv11（EXP-07 best.pt，mAP50=0.337） |
| 追踪器 | BoT-SORT（Ultralytics 内置） |
| 检测置信度阈值 | 0.25 |
| IoU 阈值 | 0.45 |
| 输入分辨率 | 640 |
| 评估集 | VisDrone-MOT val |
| 评估框架 | TrackEval（HOTA / CLEAR / Identity） |
| 脚本 | `scripts/tracking/run_track_exp09a_botsort.sh` |
| 输出目录 | `runs/track/exp09a_botsort/` |

### EXP-09b（后续）：ASE-YOLOv11 + ByteTrack 对比

| 参数 | 值 |
|---|---|
| 检测器 | 同 09a（EXP-07 best.pt） |
| 追踪器 | ByteTrack（Ultralytics 内置） |
| 检测置信度阈值 | 0.25 |
| IoU 阈值 | 0.45 |
| 输入分辨率 | 640 |
| **过滤类别** | **classes=0（仅 pedestrian，与 TrackEval 评估对齐）** |
| 目的 | 与 09a 对比追踪器差异；classes=0 消除 FP 污染 |

### EXP-09c：YOLOv11-MCTrack 自研方法

**基础框架**：BoT-SORT（Ultralytics 内置），在此基础上针对无人机低空俯拍场景调整配置。

| 创新模块 | 实现方式 | 对应配置 |
|---|---|---|
| ① GMC 相机运动补偿 | sparseOptFlow 稀疏光流估计帧间变换 | `gmc_method: sparseOptFlow` |
| ② 双阈值降低 | 主阈值 0.15 / 辅阈值 0.05（默认 0.25/0.10）| `track_high_thresh: 0.15` |
| ③ 轻量 ReID | 复用 ASE-YOLOv11 骨干特征嵌入，零额外参数 | `with_reid: True, model: auto` |
| ④ 轨迹置信传播 | 融合检测置信度与 IoU 关联距离 | `fuse_score: True` |
| ⑤ 遮挡记忆扩大 | track_buffer 从 30 扩至 60 帧 | `track_buffer: 60` |

| 参数 | 值 |
|---|---|
| 检测器 | ASE-YOLOv11（EXP-07 best.pt） |
| 追踪器配置 | `configs/trackers/mctrack.yaml` |
| 检测置信度阈值 | **0.10**（降低以提升行人召回） |
| IoU 阈值 | 0.45 |
| 输入分辨率 | 640 |
| 过滤类别 | classes=0（pedestrian only，与 09b 统一评估口径） |
| 脚本 | `scripts/tracking/run_track_exp09c_mctrack.sh` |

### EXP-09d：pedestrian 专用检测器训练（解决检测召回瓶颈）

> **背景**：09b/09c 均确认追踪瓶颈在检测端——ASE-YOLOv11（10 类通用模型）对 pedestrian 单类
> 召回率过低（DetA≈12，CLR_Re≈18%），单纯调追踪器参数收益有限。09d 训练一个 pedestrian
> 专用检测器，作为 09e/09f 的高召回检测前端。

| 参数 | 值 |
|---|---|
| 基础结构 | yolo11-p2-full.yaml（与 EXP-07 一致） |
| 预训练权重 | EXP-07 best.pt（迁移学习，而非从头训练） |
| 训练数据 | VisDrone2019-DET，仅保留 class=0（pedestrian），nc=1 |
| 数据准备脚本 | `scripts/dataset/filter_visdrone_ped.py` |
| 数据配置 | `configs/datasets/visdrone_ped.yaml` |
| epochs | 50（patience=20） |
| optimizer | AdamW，lr0=1e-4 |
| 输入分辨率 | 640，batch=32 |
| 训练脚本 | `scripts/train/run_visdrone_ped_exp09d.sh` |
| 输出目录 | `runs/detect/visdrone_ped_exp09d/` |

### EXP-09e：pedestrian 专用检测器 + ByteTrack（公平对比基线）

| 参数 | 值 |
|---|---|
| 检测器 | EXP-09d best.pt（pedestrian 专用） |
| 追踪器 | ByteTrack |
| 检测置信度阈值 | 0.25 |
| 脚本 | `scripts/tracking/run_track_exp09e_ped_bytetrack.sh` |
| 输出目录 | `runs/track/exp09e_ped_bytetrack/` |

### EXP-09f：pedestrian 专用检测器 + YOLOv11-MCTrack（本文方法，核心对比）

| 参数 | 值 |
|---|---|
| 检测器 | EXP-09d best.pt（pedestrian 专用，与 09e 相同，控制变量） |
| 追踪器配置 | `configs/trackers/mctrack.yaml`（GMC + ReID + 双阈值 + 遮挡记忆，同 09c） |
| 检测置信度阈值 | 0.25 |
| 脚本 | `scripts/tracking/run_track_exp09f_ped_mctrack.sh` |
| 输出目录 | `runs/track/exp09f_ped_mctrack/` |

> **09e vs 09f 是论文追踪章节的核心对比**：检测器相同（09d），唯一变量是追踪器
> （ByteTrack vs MCTrack），可以真正验证 MCTrack 的追踪创新点是否有效，
> 不再受检测召回率差异干扰。

### ⚠️ 09e/09f 结果一致的根因排查：外层 conf 阈值屏蔽了双阈值创新点

09e/09f 实测结果完全一致（见下方结果），逐行 diff 确认两者轨迹分配几乎逐帧相同。
排查发现根因**不是"追踪器天生无效"，而是配置问题**：

- 09e/09f 运行时外层 `conf=0.25`（yolo track 的检测置信度硬阈值，在检测输出阶段
  就丢弃所有低于该值的框，追踪器完全看不到）
- `mctrack.yaml` 的 `track_high_thresh=0.15`、`track_low_thresh=0.05` 是追踪器
  **内部**对已收到的检测框做二次分流用的阈值
- 由于外层 `conf=0.25 > track_high_thresh=0.15`，**每一个能到达追踪器的框都已经
  高于 0.15**，等于永远只走"高置信度"分支，`track_low_thresh=0.05` 对应的
  遮挡恢复/二次匹配逻辑（双阈值创新点②）**从未被触发过**

→ 需要用更低的外层 conf（对齐 track_low_thresh）重新对比，才能验证双阈值机制
是否真的有效，详见 EXP-09g/09h。

### EXP-09g：pedestrian 专用检测器 + ByteTrack（低 conf，激活双阈值分流对比基线）

| 参数 | 值 |
|---|---|
| 检测器 | EXP-09d best.pt（与 09e/09f 相同） |
| 追踪器 | ByteTrack |
| 检测置信度阈值 | **0.05**（对齐 mctrack.yaml 的 track_low_thresh，与 09h 完全一致） |
| 脚本 | `scripts/tracking/run_track_exp09g_ped_bytetrack_lowconf.sh` |
| 输出目录 | `runs/track/exp09g_ped_bytetrack_lowconf/` |

### EXP-09h：pedestrian 专用检测器 + YOLOv11-MCTrack（低 conf，真正激活双阈值创新点）

| 参数 | 值 |
|---|---|
| 检测器 | EXP-09d best.pt（与 09g 相同，控制变量） |
| 追踪器配置 | `configs/trackers/mctrack.yaml` |
| 检测置信度阈值 | **0.05**（与 09g 完全一致，唯一变量是追踪器） |
| 脚本 | `scripts/tracking/run_track_exp09h_ped_mctrack_lowconf.sh` |
| 输出目录 | `runs/track/exp09h_ped_mctrack_lowconf/` |

> **09g vs 09h 才是真正验证双阈值机制的公平对比**：外层 conf 降到 0.05 后，
> 0.05~0.25 区间的检测框能真正进入追踪器内部的高/低置信度分流逻辑，
> MCTrack 的 track_low_thresh 分支才有机会被激活。若 09h 优于 09g，
> 证明双阈值等创新点有效；若仍然相同，则说明瓶颈确实在检测密度本身，
> 而非本次配置问题。

---

## 实验结果（运行后填写）

### EXP-09a 结果（已完成 2026-06-15）

> 评估集：VisDrone-MOT val（7 序列），仅评估 pedestrian 类（TrackEval 默认）。
> 检测器置信度阈值 conf=0.25，IoU=0.45，输入分辨率 640。

**COMBINED（7 序列平均）：**

| 指标 | 值 |
|---|---|
| HOTA | **9.97** |
| MOTA | **-32.51** |
| IDF1 | **6.18** |
| ID Switch (IDS) | **54063** |
| Fragmentation (Frag) | **5116** |
| DetA（检测关联精度） | 32.55 |
| AssA（身份关联精度） | 3.74 |
| LocA（定位精度） | 77.18 |
| CLR_Re（召回） | 57.55% |
| CLR_Pr（精度） | 57.41% |
| MT（大多数时间追踪） | 287 / 758（37.9%）|
| ML（大多数时间丢失） | 176 / 758（23.2%）|
| 总检测框数 | 114,395 |
| GT 框数 | 114,132 |
| 总 ID 数（预测） | 522 |
| GT ID 数 | 758 |

**各序列 HOTA 明细：**

| 序列 | HOTA | MOTA | IDF1 | IDS | Frag |
|---|---|---|---|---|---|
| uav0000086_00000_v | 8.22 | -14.02 | 5.82 | 10471 | 1100 |
| uav0000117_02622_v | 8.04 | -52.56 | 4.85 | 8325 | 930 |
| uav0000137_00458_v | 9.05 | -33.26 | 5.52 | 15413 | 1363 |
| uav0000182_00000_v | 6.82 | -64.20 | 4.19 | 9433 | 833 |
| uav0000268_05773_v | **18.40** | **11.05** | **13.88** | 673 | 106 |
| uav0000305_00000_v | 9.29 | -78.92 | 6.41 | 4474 | 185 |
| uav0000339_00001_v | 12.07 | -15.66 | 8.28 | 5274 | 599 |
| **COMBINED** | **9.97** | **-32.51** | **6.18** | **54063** | **5116** |

> **注**：MOTA 大幅负值主要由 CLR_FP（虚警 48,718）和 IDSW（54,063）贡献，
> 根本原因是检测器未针对 pedestrian 单类过滤（输出了所有 10 类），
> 导致非行人目标被当成 pedestrian 计入 FP。
> 后续 EXP-09b 需在追踪前过滤仅保留 pedestrian（class=0）检测结果。

### EXP-09b 结果（已完成 2026-06-16）

> 评估集：VisDrone-MOT val（7 序列），仅评估 pedestrian 类。
> 检测器 conf=0.25，IoU=0.45，输入分辨率 640，**classes=0（仅追踪 pedestrian）**。

**COMBINED（7 序列平均）：**

| 指标 | 值 |
|---|---|
| HOTA | **6.12** |
| MOTA | **-9.52** |
| IDF1 | **3.51** |
| ID Switch (IDS) | **15652** |
| Fragmentation (Frag) | **2277** |
| DetA（检测关联精度） | 12.02 |
| AssA（身份关联精度） | 3.23 |
| LocA（定位精度） | 74.61 |
| CLR_Re（召回） | 17.38% |
| CLR_Pr（精度） | 56.87% |
| MT（大多数时间追踪） | 53 / 758（7.0%）|
| ML（大多数时间丢失） | 589 / 758（77.7%）|
| 总检测框数 | 34,877 |
| GT 框数 | 114,132 |
| 总 ID 数（预测） | 187 |
| GT ID 数 | 758 |

**各序列 HOTA 明细：**

| 序列 | HOTA | MOTA | IDF1 | IDS | Frag |
|---|---|---|---|---|---|
| uav0000086_00000_v | 8.21 | -9.43 | 5.79 | 9389 | 914 |
| uav0000117_02622_v | 4.93 | -31.52 | 3.07 | 3255 | 504 |
| uav0000137_00458_v | 5.35 | -4.71 | 2.72 | 1919 | 517 |
| uav0000182_00000_v | 2.42 | -4.97 | 1.05 | 118 | 57 |
| uav0000268_05773_v | 0.00 | -0.79 | 0.00 | 0 | 0 |
| uav0000305_00000_v | 2.75 | -16.90 | 1.65 | 144 | 52 |
| uav0000339_00001_v | 10.85 | -0.31 | 7.43 | 827 | 233 |
| **COMBINED** | **6.12** | **-9.52** | **3.51** | **15652** | **2277** |

> **注**：加入 `classes=0` 过滤后，FP 从 48,718 降至 15,043（↓69%），IDS 从 54,063 降至 15,652（↓71%），
> MOTA 从 -32.51 改善至 -9.52。但 CLR_Re 从 57.55% 大幅下降至 17.38%，
> 说明 ASE-YOLOv11（VisDrone 10 类训练）对 pedestrian 单类的召回率较低（DetA=12.02），
> 导致漏检严重（ML=77.7%）。HOTA 反而低于 09a（6.12 vs 9.97），根本原因是 DetA 过低。
> EXP-09c MCTrack 需重点提升 pedestrian 检测召回率或采用专项 pedestrian 检测器。

### EXP-09c YOLOv11-MCTrack 结果（已完成 2026-06-16）

> 评估集：VisDrone-MOT val（7 序列），pedestrian only（classes=0）。
> 检测器 conf=**0.10**（降低以提升召回），IoU=0.45，输入分辨率 640。
> 追踪器：BoT-SORT + GMC(sparseOptFlow) + ReID(auto) + track_buffer=60。

**COMBINED（7 序列平均）：**

| 指标 | 值 |
|---|---|
| HOTA | **6.02** |
| MOTA | **-15.51** |
| IDF1 | **3.48** |
| ID Switch (IDS) | **17154** |
| Fragmentation (Frag) | **2531** |
| DetA（检测关联精度） | 12.44 |
| AssA（身份关联精度） | 3.04 |
| LocA（定位精度） | 74.14 |
| CLR_Re（召回） | 18.77% |
| CLR_Pr（精度） | 49.37% |
| MT（大多数时间追踪） | 59 / 758（7.8%）|
| ML（大多数时间丢失） | 569 / 758（75.1%）|
| 总检测框数 | 43,393 |
| GT 框数 | 114,132 |
| 总 ID 数（预测） | 229 |
| GT ID 数 | 758 |

**各序列 HOTA 明细：**

| 序列 | HOTA | MOTA | IDF1 | IDS | Frag |
|---|---|---|---|---|---|
| uav0000086_00000_v | 8.01 | -17.73 | 5.56 | 9960 | 951 |
| uav0000117_02622_v | 4.71 | -47.27 | 2.97 | 3554 | 547 |
| uav0000137_00458_v | 5.33 | -8.37 | 2.79 | 2299 | 632 |
| uav0000182_00000_v | 2.31 | -7.27 | 1.13 | 158 | 76 |
| uav0000268_05773_v | 0.01 | -1.41 | 0.00 | 0 | 0 |
| uav0000305_00000_v | 2.76 | -21.17 | 1.66 | 169 | 59 |
| uav0000339_00001_v | 10.72 | -4.91 | 7.43 | 1014 | 266 |
| **COMBINED** | **6.02** | **-15.51** | **3.48** | **17154** | **2531** |

> **注**：相比 09b，conf=0.10 将总检测框从 34,877 增至 43,393（+24%），
> CLR_Re 从 17.38% 微升至 18.77%（+1.4pp），但 FP 从 15,043 增至 21,970（+46%），
> MOTA 从 -9.52 退至 -15.51，说明低 conf 阈值带来的 FP 增量抵消了召回收益。
> GMC+ReID 组合未能有效降低 IDS（17,154 vs 09b 的 15,652），
> 核心问题仍是 ASE-YOLOv11 对 pedestrian 单类召回率不足，非追踪器参数能解决。

### 对比汇总

| 方法 | HOTA | MOTA | IDF1 | IDS | Frag |
|---|---|---|---|---|---|
| EXP-09a ASE + BoT-SORT（全类，conf=0.25） | 9.97 | -32.51 | 6.18 | 54,063 | 5,116 |
| EXP-09b ASE（10类）+ ByteTrack（pedestrian only，conf=0.25） | 6.12 | -9.52 | 3.51 | 15,652 | 2,277 |
| EXP-09c ASE（10类）+ MCTrack（pedestrian only，conf=0.10） | 6.02 | -15.51 | 3.48 | 17,154 | 2,531 |
| EXP-09e ped专用检测器 + ByteTrack（conf=0.25） | 6.19 | -12.61 | 3.59 | 18,208 | 2,410 |
| **EXP-09f ped专用检测器 + MCTrack（conf=0.25，同检测器公平对比）** | **6.19** | **-12.61** | **3.59** | **18,208** | **2,410** |

> **说明**：
> - 09a 全类追踪导致 FP 虚高（48,718），不具可比性，仅作为"现有方法不足"的佐证。
> - 09b vs 09c：检测器相同但 conf 不同，非严格控制变量，差异主要来自 conf 阈值而非追踪器本身。
> - **09e vs 09f 是严格控制变量的对比**（检测器、conf、iou 完全相同，只有追踪器不同）：
>   两者指标完全一致，逐行 diff 证实轨迹结果几乎逐帧相同（7 序列中 3 个 0 差异，
>   其余 4 个仅 1 行浮点舍入级差异）。
> - **根因排查**：定位到 09e/09f 用的外层 `conf=0.25` 高于 `mctrack.yaml` 的
>   `track_high_thresh=0.15`，导致双阈值遮挡恢复分支从未被触发——这是配置问题，
>   不能据此判定 MCTrack 无效。**最终结论待 EXP-09g/09h（对齐 track_low_thresh 的
>   低 conf 重新对比）验证后补充。**

### EXP-09d 结果（待运行）

| 指标 | 值 |
|---|---|
| mAP50（pedestrian） | — |
| mAP50-95 | — |
| Precision | — |
| Recall | — |
| 训练轮数（实际停止于） | — |

### EXP-09e 结果（已完成 2026-07-09，评估脚本重跑修正后取得）

> 评估集：VisDrone-MOT val（7 序列），pedestrian only。
> 检测器：EXP-09d pedestrian 专用权重，conf=0.25，IoU=0.45，输入分辨率 640。追踪器：ByteTrack。
> 注：`eval_trackeval.py` 此前存在结果解析 bug（误按 tracker_name 索引顶层，导致 summary.json 恒为空 `{}`），
> 已修复；本结果取自 TrackEval 控制台原始输出（与 09a/b/c 早期人工誊抄口径一致）。

**COMBINED（7 序列平均）：**

| 指标 | 值 |
|---|---|
| HOTA | **6.19** |
| MOTA | **-12.61** |
| IDF1 | **3.59** |
| ID Switch (IDS) | **18208** |
| Fragmentation (Frag) | **2410** |
| DetA（检测关联精度） | 13.32 |
| AssA（身份关联精度） | 3.01 |
| LocA（定位精度） | 74.56 |
| CLR_Re（召回） | 19.66% |
| CLR_Pr（精度） | 54.65% |
| MT（大多数时间追踪） | 64 / 758（8.4%）|
| ML（大多数时间丢失） | 558 / 758（73.6%）|
| 总检测框数 | 41,053 |
| GT 框数 | 114,132 |
| 总 ID 数（预测） | 226 |
| GT ID 数 | 758 |

**各序列 HOTA 明细：**

| 序列 | HOTA | MOTA | IDF1 | IDS | Frag |
|---|---|---|---|---|---|
| uav0000086_00000_v | 7.97 | -16.05 | 5.40 | 10002 | 851 |
| uav0000117_02622_v | 4.92 | -37.22 | 3.34 | 3881 | 497 |
| uav0000137_00458_v | 5.42 | -5.53 | 2.90 | 2861 | 706 |
| uav0000182_00000_v | 2.45 | -5.41 | 1.12 | 188 | 86 |
| uav0000268_05773_v | 0.00 | -0.45 | 0.00 | 0 | 0 |
| uav0000305_00000_v | 2.30 | -26.26 | 1.06 | 150 | 52 |
| uav0000339_00001_v | 11.49 | -1.84 | 8.38 | 1126 | 218 |
| **COMBINED** | **6.19** | **-12.61** | **3.59** | **18208** | **2410** |

> **注**：相比 09b（10 类通用检测器过滤 pedestrian，CLR_Re=17.38%，DetA=12.02），
> 09e 换用 09d 训练的 pedestrian 专用检测器后，CLR_Re 升至 19.66%（+2.3pp），DetA 升至 13.32，
> 召回率提升有限，未达到预期的"大幅提升"。同时 CLR_Pr 从 56.87% 降至 54.65%（FP 从 15,043 增至 18,616），
> MOTA 反而从 -9.52 恶化至 -12.61。说明单纯用 EXP-07 权重迁移学习微调 50 epoch，
> 对 pedestrian 单类召回率的提升幅度不足以扭转追踪指标；检测端瓶颈依然存在，只是略有缓解。

### EXP-09f 结果（已完成 2026-07-09）

> 评估集：VisDrone-MOT val（7 序列），pedestrian only。检测器与 09e 完全相同（EXP-09d ped 专用权重，conf=0.25），
> 唯一变量是追踪器：09e=ByteTrack，09f=MCTrack（`configs/trackers/mctrack.yaml`，GMC+ReID+双阈值+track_buffer=60）。

**逐帧结果核验（关键步骤）**：对比 09e/09f 的 `mot_results/*.txt` 逐行 diff，7 个序列中 3 个完全一致（0 行差异），
其余 4 个序列仅有 1 行差异（bbox 宽度 62.88→62.89，浮点舍入级别，很可能是 GMC 光流估计的极小修正）。
即：**MCTrack 与 ByteTrack 在同一检测器输入下产出的轨迹 ID 分配、轨迹数量、跨帧关联结果几乎完全相同。**

**COMBINED（7 序列平均）：** 与 09e 相同（差异在 TrackEval 指标精度范围内不可分辨）

| 指标 | 值 | 对比 09e |
|---|---|---|
| HOTA | **6.19** | 持平 |
| MOTA | **-12.61** | 持平 |
| IDF1 | **3.59** | 持平 |
| ID Switch (IDS) | **18208** | 持平 |
| Fragmentation (Frag) | **2410** | 持平 |
| DetA | 13.32 | 持平 |
| AssA | 3.01 | 持平 |
| CLR_Re | 19.66% | 持平 |
| CLR_Pr | 54.65% | 持平 |

> **根因排查结果**：逐行 diff 证实两者轨迹分配几乎逐帧相同。深入检查配置后发现，
> 09e/09f 使用的外层 `conf=0.25` 已经高于 `mctrack.yaml` 的 `track_high_thresh=0.15`，
> 导致能到达追踪器的检测框全部被判定为"高置信度"，`track_low_thresh=0.05` 对应的
> 双阈值遮挡恢复分支从未被触发——**这是配置问题，不是"追踪器创新点必然无效"的证据**。
> 详见下方 EXP-09g/09h：用对齐 `track_low_thresh` 的低 conf（0.05）重新做公平对比，
> 真正激活双阈值机制后再下结论。

### EXP-09g 结果（已完成 2026-07-09）

> 检测器 EXP-09d（与 09h 相同），ByteTrack，conf=0.05（对齐 track_low_thresh）。

| 指标 | 值 |
|---|---|
| HOTA | **6.19** |
| MOTA | **-12.61** |
| IDF1 | **3.59** |
| ID Switch (IDS) | **18208** |
| Fragmentation (Frag) | **2410** |
| DetA | 13.32 |
| CLR_Re | 19.66% |
| CLR_Pr | 54.65% |
| 总检测框数 | 41,053 |
| 总 ID 数 | 226 |

### EXP-09h 结果（已完成 2026-07-09）

> 检测器 EXP-09d（与 09g 相同，控制变量），MCTrack（`mctrack.yaml`），conf=0.05（与 09g 一致）。
> 逐行 diff 核验：与 09g 相比 7 个序列全部有大量差异（数百至数千行不等），
> 确认这次双阈值/宽松新建轨迹阈值机制真正被激活，两者产出了实质不同的轨迹结果
> （不同于此前 09e/09f 逐帧几乎完全一致的情况）。

| 指标 | 值 | 对比 09g |
|---|---|---|
| HOTA | **6.08** | ↓ -0.11 |
| MOTA | **-18.65** | ↓ -6.04（更差）|
| IDF1 | **3.54** | ↓ -0.05 |
| ID Switch (IDS) | **19830** | ↑ +1622（更差）|
| Fragmentation (Frag) | **2652** | ↑ +242（更差）|
| DetA | 13.70 | ↑ +0.38 |
| CLR_Re | 21.15% | ↑ +1.5pp |
| CLR_Pr | 48.54% | ↓ -6.1pp（更差）|
| 总检测框数 | 49,722 | ↑ +21% |
| 总 ID 数 | 274 | ↑ +48 |

> **最终结论（双阈值机制真正激活后的公平对比）**：
> MCTrack 的 `new_track_thresh=0.20` 比 ByteTrack 默认 `0.25` 更宽松，激活后确实多捕获了
> 更多检测框（+21%）、召回率也略有提升（CLR_Re 19.66%→21.15%），证明双阈值/宽松新建
> 轨迹阈值机制本身在起作用，不再是"配置屏蔽导致零差异"的假阴性。
>
> **但这次差异是负面的**：多捕获的检测框里混入了更多误检（CLR_Pr 从 54.65% 降至 48.54%，
> 降幅达 6.1 个百分点），导致 MOTA、IDS、Frag、HOTA、IDF1 全面变差。即放宽阈值换来的
> 召回率提升，被随之增加的虚警和轨迹碎片化完全抵消、甚至得不偿失。
>
> **技术解释**：`new_track_thresh` 放宽后，很多低置信度误检框被直接当成新目标建立轨迹
> （而不是用来挽救已有轨迹），这些误检轨迹存活时间短、频繁产生新 ID 和碎片，
> 拉低了整体指标。GMC/ReID 等模块本应帮助过滤这类噪声，但显然没能弥补这个副作用。

### EXP-09i：MCTrack v2（隔离验证——只保留遮挡恢复能力，排除误开新轨迹副作用）

09h 证实了"放宽 new_track_thresh 换召回"这条路是净负收益。但双阈值机制本来的设计意图
是"用低置信度框挽救已有轨迹"（遮挡恢复），而不是"用低置信度框开新轨迹"。09h 里这两个
效应被搅在一起，无法单独评价遮挡恢复能力本身的价值。09i 把 `new_track_thresh` 调回
与 ByteTrack 默认一致的 0.25（消除误开新轨迹的副作用），只保留更低的 `track_low_thresh=0.05`
+ GMC + ReID + track_buffer=60，单独检验遮挡恢复机制本身是否有正向价值。

| 参数 | 值 |
|---|---|
| 检测器 | EXP-09d best.pt（与 09g/09h 相同） |
| 追踪器配置 | `configs/trackers/mctrack_v2_strictnew.yaml`（new_track_thresh 0.20→0.25，其余同 mctrack.yaml） |
| 检测置信度阈值 | 0.05（与 09g/09h 一致） |
| 脚本 | `scripts/tracking/run_track_exp09i_ped_mctrack_v2.sh` |
| 输出目录 | `runs/track/exp09i_ped_mctrack_v2/` |

### EXP-09i 结果（待运行）

| 指标 | 值 | 对比 09g（ByteTrack 基线） |
|---|---|---|
| HOTA | — | — |
| MOTA | — | — |
| IDF1 | — | — |
| ID Switch (IDS) | — | — |
| Fragmentation (Frag) | — | — |
| CLR_Re | — | — |
| CLR_Pr | — | — |

> **注**：这是本轮追踪实验的最后一次隔离验证。
> - 若 09i 优于 09g（HOTA/MOTA/IDF1 提升，CLR_Pr 不明显下降）→ 证明"遮挡恢复"机制本身
>   有效，09h 的负面结果是"误开新轨迹"这个副作用掩盖了真实收益，可以在论文中呈现
>   "正确配置后 MCTrack 有效，并说明为什么 09h 的直接放宽阈值方案不可取"这个更细致、
>   更有技术深度的叙事。
> - 若 09i 仍不如 09g → 基本可以确认，在当前检测密度和数据条件下，GMC/ReID/遮挡恢复
>   这几个机制对本任务确实没有可测量收益，瓶颈根本上在检测端，这也是一个完整、
>   诚实、有充分排查过程支撑的结论。

---

## 执行步骤记录

### Step 1：✅ 数据集已下载并解压（2026-06-15）

```
VisDrone2019-MOT-train.zip  8.0 GB -> VisDrone2019-MOT-train/
VisDrone2019-MOT-val.zip    1.5 GB -> VisDrone2019-MOT-val/
VisDrone2019-MOT-test-dev.zip 2.2 GB -> VisDrone2019-MOT-test-dev/
```

### Step 2：上传脚本文件

通过 git push → git pull，将以下文件同步到服务器：
- `scripts/tracking/run_track_exp09a_botsort.sh`
- `scripts/tracking/convert_to_mot.py`
- `scripts/tracking/eval_trackeval.py`
- `scripts/tracking/install_trackeval.sh`

### Step 3：安装 TrackEval

```bash
cd /root/autodl-tmp/yolov11-drone-inspection
source venv/bin/activate
bash scripts/tracking/install_trackeval.sh
```

### Step 4：运行 EXP-09a 追踪基线

```bash
bash scripts/tracking/run_track_exp09a_botsort.sh
```

运行开始时间：___
运行结束时间：___

### Step 5：查看评估结果

```bash
cat runs/track/exp09a_botsort/eval/summary.json
```

### Step 6（待执行）：09d 训练 → 09e/09f 追踪评估

```bash
cd /root/autodl-tmp/yolov11-drone-inspection
git pull                                            # 同步 09d/e/f 脚本（已 push，commit 118181b）
source venv/bin/activate

# 1. 训练 pedestrian 专用检测器（50 epochs，耗时视 GPU 而定）
bash scripts/train/run_visdrone_ped_exp09d.sh

# 2. 用 09d 权重 + ByteTrack 跑基线
bash scripts/tracking/run_track_exp09e_ped_bytetrack.sh

# 3. 用 09d 权重 + MCTrack 跑本文方法
bash scripts/tracking/run_track_exp09f_ped_mctrack.sh

# 4. 查看结果
cat runs/detect/visdrone_ped_exp09d/results.csv | tail -5
cat runs/track/exp09e_ped_bytetrack/eval/summary.json
cat runs/track/exp09f_ped_mctrack/eval/summary.json
```

运行开始时间：___
运行结束时间：___

### Step 7（待执行）：09g/09h 低 conf 重新对比（验证双阈值机制真实效果）

```bash
cd /root/autodl-tmp/yolov11-drone-inspection
git pull                                            # 同步 09g/09h 脚本 + eval_trackeval.py 修复
source venv/bin/activate

# 1. ByteTrack，conf=0.05
bash scripts/tracking/run_track_exp09g_ped_bytetrack_lowconf.sh

# 2. MCTrack，conf=0.05（与 09g 唯一变量仅追踪器不同）
bash scripts/tracking/run_track_exp09h_ped_mctrack_lowconf.sh

# 3. 查看结果
cat runs/track/exp09g_ped_bytetrack_lowconf/eval/summary.json
cat runs/track/exp09h_ped_mctrack_lowconf/eval/summary.json

# 4. 逐行核验两者是否仍然产出相同轨迹（关键：判断双阈值机制是否真的被激活并起作用）
for seq in uav0000086_00000_v uav0000117_02622_v uav0000137_00458_v uav0000182_00000_v uav0000268_05773_v uav0000305_00000_v uav0000339_00001_v; do
    echo "=== $seq ==="
    diff runs/track/exp09g_ped_bytetrack_lowconf/mot_results/${seq}.txt \
         runs/track/exp09h_ped_mctrack_lowconf/mot_results/${seq}.txt | wc -l
done
```

### Step 8（待执行）：EXP-09i 隔离验证遮挡恢复机制

```bash
cd /root/autodl-tmp/yolov11-drone-inspection
git pull
source venv/bin/activate

bash scripts/tracking/run_track_exp09i_ped_mctrack_v2.sh

cat runs/track/exp09i_ped_mctrack_v2/eval/summary.json

# 与 09g（ByteTrack 基线）逐行核验差异程度
for seq in uav0000086_00000_v uav0000117_02622_v uav0000137_00458_v uav0000182_00000_v uav0000268_05773_v uav0000305_00000_v uav0000339_00001_v; do
    echo "=== $seq ==="
    diff runs/track/exp09g_ped_bytetrack_lowconf/mot_results/${seq}.txt \
         runs/track/exp09i_ped_mctrack_v2/mot_results/${seq}.txt | wc -l
done
```

---

## 观察与结论（运行后填写）

> **注意**：以下内容必须基于实际评估结果，禁止填写预期值。

- [x] EXP-09a 完成，指标已记录（HOTA=9.97, MOTA=-32.51, IDF1=6.18）
- [x] EXP-09b 完成，指标已记录（HOTA=6.12, MOTA=-9.52, IDF1=3.51）
- [x] EXP-09c（MCTrack）完成，指标已记录（HOTA=6.02, MOTA=-15.51, IDF1=3.48）
- [x] 对比分析已完成
- [x] EXP-09d（pedestrian 专用检测器训练）完成，训练跑满 50 epoch（末轮 precision=0.570, recall=0.443, mAP50=0.472, mAP50-95=0.204；best.pt 对应轮次待确认）
- [x] EXP-09e（ped 检测器 + ByteTrack 基线）完成，指标已记录（HOTA=6.19, MOTA=-12.61, IDF1=3.59）
- [x] EXP-09f（ped 检测器 + MCTrack 本文方法）完成，指标已记录（与 09e 持平：HOTA=6.19, MOTA=-12.61, IDF1=3.59）
- [x] 09e vs 09f 公平对比分析已完成——逐行 diff 证实两者轨迹结果几乎完全相同
- [x] 根因排查：定位到外层 conf=0.25 屏蔽了 mctrack.yaml 的双阈值分支（track_high_thresh=0.15）
- [x] EXP-09g（ped 检测器 + ByteTrack，conf=0.05）完成，指标已记录（HOTA=6.19, MOTA=-12.61, IDF1=3.59）
- [x] EXP-09h（ped 检测器 + MCTrack，conf=0.05，真正激活双阈值）完成，指标已记录（HOTA=6.08, MOTA=-18.65, IDF1=3.54，全面劣于 09g）
- [x] 09g vs 09h 对比分析已完成：双阈值机制被激活但净效果为负（召回+1.5pp，误检+6.1pp precision 损失更大）
- [ ] EXP-09i（MCTrack v2，new_track_thresh 调回 0.25，隔离验证遮挡恢复机制本身）完成，指标已记录
- [ ] 09i vs 09g 最终对比完成，MCTrack 有效性最终结论待此确认

**初步结论**（基于 09a/09b）：

1. **评估对齐问题**：09a 全类追踪导致 MOTA 虚低（FP=48,718），09b 加 `classes=0` 后 FP 降至 15,043，MOTA 改善 22.99pp，但这是评估配置问题，非追踪器本身的差距。
2. **核心瓶颈在检测**：09b 中 DetA=12.02，CLR_Re=17.38%，说明 ASE-YOLOv11 对 pedestrian 的单类召回率不足，漏检（ML=77.7%）是追踪失效的主因。
3. **EXP-09c 方向**：MCTrack 需要同时改善 pedestrian 检测召回（可考虑针对 pedestrian 微调检测器）+ 相机运动补偿降低 IDS。

---

## 数据完整性声明

- 数据集来源：VisDrone 官方 GitHub Release，学术使用许可
- 实验结果来源：服务器 TrackEval 评估输出，非手动估计
- 本文件中所有空白字段（—）须在实验完成后用真实数据填写
