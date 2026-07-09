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
| 状态 | 🔄 进行中（09a ✅ / 09b ✅ / 09c ✅ / 09d ⬜ / 09e ⬜ / 09f ⬜）|

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
| EXP-09b ASE + ByteTrack（pedestrian only，conf=0.25） | 6.12 | -9.52 | 3.51 | 15,652 | 2,277 |
| **EXP-09c MCTrack（pedestrian only，conf=0.10，GMC+ReID）** | **6.02** | **-15.51** | **3.48** | **17,154** | **2,531** |

> **说明**：
> - 09a 全类追踪导致 FP 虚高（48,718），不具可比性，仅作为"现有方法不足"的佐证。
> - 09b vs 09c 为公平对比（均 pedestrian only）：MCTrack 通过 conf 降低捕获更多检测，
>   但 FP 增量使 MOTA 劣于 09b；HOTA 相近（6.02 vs 6.12）。
> - **核心发现**：追踪性能瓶颈在于 pedestrian 检测端召回不足（DetA≈12），
>   单纯调整追踪器参数收益有限；论文需重点论述检测器专项优化方向。

### EXP-09d 结果（待运行）

| 指标 | 值 |
|---|---|
| mAP50（pedestrian） | — |
| mAP50-95 | — |
| Precision | — |
| Recall | — |
| 训练轮数（实际停止于） | — |

### EXP-09e 结果（待运行）

| 指标 | 值 |
|---|---|
| HOTA | — |
| MOTA | — |
| IDF1 | — |
| ID Switch (IDS) | — |
| Fragmentation (Frag) | — |
| DetA | — |
| AssA | — |
| CLR_Re | — |
| CLR_Pr | — |

### EXP-09f 结果（待运行）

| 指标 | 值 |
|---|---|
| HOTA | — |
| MOTA | — |
| IDF1 | — |
| ID Switch (IDS) | — |
| Fragmentation (Frag) | — |
| DetA | — |
| AssA | — |
| CLR_Re | — |
| CLR_Pr | — |

> **注**：09e vs 09f 结果出来后，直接对比 HOTA/MOTA/IDS 即可判断 MCTrack 的追踪器创新
> （GMC/ReID/双阈值/遮挡记忆）在检测召回率提升之后是否真正带来增益。若 09f 优于 09e，
> 则验证了创新点二的有效性；若仍未见提升，需在论文中如实讨论并转向分析检测-追踪耦合问题。

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

---

## 观察与结论（运行后填写）

> **注意**：以下内容必须基于实际评估结果，禁止填写预期值。

- [x] EXP-09a 完成，指标已记录（HOTA=9.97, MOTA=-32.51, IDF1=6.18）
- [x] EXP-09b 完成，指标已记录（HOTA=6.12, MOTA=-9.52, IDF1=3.51）
- [x] EXP-09c（MCTrack）完成，指标已记录（HOTA=6.02, MOTA=-15.51, IDF1=3.48）
- [x] 对比分析已完成
- [ ] EXP-09d（pedestrian 专用检测器训练）完成，mAP/Recall 已记录
- [ ] EXP-09e（ped 检测器 + ByteTrack 基线）完成，指标已记录
- [ ] EXP-09f（ped 检测器 + MCTrack 本文方法）完成，指标已记录
- [ ] 09e vs 09f 公平对比分析已完成

**初步结论**（基于 09a/09b）：

1. **评估对齐问题**：09a 全类追踪导致 MOTA 虚低（FP=48,718），09b 加 `classes=0` 后 FP 降至 15,043，MOTA 改善 22.99pp，但这是评估配置问题，非追踪器本身的差距。
2. **核心瓶颈在检测**：09b 中 DetA=12.02，CLR_Re=17.38%，说明 ASE-YOLOv11 对 pedestrian 的单类召回率不足，漏检（ML=77.7%）是追踪失效的主因。
3. **EXP-09c 方向**：MCTrack 需要同时改善 pedestrian 检测召回（可考虑针对 pedestrian 微调检测器）+ 相机运动补偿降低 IDS。

---

## 数据完整性声明

- 数据集来源：VisDrone 官方 GitHub Release，学术使用许可
- 实验结果来源：服务器 TrackEval 评估输出，非手动估计
- 本文件中所有空白字段（—）须在实验完成后用真实数据填写
