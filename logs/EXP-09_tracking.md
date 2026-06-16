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
| 状态 | 🔄 进行中（09a ✅ / 09b ✅ / 09c ⬜）|

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

### EXP-09c（后续）：YOLOv11-MCTrack 自研方法

| 模块 | 说明 |
|---|---|
| 相机运动补偿 | 帧间特征匹配估计单应性，补偿 Kalman 预测偏差 |
| 双阈值关联 | 高置信度检测主匹配 + 低置信度遮挡补偿（ByteTrack 思路） |
| 轻量 ReID | 轻量特征提取，遮挡/交叉场景降低 ID Switch |
| 轨迹置信传播 | 融合检测置信度 + 轨迹连续性 → 轨迹置信度分数 |
| 遮挡记忆 | 短时消失目标保留有限帧记忆 + 运动补偿重识别 |

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

### EXP-09c YOLOv11-MCTrack 结果（后续）

| 指标 | 值 |
|---|---|
| HOTA | — |
| MOTA | — |
| IDF1 | — |
| ID Switch (IDS) | — |
| Fragmentation (Frag) | — |

### 对比汇总

| 方法 | HOTA | MOTA | IDF1 | IDS | Frag |
|---|---|---|---|---|---|
| EXP-09a ASE + BoT-SORT（全类） | 9.97 | -32.51 | 6.18 | 54,063 | 5,116 |
| EXP-09b ASE + ByteTrack（pedestrian only） | 6.12 | -9.52 | 3.51 | 15,652 | 2,277 |
| EXP-09c YOLOv11-MCTrack | — | — | — | — | — |

> **说明**：09a 与 09b 不具备直接可比性（09a 全类追踪导致 FP 虚高，09b 单类追踪暴露召回不足）。
> 后续 09c 统一采用 pedestrian only 评估，与 09b 形成公平对比。

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

---

## 观察与结论（运行后填写）

> **注意**：以下内容必须基于实际评估结果，禁止填写预期值。

- [x] EXP-09a 完成，指标已记录（HOTA=9.97, MOTA=-32.51, IDF1=6.18）
- [x] EXP-09b 完成，指标已记录（HOTA=6.12, MOTA=-9.52, IDF1=3.51）
- [ ] EXP-09c（MCTrack）完成，指标已记录
- [ ] 对比分析已完成

**初步结论**（基于 09a/09b）：

1. **评估对齐问题**：09a 全类追踪导致 MOTA 虚低（FP=48,718），09b 加 `classes=0` 后 FP 降至 15,043，MOTA 改善 22.99pp，但这是评估配置问题，非追踪器本身的差距。
2. **核心瓶颈在检测**：09b 中 DetA=12.02，CLR_Re=17.38%，说明 ASE-YOLOv11 对 pedestrian 的单类召回率不足，漏检（ML=77.7%）是追踪失效的主因。
3. **EXP-09c 方向**：MCTrack 需要同时改善 pedestrian 检测召回（可考虑针对 pedestrian 微调检测器）+ 相机运动补偿降低 IDS。

---

## 数据完整性声明

- 数据集来源：VisDrone 官方 GitHub Release，学术使用许可
- 实验结果来源：服务器 TrackEval 评估输出，非手动估计
- 本文件中所有空白字段（—）须在实验完成后用真实数据填写
