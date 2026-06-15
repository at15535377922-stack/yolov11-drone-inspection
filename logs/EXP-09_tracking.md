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
| 状态 | ⬜ 待执行 |

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
| 检测器 | 同 09a |
| 追踪器 | ByteTrack（Ultralytics 内置） |
| 其余参数 | 同 09a |
| 目的 | 与 09a 对比，选取更强基线 |

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

### EXP-09a 结果

| 指标 | 值 |
|---|---|
| HOTA | — |
| MOTA | — |
| IDF1 | — |
| ID Switch (IDS) | — |
| Fragmentation (Frag) | — |
| MT（大多数时间追踪） | — |
| ML（大多数时间丢失） | — |
| FPS（追踪，含检测） | — |

### EXP-09b 结果（后续）

| 指标 | 值 |
|---|---|
| HOTA | — |
| MOTA | — |
| IDF1 | — |
| ID Switch (IDS) | — |
| Fragmentation (Frag) | — |

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
| EXP-09a ASE + BoT-SORT | — | — | — | — | — |
| EXP-09b ASE + ByteTrack | — | — | — | — | — |
| EXP-09c YOLOv11-MCTrack | — | — | — | — | — |

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

- [ ] EXP-09a 完成，指标已记录
- [ ] EXP-09b 完成，指标已记录
- [ ] EXP-09c（MCTrack）完成，指标已记录
- [ ] 对比分析已完成

**初步结论**（运行后填写）：

---

## 数据完整性声明

- 数据集来源：VisDrone 官方 GitHub Release，学术使用许可
- 实验结果来源：服务器 TrackEval 评估输出，非手动估计
- 本文件中所有空白字段（—）须在实验完成后用真实数据填写
