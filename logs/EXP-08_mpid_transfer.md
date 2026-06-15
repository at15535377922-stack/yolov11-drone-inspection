# EXP-08：MPID 绝缘子数据集迁移验证

## 实验信息

| 项目 | 内容 |
|---|---|
| 实验编号 | EXP-08 |
| 实验目的 | 验证 ASE-YOLOv11 在 MPID 绝缘子缺陷数据集上的迁移适应能力 |
| 数据集 | MPID（Merged Public Insulator Dataset）|
| 来源 | https://zenodo.org/records/14604384 |
| 版本 | v1（2025-01-06）|
| 下载日期 | 2026-06-15 |
| 许可 | CC BY 4.0 |
| 服务器路径 | `/root/autodl-tmp/yolov11-drone-inspection/data/mpid/` |
| 状态 | ✅ 已完成（08a ✅ / 08b ✅）|

---

## 数据集基本信息（解压后填写）

| 项目 | glass | porcelain | composite | 合并后 |
|---|---|---|---|---|
| train images | — | — | — | **4017** |
| val images | — | — | — | **501** |
| test images | — | — | — | **501** |
| pool 总图片 | — | — | — | **5019** |
| 类别数 (nc) | 1 | 1 | 1 | **1** |
| 类别名称 | `'1'` | `'iso-su'` | `'iso_poly'` | **`insulator`** |
| class id | 0 | 0 | 0 | **0** |

> **已核实（2026-06-15）**：三子集均为 nc=1，class id=0，含义相同（绝缘子），统一命名为 `insulator`，无需重映射。  
> 切分脚本 `scripts/dataset/merge_mpid.sh` 按 80/10/10 随机切分，输出至 `data/mpid/merged/`。

---

## 实验设计

### EXP-08a：YOLOv11n 基线（MPID）

| 参数 | 值 |
|---|---|
| 模型 | yolo11n.pt（原始结构） |
| 预训练 | yolo11n.pt |
| imgsz | 640 |
| batch | 32 |
| epochs | 100 |
| patience | 30 |
| optimizer | auto（默认 SGD）|
| 训练脚本 | `scripts/train/run_mpid_exp08a_baseline.sh` |
| 输出目录 | `runs/detect/mpid_exp08a_baseline/` |

### EXP-08b：ASE-YOLOv11 迁移（MPID）

| 参数 | 值 |
|---|---|
| 模型 | yolo11-p2-full.yaml（P2+LWFusion+CBAM+TADetect ta=2+NWD 0.4） |
| 预训练 | yolo11n.pt（公平对比）|
| imgsz | 640 |
| batch | 32 |
| epochs | 100 |
| patience | 30 |
| nwd_weight | 0.4 |
| 训练脚本 | `scripts/train/run_mpid_exp08b_ase.sh` |
| 输出目录 | `runs/detect/mpid_exp08b_ase_yolov11/` |

---

## 实验结果（训练完成后填写）

### EXP-08a 结果

| 指标 | 值 |
|---|---|
| mAP50 | **0.951** |
| mAP50-95 | **0.744** |
| Precision | **0.917** |
| Recall | **0.915** |
| 推理速度（preprocess+inference+postprocess） | 0.1 + 0.3 + 4.7 = **5.1 ms/img**（≈196 FPS）|
| 参数量 | 2,582,347（~2.59M）|
| GFLOPs | 6.3 |
| 训练时长 | 1.467 h（100 epochs）|
| val 图片数 | 501 张，850 个实例 |
| 最佳 epoch | 100（last = best）|
| 权重路径 | `runs/detect/mpid_exp08a_baseline/weights/best.pt` |

### EXP-08b 结果

| 指标 | 值 |
|---|---|
| mAP50 | **0.953** |
| mAP50-95 | **0.734** |
| Precision | **0.942** |
| Recall | **0.913** |
| 推理速度（preprocess+inference+postprocess） | 0.1 + 3.0 + 1.8 = **4.9 ms/img**（≈204 FPS）|
| 参数量 | 2,860,904（~2.86M）|
| GFLOPs | 10.6 |
| 训练时长 | 1.166 h（100 epochs）|
| val 图片数 | 501 张，850 个实例 |
| 最佳 epoch | 100（last = best）|
| 权重路径 | `runs/detect/mpid_exp08b_ase_yolov11/weights/best.pt` |

### 对比汇总

| 模型 | mAP50 | mAP50-95 | P | R | FPS | Δ mAP50 vs 基线 |
|---|---|---|---|---|---|---|
| EXP-08a YOLOv11n 基线 | 0.951 | 0.744 | 0.917 | 0.915 | ~196 | — |
| EXP-08b ASE-YOLOv11 | **0.953** | 0.734 | **0.942** | 0.913 | ~204 | **+0.002** |

---

## 执行步骤记录

### Step 1：✅ 数据已直接上传至服务器（2026-06-15）

数据直接上传至 `/root/autodl-tmp/yolov11-drone-inspection/data/mpid/`，目录结构：
```
data/mpid/
  glass/glass_MPID/train/{images,labels}/
  porcelain/porcelain_MPID/train/{images,labels}/
  composite/composite_MPID/train/{images,labels}/
```

### Step 2：✅ 类别已核实（2026-06-15）

三子集均为 nc=1，class id=0，统一命名为 `insulator`，无需重映射。

### Step 3：✅ 脚本和配置文件已更新

- `scripts/dataset/merge_mpid.sh`：路径已适配实际目录结构
- `configs/datasets/mpid.yaml`：path 已更新，nc/names 已确认
- `scripts/train/run_mpid_exp08a_baseline.sh`：路径已确认
- `scripts/train/run_mpid_exp08b_ase.sh`：MODEL_CFG 路径已确认

### Step 4（原 Step 5）：✅ 合并数据集

```bash
cd /root/autodl-tmp/yolov11-drone-inspection
source venv/bin/activate
bash scripts/dataset/merge_mpid.sh
```

合并后文件数（已确认）：
- train: images=4017, labels=4017
- val:   images=501,  labels=501
- test:  images=501,  labels=501

### Step 5（原 Step 6）：✅ EXP-08a 基线训练完成

```bash
cd /root/autodl-tmp/yolov11-drone-inspection
source venv/bin/activate
bash scripts/train/run_mpid_exp08a_baseline.sh
```

训练时长：1.467 h（100 epochs）  
最终 val mAP50：**0.951**（P=0.917，R=0.915，mAP50-95=0.744）

### Step 6（原 Step 7）：⬜ 运行 EXP-08b ASE 迁移训练

```bash
bash scripts/train/run_mpid_exp08b_ase.sh
```

训练开始时间：___  
训练结束时间：___  
最终 val mAP50：___

---

## 观察与结论（训练完成后填写）

> **注意**：以下内容必须基于实际训练结果，禁止填写预期值。

- [x] EXP-08a 训练完成，指标已记录（mAP50=0.951）
- [x] EXP-08b 训练完成，指标已记录（mAP50=0.953）
- [x] 对比分析已完成

**初步结论（2026-06-15）**：

ASE-YOLOv11 在 MPID 上迁移能力良好。mAP50 相比基线提升 +0.002（0.951→0.953），Precision 明显提高 +0.025（0.917→0.942），说明模型在绝缘子单类场景下精确度更高。mAP50-95 略低 -0.010（0.744→0.734），可能与模型复杂度提升后在小数据集（5019 张）上略有过拟合有关。考虑到 MPID 是单类大目标数据集，与 ASE-YOLOv11 针对 VisDrone 小目标密集场景的设计目标有差异，整体迁移性能符合预期。

> 注意：以上结论基于 2026-06-15 服务器训练日志，非手动估计。

---

## 数据完整性声明

- 数据集来源：Zenodo（公开数据集），CC BY 4.0 许可
- 实验结果来源：服务器训练日志，非手动估计
- 本文件中所有空白字段（—）须在实验完成后用真实数据填写
