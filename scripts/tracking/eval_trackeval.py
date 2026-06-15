#!/usr/bin/env python3
"""
eval_trackeval.py
使用 TrackEval 对 VisDrone-MOT 追踪结果进行评估。

评估指标：HOTA / MOTA / IDF1 / MT / ML / FP / FN / IDS / Frag

安装 TrackEval：
  pip install trackeval
  # 或从源码：
  git clone https://github.com/JonathonLuiten/TrackEval.git
  cd TrackEval && pip install -e .

用法：
  python scripts/tracking/eval_trackeval.py \
      --tracker_name exp09a_botsort \
      --results_dir  runs/track/exp09a_botsort/mot_results \
      --gt_dir       data/visdrone-mot/VisDrone2019-MOT-val/annotations \
      --output_dir   runs/track/exp09a_botsort/eval
"""

import argparse
import os
import sys
import json
import glob
from pathlib import Path


def build_trackeval_dataset(gt_dir: str, results_dir: str, tmp_dir: str, tracker_name: str) -> dict:
    """
    将 VisDrone-MOT GT 和结果整理为 TrackEval 期望的目录结构：
    tmp_dir/
      gt/
        MOT/
          <seq>/
            gt/gt.txt
            seqinfo.ini
      trackers/
        <tracker_name>/
          data/
            <seq>.txt
    """
    gt_out = os.path.join(tmp_dir, "gt", "MOT")
    tr_out = os.path.join(tmp_dir, "trackers", tracker_name, "data")
    os.makedirs(gt_out, exist_ok=True)
    os.makedirs(tr_out, exist_ok=True)

    seq_names = []
    for gt_file in sorted(glob.glob(os.path.join(gt_dir, "*.txt"))):
        seq = Path(gt_file).stem
        seq_names.append(seq)

        # 写 GT
        seq_gt_dir = os.path.join(gt_out, seq, "gt")
        os.makedirs(seq_gt_dir, exist_ok=True)
        # VisDrone GT 格式已是 MOT Challenge 格式，直接复制
        import shutil
        shutil.copy(gt_file, os.path.join(seq_gt_dir, "gt.txt"))

        # 写 seqinfo.ini（TrackEval 需要帧率和长度信息）
        with open(gt_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        n_frames = max(int(l.split(",")[0]) for l in lines) if lines else 1
        seqinfo = (
            f"[Sequence]\n"
            f"name={seq}\n"
            f"imDir=img1\n"
            f"frameRate=30\n"
            f"seqLength={n_frames}\n"
            f"imWidth=1920\n"
            f"imHeight=1080\n"
            f"imExt=.jpg\n"
        )
        with open(os.path.join(gt_out, seq, "seqinfo.ini"), "w") as f:
            f.write(seqinfo)

        # 复制追踪结果
        result_file = os.path.join(results_dir, f"{seq}.txt")
        if os.path.exists(result_file):
            shutil.copy(result_file, os.path.join(tr_out, f"{seq}.txt"))
        else:
            print(f"[WARN] 追踪结果文件不存在: {result_file}")

    return {"seq_names": seq_names, "gt_dir": gt_out, "trackers_dir": os.path.join(tmp_dir, "trackers")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker_name", required=True)
    parser.add_argument("--results_dir",  required=True, help="MOT Challenge 格式追踪结果目录")
    parser.add_argument("--gt_dir",       required=True, help="VisDrone-MOT GT annotations 目录")
    parser.add_argument("--output_dir",   required=True, help="评估输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_trackeval_tmp")

    info = build_trackeval_dataset(args.gt_dir, args.results_dir, tmp_dir, args.tracker_name)
    print(f"[INFO] 共 {len(info['seq_names'])} 个序列: {info['seq_names']}")

    try:
        import trackeval
    except ImportError:
        print("[ERROR] TrackEval 未安装，请运行: pip install trackeval")
        print("        或: git clone https://github.com/JonathonLuiten/TrackEval && pip install -e TrackEval/")
        sys.exit(1)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["OUTPUT_FOLDER"] = args.output_dir
    eval_config["TRACKERS_FOLDER"] = info["trackers_dir"]
    eval_config["GT_FOLDER"] = info["gt_dir"]
    eval_config["BENCHMARK"] = "MOT"
    eval_config["TRACKERS_TO_EVAL"] = [args.tracker_name]
    eval_config["METRICS"] = ["HOTA", "CLEAR", "Identity"]
    eval_config["USE_PARALLEL"] = False
    eval_config["NUM_PARALLEL_CORES"] = 1

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"]       = info["gt_dir"]
    dataset_config["TRACKERS_FOLDER"] = info["trackers_dir"]
    dataset_config["BENCHMARK"]       = "MOT"
    dataset_config["SPLIT_TO_EVAL"]   = "train"   # gt 子目录在 MOT/ 下
    dataset_config["SEQ_INFO"]        = {s: None for s in info["seq_names"]}
    dataset_config["DO_PREPROC"]      = False      # VisDrone 已是 MOT 格式，不需预处理

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    res, _ = evaluator.evaluate(dataset_list, metrics_list)

    # 简洁汇总
    summary = {}
    for seq, seq_res in res[args.tracker_name]["MOT"]["COMBINED_SEQ"].items():
        summary[seq] = {
            "HOTA":  round(float(seq_res.get("HOTA", {}).get("HOTA", [0])[0]) * 100, 2),
            "MOTA":  round(float(seq_res.get("CLEAR", {}).get("MOTA", 0)) * 100, 2),
            "IDF1":  round(float(seq_res.get("Identity", {}).get("IDF1", 0)) * 100, 2),
            "IDS":   int(seq_res.get("CLEAR", {}).get("IDSW", 0)),
            "Frag":  int(seq_res.get("CLEAR", {}).get("Frag", 0)),
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 评估完成，摘要已保存到: {summary_path}")

    # 打印总体结果
    overall = summary.get("COMBINED_SEQ", list(summary.values())[-1] if summary else {})
    print("\n===== EXP-09a 追踪评估结果 =====")
    for k, v in overall.items():
        print(f"  {k:8s}: {v}")
    print("=================================")


if __name__ == "__main__":
    main()
