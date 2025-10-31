# -*- coding: utf-8 -*-
"""
YOLO Multi-Model Benchmark — Laboratory Mode (FP16, CUDA)
----------------------------------------------------------
- Split into sessions of 3 models each
- Cooldown between sessions
- GPU clock stabilization
- Full logs and result saving per session + master file
"""

import os
import sys
import time
import datetime
import statistics as st
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
import traceback
from itertools import islice


# ===================== USER CONFIG =====================
MODELS = [
    {"name": "YOLOv8s_RGB8S", "path": r"C:\Users\HARDPC\runs\train_RGB8S\yolov8s_RGB8S\weights\best.pt", "map5095": 0.975},
    {"name": "YOLOv11s_RGBG11S", "path": r"C:\Users\HARDPC\runs\train_RGB11S\yolov11s_RGB11S\weights\best.pt", "map5095": 0.975},
    {"name": "YOLOv12s_RGBG12S", "path": r"C:\Users\HARDPC\runs\train_RGB12S\yolov12s_RGB12S\weights\best.pt", "map5095": 0.973},
    {"name": "YOLOv8s_NIR8S", "path": r"C:\Users\HARDPC\runs\train_NIR8S\yolov8s_NIR8S\weights\best.pt", "map5095": 0.984},
    {"name": "YOLOv11s_NIR11S", "path": r"C:\Users\HARDPC\runs\train_NIR11S\yolov11s_NIR11S\weights\best.pt", "map5095": 0.986},
    {"name": "YOLOv12s_NIR12S", "path": r"C:\Users\HARDPC\runs\train_NIR12S\yolov12s_NIR12S\weights\best.pt", "map5095": 0.986},
    {"name": "YOLOv8s_RE8S", "path": r"C:\Users\HARDPC\runs\train_RE8S\yolov8s_RE8S\weights\best.pt", "map5095": 0.985},
    {"name": "YOLOv11s_RE11S", "path": r"C:\Users\HARDPC\runs\train_RE11S\yolov11s_RE11S\weights\best.pt", "map5095": 0.985},
    {"name": "YOLOv12s_RE12S", "path": r"C:\Users\HARDPC\runs\train_RE12S\yolov12s_RE12S\weights\best.pt", "map5095": 0.983},
    {"name": "YOLOv8s_R8S", "path": r"C:\Users\HARDPC\runs\train_R8S\yolov8s_R8S\weights\best.pt", "map5095": 0.941},
    {"name": "YOLOv11s_R11S", "path": r"C:\Users\HARDPC\runs\train_R11S\yolov11s_R11S\weights\best.pt", "map5095": 0.949},
    {"name": "YOLOv12s_R12S", "path": r"C:\Users\HARDPC\runs\train_R12S\yolov12s_R12S\weights\best.pt", "map5095": 0.952},
    {"name": "YOLOv8s_G8S", "path": r"C:\Users\HARDPC\runs\train_G8S\yolov8s_G8S\weights\best.pt", "map5095": 0.973},
    {"name": "YOLOv11s_G11S", "path": r"C:\Users\HARDPC\runs\train_G11S\yolov11s_G11S\weights\best.pt", "map5095": 0.974},
    {"name": "YOLOv12s_G12S", "path": r"C:\Users\HARDPC\runs\train_G12S\yolov12s_G12S\weights\best.pt", "map5095": 0.972},
]

SAVE_DIR = r"C:\Users\HARDPC\Desktop\PROJEKTY CNN\PERFOMANCERZEDY"
INPUT_RES = (640, 640)
WARMUP_IT = 50
TIMED_IT = 200
NUM_RUNS = 5
FORCE_FP16 = True
SESSION_COOLDOWN = 90  # seconds between sessions
# =======================================================


def ensure_cuda_or_exit():
    if not torch.cuda.is_available():
        sys.exit("❌ No CUDA GPU detected.")
    torch.cuda.init()


def use_fp16_flag():
    if not (FORCE_FP16 and torch.cuda.is_available()):
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 7


def make_dummy(res, device="cuda"):
    return torch.zeros((1, 3, res[0], res[1]), dtype=torch.float32, device=device)


def stabilize_gpu(model, dummy, seconds=5):
    print(f"⚙️ Stabilizing GPU clocks ({seconds}s)...")
    end = time.time() + seconds
    with torch.inference_mode():
        while time.time() < end:
            _ = model(dummy, half=True, verbose=False)
    torch.cuda.synchronize()


def sanity_check_gpu():
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    used = torch.cuda.memory_allocated()
    if used / total > 0.9:
        print("⚠️ GPU memory almost full — clearing cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)


def cuda_timed_forward(model, dummy, warmup, iters, half_infer):
    fps_list = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy, half=half_infer, verbose=False)
        torch.cuda.synchronize()

        for _ in range(iters):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(dummy, half=half_infer, verbose=False)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            fps_list.append(1000.0 / total_ms)

    fps_array = np.array(fps_list)
    trimmed = fps_array[(fps_array > np.percentile(fps_array, 10)) &
                        (fps_array < np.percentile(fps_array, 90))]
    fps_mean = np.mean(trimmed)
    latency_ms = 1000.0 / fps_mean
    jitter = np.std(trimmed) / fps_mean * 100
    return fps_mean, latency_ms, jitter


def summarize(vals):
    s = list(map(float, vals))
    q1, q3 = pd.Series(s).quantile([0.25, 0.75])
    return {
        "mean": float(st.mean(s)),
        "std": float(st.pstdev(s)) if len(s) > 1 else 0.0,
        "median": float(st.median(s)),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(min(s)), "max": float(max(s)),
    }


def benchmark_model(model_info, device, half_infer=True):
    name = model_info["name"]
    path = model_info["path"]
    mapv = model_info["map5095"]
    print(f"\n🚀 Benchmarking {name}  ({os.path.basename(path)})")

    sanity_check_gpu()
    model = YOLO(path).to(device)
    dummy = make_dummy(INPUT_RES, device)
    stabilize_gpu(model, dummy)

    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    size_mb = os.path.getsize(path) / (1024 * 1024)
    runs = []

    for i in range(1, NUM_RUNS + 1):
        fps, lat, jitter = cuda_timed_forward(model, dummy, WARMUP_IT, TIMED_IT, half_infer)
        ratio = mapv / fps if fps > 0 else float("nan")
        runs.append({"run": i, "fps": fps, "latency_ms": lat, "jitter_percent": jitter, "map50_95_over_fps": ratio})
        print(f"Run {i:02d}: FPS={fps:.2f} | Lat={lat:.2f}ms | Jitter={jitter:.2f}%")

    df_runs = pd.DataFrame(runs)
    df_runs.insert(0, "model", name)
    df_summary = pd.DataFrame([
        {"metric": "FPS", **summarize(df_runs["fps"])},
        {"metric": "Latency (ms)", **summarize(df_runs["latency_ms"])},
        {"metric": "Jitter (%)", **summarize(df_runs["jitter_percent"])},
    ])
    df_summary.insert(0, "model", name)
    return df_runs, df_summary


def chunk_models(models, n):
    """Split the model list into batches of n."""
    it = iter(models)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def run_session(session_id, models, device, half_infer):
    print(f"\n🧪 --- Session {session_id} --- ({len(models)} models) ---")
    all_runs, all_summaries = [], []
    for m in models:
        try:
            df_r, df_s = benchmark_model(m, device, half_infer)
            all_runs.append(df_r)
            all_summaries.append(df_s)
        except Exception as e:
            print(f"❗ Skipped {m['name']}: {e}")
            traceback.print_exc()

    df_all_runs = pd.concat(all_runs, ignore_index=True)
    df_all_summaries = pd.concat(all_summaries, ignore_index=True)
    save_path = os.path.join(SAVE_DIR, f"session_{session_id}_{datetime.datetime.now():%Y%m%d_%H%M}.xlsx")
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df_all_runs.to_excel(writer, sheet_name="runs", index=False)
        df_all_summaries.to_excel(writer, sheet_name="summary", index=False)
    print(f"✅ Session {session_id} saved: {save_path}")
    return df_all_runs, df_all_summaries


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    ensure_cuda_or_exit()
    device = "cuda"
    half_infer = use_fp16_flag()

    all_runs_global, all_summaries_global = [], []
    session_id = 1

    for models_chunk in chunk_models(MODELS, 3):
        df_r, df_s = run_session(session_id, models_chunk, device, half_infer)
        all_runs_global.append(df_r)
        all_summaries_global.append(df_s)
        print(f"🕒 Cooldown {SESSION_COOLDOWN}s before next session...")
        time.sleep(SESSION_COOLDOWN)
        session_id += 1

    df_global_runs = pd.concat(all_runs_global, ignore_index=True)
    df_global_summaries = pd.concat(all_summaries_global, ignore_index=True)
    master_path = os.path.join(SAVE_DIR, f"benchmark_master_{datetime.datetime.now():%Y%m%d_%H%M}.xlsx")
    with pd.ExcelWriter(master_path, engine="openpyxl") as writer:
        df_global_runs.to_excel(writer, sheet_name="runs", index=False)
        df_global_summaries.to_excel(writer, sheet_name="summary", index=False)
    print(f"\n🏁 All sessions completed. Master file saved: {master_path}")


if __name__ == "__main__":
    main()
