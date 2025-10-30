# -*- coding: utf-8 -*-
"""
Multi-Model Benchmark (RTX & Jetson-ready, FP16-aware)
------------------------------------------------------
- Wymusza FP16 przy inferencji (half=True) bez ręcznego model.half()
- Działa na RTX oraz Jetson Orin NX
- Zapisuje: runs, summary, metadata (minimum potrzebne do publikacji)
"""

import os
import sys
import datetime
import statistics as st
import pandas as pd
import torch
from ultralytics import YOLO
import traceback

# ===================== USER CONFIG =====================

MODELS = [
    {"name": "YOLOv8s_RGB8S",   "path": r"C:\Users\HARDPC\runs\train_RGB8S\yolov8s_RGB8S\weights\best.pt",  "map5095": 0.975},
    {"name": "YOLOv11s_RGBG11S","path": r"C:\Users\HARDPC\runs\train_RGB11S\yolov11s_RGB11S\weights\best.pt","map5095": 0.975},
    {"name": "YOLOv12s_RGBG12S","path": r"C:\Users\HARDPC\runs\train_RGB12S\yolov12s_RGB12S\weights\best.pt","map5095": 0.973},
    {"name": "YOLOv8s_NIR8S",   "path": r"C:\Users\HARDPC\runs\train_NIR8S\yolov8s_NIR8S\weights\best.pt",  "map5095": 0.984},
    {"name": "YOLOv11s_NIR11S", "path": r"C:\Users\HARDPC\runs\train_NIR11S\yolov11s_NIR11S\weights\best.pt","map5095": 0.986},
    {"name": "YOLOv12s_NIR12S", "path": r"C:\Users\HARDPC\runs\train_NIR12S\yolov12s_NIR12S\weights\best.pt","map5095": 0.986},
    {"name": "YOLOv8s_RE8S",    "path": r"C:\Users\HARDPC\runs\train_RE8S\yolov8s_RE8S\weights\best.pt",    "map5095": 0.985},
    {"name": "YOLOv11s_RE11S",  "path": r"C:\Users\HARDPC\runs\train_RE11S\yolov11s_RE11S\weights\best.pt", "map5095": 0.985},
    {"name": "YOLOv12s_RE12S",  "path": r"C:\Users\HARDPC\runs\train_RE12S\yolov12s_RE12S\weights\best.pt", "map5095": 0.983},
    {"name": "YOLOv8s_R8S",     "path": r"C:\Users\HARDPC\runs\train_R8S\yolov8s_R8S\weights\best.pt",      "map5095": 0.941},
    {"name": "YOLOv11s_R11S",   "path": r"C:\Users\HARDPC\runs\train_R11S\yolov11s_R11S\weights\best.pt",   "map5095": 0.949},
    {"name": "YOLOv12s_R12S",   "path": r"C:\Users\HARDPC\runs\train_R12S\yolov12s_R12S\weights\best.pt",   "map5095": 0.952},
    {"name": "YOLOv8s_G8S",     "path": r"C:\Users\HARDPC\runs\train_G8S\yolov8s_G8S\weights\best.pt",      "map5095": 0.973},
    {"name": "YOLOv11s_G11S",   "path": r"C:\Users\HARDPC\runs\train_G11S\yolov11s_G11S\weights\best.pt",   "map5095": 0.974},
    {"name": "YOLOv12s_G12S",   "path": r"C:\Users\HARDPC\runs\train_G12S\yolov12s_G12S\weights\best.pt",   "map5095": 0.972},
]

SAVE_DIR   = r"C:\Users\HARDPC\Desktop\PROJEKTY CNN\PERFOMANCERZEDY"
INPUT_RES  = (640, 640)
WARMUP_IT  = 10
TIMED_IT   = 100
NUM_RUNS   = 5
FORCE_FP16 = True  # możesz przełączyć na False dla porównania FP32
# =======================================================


def ensure_cuda_or_exit():
    if not torch.cuda.is_available():
        sys.exit("❌ No CUDA GPU detected.")
    torch.cuda.init()


def use_fp16_flag():
    """FP16 tylko gdy GPU i framework wspierają (na Orin NX/RTX będzie True)."""
    if not (FORCE_FP16 and torch.cuda.is_available()):
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 7  # Volta/Ampere/Hooper — OK dla FP16


def load_model(model_path: str, device: str = "cuda"):
    return YOLO(model_path).to(device)


def make_dummy(input_res, device="cuda"):
    return torch.zeros((1, 3, input_res[0], input_res[1]), dtype=torch.float32, device=device)


def cuda_timed_forward(model, dummy, warmup=10, iters=100, half_infer=True):
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy, half=half_infer, verbose=False)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = model(dummy, half=half_infer, verbose=False)
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        fps = iters / (total_ms / 1000.0)
        latency_ms = 1000.0 / fps
        return fps, latency_ms


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
    model = load_model(path, device)
    dummy = make_dummy(INPUT_RES, device)

    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    size_mb  = os.path.getsize(path) / (1024 * 1024)
    precision_mode = "FP16" if half_infer else "FP32"

    runs = []
    for i in range(1, NUM_RUNS + 1):
        fps, lat = cuda_timed_forward(model, dummy, warmup=WARMUP_IT, iters=TIMED_IT, half_infer=half_infer)
        ratio = mapv / fps if fps > 0 else float("nan")
        runs.append({"run": i, "fps": fps, "latency_ms": lat, "map50_95_over_fps": ratio})
        print(f"Run {i:02d}: FPS={fps:.2f} | Latency={lat:.2f} ms | mAP50:95/FPS={ratio:.4f}")

    df_runs = pd.DataFrame(runs)
    df_runs.insert(0, "model", name)
    df_runs.insert(1, "precision", precision_mode)
    df_runs.insert(2, "params_M", f"{params_m:.2f}")
    df_runs.insert(3, "size_MB", f"{size_mb:.2f}")

    df_summary = pd.DataFrame([
        {"metric": "FPS",           **summarize(df_runs["fps"])},
        {"metric": "Latency (ms)",  **summarize(df_runs["latency_ms"])},
        {"metric": "mAP50:95/FPS",  **summarize(df_runs["map50_95_over_fps"])},
    ])
    df_summary.insert(0, "model", name)

    return df_runs, df_summary


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    ensure_cuda_or_exit()

    device = "cuda"
    half_infer = use_fp16_flag()
    print(f"⚙️  Device: {torch.cuda.get_device_name(0)} | Precision: {'FP16' if half_infer else 'FP32'}")

    all_runs, all_summaries = [], []
    for m in MODELS:
        try:
            df_r, df_s = benchmark_model(m, device, half_infer=half_infer)
            all_runs.append(df_r)
            all_summaries.append(df_s)
        except Exception as e:
            print(f"❗ Pominięto {m['name']} z powodu błędu: {e}")
        finally:
            torch.cuda.empty_cache()

    # --- zapis wyników ---
    if not all_runs:
        print("⚠️  Nie wygenerowano żadnych danych — brak poprawnych modeli.")
        return

    print(f"\n💡 Zapisuję dane dla {len(all_runs)} modeli...")
    df_all_runs = pd.concat(all_runs, ignore_index=True)
    df_all_summaries = pd.concat(all_summaries, ignore_index=True)

    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": "FP16" if half_infer else "FP32",
        "input_size": f"{INPUT_RES[0]}x{INPUT_RES[1]}",
        "warmup_iters": WARMUP_IT,
        "timed_iters": TIMED_IT,
        "runs_per_model": NUM_RUNS,
        "num_models": len(MODELS),
    }
    df_meta = pd.DataFrame(list(meta.items()), columns=["key", "value"])

    xlsx_path = os.path.join(SAVE_DIR, f"multi_model_performance_{datetime.datetime.now():%Y%m%d_%H%M}.xlsx")
    print(f"💾 Próba zapisu pliku: {xlsx_path}")

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_all_runs.to_excel(writer, sheet_name="runs", index=False)
            df_all_summaries.to_excel(writer, sheet_name="summary", index=False)
            df_meta.to_excel(writer, sheet_name="metadata", index=False)
        print(f"✅ Wyniki zapisano: {xlsx_path}")
    except Exception as e:
        print("❌ Błąd przy zapisie Excela:")
        traceback.print_exc()

    print("🏁 Benchmark zakończony.\n")


if __name__ == "__main__":
    main()
