from ultralytics import YOLO
import torch, os, time, csv, sys

# === CONFIGURATION ===
model_path = "C:/Users/HARDPC/runs/train_RGB8S/yolov8s_RGB8S/weights/best.pt"  # path to model
save_dir = "C:/Users/HARDPC/Desktop/PROJECTS_CNN/RESULTS/ROWS/PERFORMANCE/RGB/8S"  # results directory
os.makedirs(save_dir, exist_ok=True)

# mAP@[.5:.95] value from TEST SET evaluation
map5095_value = 0.672  # ← enter your test set evaluation result here

# Test parameters
warmup_iter = 10      # number of warm-up iterations
test_iter = 100       # number of benchmark iterations
input_res = (640, 640)
input_size = f"{input_res[0]}x{input_res[1]}"

# === GPU DETECTION ===
if not torch.cuda.is_available():
    sys.exit("\n❌ No GPU (CUDA) detected. Performance test can only be run on RTX 5080 or Jetson Orin NX.\n")

device = "cuda"
device_name = torch.cuda.get_device_name(0)
print(f"\n⚙️ Detected device: {device_name}")

# === LOAD MODEL ===
model = YOLO(model_path)
model.to(device)

# --- MODEL PARAMETERS ---
params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n📊 Model parameters: {params_m:.2f} M | Size: {model_size_mb:.2f} MB")

# === GPU WARM-UP ===
print("\n🔥 Warming up GPU...")
dummy = torch.zeros((1, 3, *input_res)).to(device)
for _ in range(warmup_iter):
    _ = model(dummy)  # pure forward pass
torch.cuda.empty_cache()
print("✅ Warm-up completed.\n")

# === INFERENCE TIME MEASUREMENT ===
print("🚀 Measuring performance (pure forward pass)...")
torch.cuda.synchronize()
start = time.time()
for _ in range(test_iter):
    _ = model(dummy)
torch.cuda.synchronize()
end = time.time()

# === BASIC METRICS CALCULATION ===
total_time = end - start
fps = test_iter / total_time
latency_ms = 1000 / fps

# === ADDITIONAL METRICS ===
ap_fps_ratio = map5095_value / fps if fps > 0 else 0.0  # mAP/FPS ratio

# === DISPLAY RESULTS ===
print("\n=== PERFORMANCE RESULTS ===")
print(f"Model:              {os.path.basename(model_path)}")
print(f"Device:             {device_name}")
print(f"Input size:         {input_size}")
print(f"Params:             {params_m:.2f} M")
print(f"Model size:         {model_size_mb:.2f} MB")
print(f"FPS:                {fps:.2f}")
print(f"Latency:            {latency_ms:.2f} ms")
print(f"mAP@[.5:.95]/FPS:   {ap_fps_ratio:.4f}")

# === SAVE RESULTS TO CSV ===
csv_path = os.path.join(save_dir, "hardware_performance.csv")

# automatic precision mode detection
precision_mode = "FP16" if torch.cuda.get_device_capability()[0] >= 7 else "FP32"

if "rtx" in device_name.lower():
    platform_name = "RTX 5080"
elif "orin" in device_name.lower() or "jetson" in device_name.lower():
    platform_name = "reComputer J4012 (Jetson Orin NX)"
else:
    platform_name = device_name

# append results without overwriting previous entries
file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "Device", "Precision mode", "Input size",
            "Params (M)", "Model size (MB)",
            "FPS", "Latency (ms)", "mAP@[.5:.95]/FPS"
        ])
    writer.writerow([
        platform_name,
        precision_mode, input_size, f"{params_m:.2f}", f"{model_size_mb:.2f}",
        f"{fps:.2f}", f"{latency_ms:.2f}", f"{ap_fps_ratio:.4f}"
    ])

print(f"\n✅ Performance results saved to: {csv_path}")
print("=== Performance test completed successfully ===\n")
