from ultralytics import YOLO
import torch, os, time, csv, sys


model_path = "/"  
save_dir = "/"
os.makedirs(save_dir, exist_ok=True)

map5095_value = 0
warmup_iter = 10
test_iter = 100
input_res = (640, 640)
input_size = f"{input_res[0]}x{input_res[1]}"


if not torch.cuda.is_available():
    sys.exit("\n❌ Brak dostępnego GPU (CUDA)\n")

device_name = torch.cuda.get_device_name(0)
print(f"\n⚙️ Wykryto urządzenie: {device_name}")


print(f"\n📦 Ładowanie modelu TensorRT z: {model_path}")
model = YOLO(model_path)


model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n📊 Rozmiar pliku modelu: {model_size_mb:.2f} MB")


print("\n🔥 Rozgrzewanie TensorRT...")
dummy = torch.zeros((1, 3, *input_res)).to("cuda")
for _ in range(warmup_iter):
    _ = model(dummy)
torch.cuda.empty_cache()
print("✅ Warm-up zakończony.\n")


print("🚀 Pomiar wydajności TensorRT...")
torch.cuda.synchronize()
start = time.time()
for _ in range(test_iter):
    _ = model(dummy)
torch.cuda.synchronize()
end = time.time()

total_time = end - start
fps = test_iter / total_time
latency_ms = 1000 / fps
ap_fps_ratio = map5095_value / fps if fps > 0 else 0.0


print("\n=== WYNIKI WYDAJNOŚCIOWE (TensorRT) ===")
print(f"Model: {os.path.basename(model_path)}")
print(f"Device: {device_name}")
print(f"Input size: {input_size}")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"FPS: {fps:.2f}")
print(f"Latency: {latency_ms:.2f} ms")
print(f"mAP@[.5:.95]/FPS: {ap_fps_ratio:.4f}")


csv_path = os.path.join(save_dir, "hardware_performance_TENSOR.csv")

precision_mode = "FP16"
if "rtx" in device_name.lower():
    platform_name = "RTX 5080"
elif "orin" in device_name.lower() or "jetson" in device_name.lower():
    platform_name = "reComputer J4012 (Jetson Orin NX)"
else:
    platform_name = device_name

file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "Device", "Precision mode", "Input size",
            "Model size (MB)", "FPS", "Latency (ms)", "mAP@[.5:.95]/FPS"
        ])
    writer.writerow([
        platform_name, precision_mode, input_size,
        f"{model_size_mb:.2f}", f"{fps:.2f}",
        f"{latency_ms:.2f}", f"{ap_fps_ratio:.4f}"
    ])

print(f"\n✅ Wyniki zapisano do: {csv_path}")
print("=== Test TensorRT zakończony pomyślnie ===\n")

