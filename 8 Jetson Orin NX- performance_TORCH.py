from ultralytics import YOLO
import torch, os, time, csv, sys, subprocess

# === KONFIGURACJA ===
model_path = "/"  # ścieżka do modelu
save_dir = "/"  # katalog wyników
os.makedirs(save_dir, exist_ok=True)

# Wartość mAP@[.5:.95] z ewaluacji TEST SET
map5095_value = 0.672  # ← wpisz swój wynik z test set

# Parametry testu
warmup_iter = 10       # liczba iteracji rozgrzewających
test_iter = 100        # liczba iteracji pomiarowych
input_res = (640, 640)
input_size = f"{input_res[0]}x{input_res[1]}"

# === GPU DETEKCJA ===
if not torch.cuda.is_available():
    sys.exit("\n❌ Brak dostępnego GPU (CUDA). Test wydajności można uruchomić tylko na RTX 5080 lub Jetson Orin NX.\n")

device = "cuda"
device_name = torch.cuda.get_device_name(0)
print(f"\n⚙️ Wykryto urządzenie: {device_name}")

# === ZAŁADUJ MODEL ===
print(f"\n📦 Ładowanie modelu z: {model_path}")
model = YOLO(model_path)
model.to(device)

# --- PARAMETRY MODELU ---
params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n📊 Parametry modelu: {params_m:.2f} M | Rozmiar: {model_size_mb:.2f} MB")

# === WARM-UP GPU ===
print("\n🔥 Rozgrzewanie GPU...")
dummy = torch.zeros((1, 3, *input_res)).to(device)
for _ in range(warmup_iter):
    _ = model(dummy)
torch.cuda.empty_cache()
print("✅ Warm-up zakończony.\n")

# === POMIAR CZASU INFERENCJI ===
print("🚀 Pomiar wydajności (czysty forward pass)...")
torch.cuda.synchronize()
start = time.time()
for _ in range(test_iter):
    _ = model(dummy)
torch.cuda.synchronize()
end = time.time()

# === OBLICZENIA METRYK ===
total_time = end - start
fps = test_iter / total_time
latency_ms = 1000 / fps
ap_fps_ratio = map5095_value / fps if fps > 0 else 0.0

# === WYNIKI ===
print("\n=== WYNIKI WYDAJNOŚCIOWE ===")
print(f"Model: {os.path.basename(model_path)}")
print(f"Device: {device_name}")
print(f"Input size: {input_size}")
print(f"Params: {params_m:.2f} M")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"FPS: {fps:.2f}")
print(f"Latency: {latency_ms:.2f} ms")
print(f"mAP@[.5:.95]/FPS: {ap_fps_ratio:.4f}")

# === ZAPIS DO CSV ===
csv_path = os.path.join(save_dir, "hardware_performance.csv")

precision_mode = "FP16" if torch.cuda.get_device_capability()[0] >= 7 else "FP32"

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
            "Device", "Precision mode", "Input size", "Params (M)",
            "Model size (MB)", "FPS", "Latency (ms)", "mAP@[.5:.95]/FPS"
        ])
    writer.writerow([
        platform_name, precision_mode, input_size,
        f"{params_m:.2f}", f"{model_size_mb:.2f}", f"{fps:.2f}",
        f"{latency_ms:.2f}", f"{ap_fps_ratio:.4f}"
    ])

print(f"\n✅ Wyniki zapisano do: {csv_path}")
print("=== Test zakończony pomyślnie ===\n")
