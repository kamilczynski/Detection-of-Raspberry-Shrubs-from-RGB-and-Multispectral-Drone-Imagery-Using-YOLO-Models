from ultralytics import YOLO
import torch
import os
import csv
import yaml
import json

# === KONFIGURACJA ===
model_path = "C:/Users/HARDPC/runs/train_RGB8S/yolov8s_RGB8S/weights/best.pt"
data_yaml = "C:/Users/HARDPC/data_RGB.yaml"
save_dir = "C:/Users/HARDPC/Desktop/PROJEKTY CNN/WYNIKI/RZEDY/METRYKI/RGB/8S"

# === CUDA DETEKCJA ===
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("❌ Brak GPU! Walidacja wymaga urządzenia CUDA (np. RTX 5080 lub Jetson Orin NX).")

print(f"\n🔥 Uruchamiam ewaluację na TEST SET na urządzeniu: {torch.cuda.get_device_name(0)}")

os.makedirs(save_dir, exist_ok=True)

# === 1️⃣ Załaduj model ===
model = YOLO(model_path)

# === 2️⃣ Ewaluacja modelu NA TEST SET ===
print("\n🚀 Rozpoczynam ewaluację modelu na TEST SET...")
results = model.val(
    data=data_yaml,
    split="test",
    imgsz=640,
    device=device,
    save_json=True,      # 👈 wymusza zapis predictions.json
    save_hybrid=False,
    verbose=True,
    plots=True
)

# === 3️⃣ Zbierz i wypisz metryki ===
precision = results.box.p.mean()
recall = results.box.r.mean()
f1 = results.box.f1.mean()
map50 = results.box.map50
map5095 = results.box.map

print("\n=== METRYKI TEST SET ===")
print(f"Precision:    {precision:.3f}")
print(f"Recall:       {recall:.3f}")
print(f"F1-score:     {f1:.3f}")
print(f"mAP@0.5:      {map50:.3f}")
print(f"mAP@[.5:.95]: {map5095:.3f}")

# === 4️⃣ Zapis metryk do CSV ===
csv_path = os.path.join(save_dir, "metrics_summary_testset.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for name, val in [
        ("Precision", precision),
        ("Recall", recall),
        ("F1-score", f1),
        ("mAP@0.5", map50),
        ("mAP@[.5:.95]", map5095)
    ]:
        writer.writerow([name, f"{val:.4f}"])
print(f"\n📁 Wyniki metryk TEST SET zapisano do: {csv_path}")

# === 5️⃣ Wczytaj ścieżkę TEST SET z data.yaml ===
with open(data_yaml, 'r') as f:
    data_config = yaml.safe_load(f)
base_path = data_config.get('path', '')
test_split = data_config.get('test', None)
if not test_split:
    raise FileNotFoundError("❌ W pliku data.yaml nie zdefiniowano sekcji 'test:'.")

if os.path.isabs(test_split):
    test_dir = test_split
else:
    test_dir = os.path.join(base_path, test_split)
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ Nie znaleziono folderu test set: {test_dir}")
print(f"\n📂 Test set wykryty: {test_dir}")

# === 6️⃣ Detekcja i zapis obrazów TEST SET + JSON z wynikami ===
print("\n📸 Uruchamiam detekcję na TEST SET...")
pred_results = model.predict(
    source=test_dir,
    imgsz=640,
    conf=0.25,
    device=device,
    save=True,
    save_txt=False,
    project=save_dir,
    name="predicted_testset",
    exist_ok=True,
    verbose=True
)

# === 7️⃣ Zapisz wszystkie predykcje do JSON ===
json_path = os.path.join(save_dir, "predictions_testset.json")
json_data = []
for result in pred_results:
    boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes else []
    confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes else []
    classes = result.boxes.cls.cpu().numpy().tolist() if result.boxes else []
    json_data.append({
        "image": result.path,
        "boxes": boxes,
        "confidences": confs,
        "classes": classes
    })

with open(json_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"\n💾 Zapisano pełne wyniki detekcji TEST SET do pliku: {json_path}")
print(f"✅ Obrazy TEST SET z ramkami zapisano w: {os.path.join(save_dir, 'predicted_testset')}")
print("\n=== Ewaluacja TEST SET zakończona pomyślnie ===")
