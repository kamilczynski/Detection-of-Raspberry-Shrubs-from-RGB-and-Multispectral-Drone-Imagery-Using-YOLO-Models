import pandas as pd

# === ŚCIEŻKA DO TWOJEGO PLIKU ===
file_path = r"C:\Users\"

# === WCZYTANIE DANYCH ===
df = pd.read_excel(file_path)
df.columns = [col.strip() for col in df.columns]  # usunięcie ewentualnych spacji

# === ANALIZA RÓŻNIC TENSOR vs TORCH ===
results = []

for (modality, model), group in df.groupby(["Image modality", "Model"]):
    if "Torch" in group["Library"].values and "Tensor" in group["Library"].values:
        torch_row = group[group["Library"] == "Torch"].iloc[0]
        tensor_row = group[group["Library"] == "Tensor"].iloc[0]

        # Obliczenia różnic absolutnych
        fps_diff = tensor_row["FPS"] - torch_row["FPS"]
        latency_diff = tensor_row["Latency (ms)"] - torch_row["Latency (ms)"]
        map_fps_diff = tensor_row["mAP50:95/FPS"] - torch_row["mAP50:95/FPS"]

        # Obliczenia różnic procentowych
        fps_gain_pct = (fps_diff / torch_row["FPS"]) * 100
        latency_gain_pct = (latency_diff / torch_row["Latency (ms)"]) * 100
        map_fps_gain_pct = (map_fps_diff / torch_row["mAP50:95/FPS"]) * 100

        # Zapisanie wyników
        results.append({
            "Image modality": modality,
            "Model": model,
            "FPS (Torch)": torch_row["FPS"],
            "FPS (Tensor)": tensor_row["FPS"],
            "Δ FPS": fps_diff,
            "Δ FPS (%)": fps_gain_pct,
            "Latency (Torch)": torch_row["Latency (ms)"],
            "Latency (Tensor)": tensor_row["Latency (ms)"],
            "Δ Latency": latency_diff,
            "Δ Latency (%)": latency_gain_pct,
            "mAP50:95/FPS (Torch)": torch_row["mAP50:95/FPS"],
            "mAP50:95/FPS (Tensor)": tensor_row["mAP50:95/FPS"],
            "Δ mAP50:95/FPS": map_fps_diff,
            "Δ mAP50:95/FPS (%)": map_fps_gain_pct
        })

# === TWORZENIE NOWEGO ARKUSZA Z PORÓWNANIEM ===
df_comparison = pd.DataFrame(results)

# === ZAPIS DO EXCELA ===
output_path = r"C:\Users\topgu\Downloads\comparison_torch_tensor.xlsx"
df_comparison.to_excel(output_path, index=False)

print(f"\n✅ Wyniki zapisano do: {output_path}")
print("Zawiera różnice absolutne i procentowe dla FPS, Latency i mAP50:95/FPS.")
