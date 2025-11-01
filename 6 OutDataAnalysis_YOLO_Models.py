import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ===  MDPI STYLE SETTINGS ===
mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# === CONFIGURATION ===
xlsx_path = r"C:/Users/.xlsx"
save_dir = r"C:/Users/"
os.makedirs(save_dir, exist_ok=True)

# === 1️⃣ Load data ===
df = pd.read_excel(xlsx_path)
csv_path = os.path.join(save_dir, "cross_modality_results.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Data loaded ({len(df)} records) and saved as {csv_path}")

df.columns = [c.strip().replace(" ", "_").upper() for c in df.columns]
df.rename(columns={"MAP5095": "MAP_5095"}, inplace=True)

# === 2️⃣ Compute EI, SELF_MAP, CROSS_MAP ===
elasticity = []
models = df["MODEL"].unique()
modalities = df["TRAINING_DATA"].unique()

for model in models:
    model_df = df[df["MODEL"] == model]
    for train_mod in modalities:
        m_train = model_df[model_df["TRAINING_DATA"] == train_mod]
        self_map = m_train[m_train["TESTING_DATA"] == train_mod]["MAP_5095"].mean()
        if pd.isna(self_map) or self_map == 0:
            continue
        cross_vals = m_train[m_train["TESTING_DATA"] != train_mod]["MAP_5095"].mean()
        ei = cross_vals / self_map if self_map > 0 else np.nan
        elasticity.append({
            "MODEL": model,
            "TRAINING": train_mod,
            "EI": ei,
            "SELF_MAP": self_map,
            "CROSS_MAP": cross_vals
        })

ei_df = pd.DataFrame(elasticity)
ei_summary = ei_df.groupby("MODEL")["EI"].mean().reset_index()
ei_summary.rename(columns={"EI": "MEAN_EI"}, inplace=True)
ei_path = os.path.join(save_dir, "elasticity_index_summary.csv")
ei_df.to_csv(ei_path, index=False)
print(f"📈 EI results saved to: {ei_path}")

# === 3️⃣ Compute WCR and Stability ===
wcr_data = []
for model in models:
    sub = ei_df[ei_df["MODEL"] == model]
    wcr = sub["CROSS_MAP"].min() / sub["SELF_MAP"].mean()
    stability = sub["CROSS_MAP"].std()
    wcr_data.append({"MODEL": model, "WCR": wcr, "STABILITY": stability})

wcr_df = pd.DataFrame(wcr_data)
wcr_path = os.path.join(save_dir, "wcr_stability_summary.csv")
wcr_df.to_csv(wcr_path, index=False)
print(f"📊 WCR/Stability results saved to: {wcr_path}")

# === 4️⃣ Cross-modality heatmaps (MDPI unified look) ===
for model in models:
    pivot = df[df["MODEL"] == model].pivot_table(
        index="TRAINING_DATA", columns="TESTING_DATA", values="MAP_5095"
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=0.3, vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={
            'label': r"$mAP_{50:95}$",
            'shrink': 0.85,
            'pad': 0.02
        },
        annot_kws={'size': 10, 'color': 'black'}
    )

    plt.title("", pad=8)
    plt.xlabel("Testing modality", labelpad=8)
    plt.ylabel("Training modality", labelpad=8)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heatmap_{model}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"heatmap_{model}.png"), dpi=600, bbox_inches="tight")
    plt.close()

# === 5️⃣ Barplot: EI (average across models) ===
plt.figure(figsize=(6.5, 4.5))
sns.barplot(data=ei_summary, x="MODEL", y="MEAN_EI", palette="viridis")
plt.title("")
plt.ylabel("Relative mAP Retention")
plt.ylim(0, 1.0)
plt.xlabel("")
for i, v in enumerate(ei_summary["MEAN_EI"]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "EI_barplot.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_dir, "EI_barplot.png"), dpi=600, bbox_inches="tight")
plt.close()

# === 6️⃣ Barplot: Elasticity-related metrics (final MDPI-consistent version) ===
plt.figure(figsize=(6.8, 4.8))
palette = sns.color_palette("viridis", 3)

metric_labels = {
    "EI": "Relative $mAP$ Retention",
    "SELF_MAP": r"$mAP_{in}$",
    "CROSS_MAP": r"$mAP_{out}$"
}

melted = ei_df.melt(
    id_vars=["MODEL"],
    value_vars=["EI", "SELF_MAP", "CROSS_MAP"],
    var_name="Metric",
    value_name="Value"
)
melted["Metric"] = melted["Metric"].map(metric_labels)

sns.barplot(
    data=melted,
    x="MODEL",
    y="Value",
    hue="Metric",
    palette=palette,
    errorbar=None
)

plt.title("")
plt.ylabel("Value")
plt.xlabel(None)
plt.ylim(0, 1.05)

# Reduce bottom spacing between x-axis and model labels
plt.subplots_adjust(bottom=0.18)

# Values above bars
grouped = melted.groupby(["MODEL", "Metric"])["Value"].mean().reset_index()
unique_metrics = list(metric_labels.values())
bar_positions = np.linspace(-0.25, 0.25, len(unique_metrics))

for i, model in enumerate(models):
    for j, metric in enumerate(unique_metrics):
        val = grouped.loc[
            (grouped["MODEL"] == model) & (grouped["Metric"] == metric),
            "Value"
        ].values
        if len(val) > 0:
            plt.text(
                i + bar_positions[j],
                val[0] + 0.015,
                f"{val[0]:.3f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

# Compact legend directly below the plot
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "EI_SELF_CROSS_barplot.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_dir, "EI_SELF_CROSS_barplot.png"), dpi=600, bbox_inches="tight")
plt.close()

# === 7️⃣ Barplot: Robustness and Stability (final MDPI version) ===
plt.figure(figsize=(6.8, 4.8))
palette = sns.color_palette("viridis", 2)

metric_labels_wcr = {
    "WCR": "Worst-case ratio",
    "STABILITY": r"Cross-Domain Variability"
}

melted_wcr = wcr_df.melt(
    id_vars=["MODEL"],
    value_vars=["WCR", "STABILITY"],
    var_name="Metric",
    value_name="Value"
)
melted_wcr["Metric"] = melted_wcr["Metric"].map(metric_labels_wcr)

sns.barplot(
    data=melted_wcr,
    x="MODEL",
    y="Value",
    hue="Metric",
    palette=palette,
    errorbar=None
)

plt.title("")
plt.ylabel("Value")
plt.xlabel(None)
plt.ylim(0, 1.05)
plt.subplots_adjust(bottom=0.18)

# Values above bars
grouped_wcr = melted_wcr.groupby(["MODEL", "Metric"])["Value"].mean().reset_index()
unique_metrics_wcr = list(metric_labels_wcr.values())
bar_offsets = np.linspace(-0.17, 0.17, len(unique_metrics_wcr))

for i, model in enumerate(models):
    for j, metric in enumerate(unique_metrics_wcr):
        val = grouped_wcr.loc[
            (grouped_wcr["MODEL"] == model) & (grouped_wcr["Metric"] == metric),
            "Value"
        ].values
        if len(val) > 0:
            plt.text(
                i + bar_offsets[j],
                val[0] + 0.015,
                f"{val[0]:.3f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

# Legend below the chart
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "WCR_Stability_barplot.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_dir, "WCR_Stability_barplot.png"), dpi=600, bbox_inches="tight")
plt.close()

print("\n✅ Analysis completed!")
print("📂 The following files have been saved:")
print(" • Cross-modality heatmaps (PDF + PNG)")
print(" • Barplots for EI, WCR, Stability, Self/Cross Map (PDF + PNG)")
print(" • CSV summaries for EI, WCR, and Stability (MDPI-ready)\n")
