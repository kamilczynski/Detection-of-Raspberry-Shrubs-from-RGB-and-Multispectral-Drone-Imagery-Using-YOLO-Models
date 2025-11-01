import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ===  MDPI STYLE ===
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

# === PATHS ===
input_path = r"C:/Users/topgu/Desktop/Splotowe Sieci Neuronowe/ELASTYCZNOSC/cross_modality_results.csv"
save_dir = r"C:/Users/topgu/Desktop/Splotowe Sieci Neuronowe/ELASTYCZNOSC/DANEWYNIKIMDPI3FINAL33"
os.makedirs(save_dir, exist_ok=True)

# === Load data ===
df = pd.read_csv(input_path, encoding="utf-8", on_bad_lines='skip')
print(f"✅ Data loaded ({len(df)} records)")

df.columns = df.columns.str.strip()
assert {'TRAINING DATA', 'TESTING DATA', 'Map5095'}.issubset(df.columns), "❌ Missing required columns!"

# === Aggregate by modalities ===
df_grouped = df.groupby(['TRAINING DATA', 'TESTING DATA'], as_index=False)['Map5095'].mean()
pivot = df_grouped.pivot(index='TRAINING DATA', columns='TESTING DATA', values='Map5095')
pivot.to_csv(os.path.join(save_dir, "modality_cross_performance_matrix.csv"), float_format="%.4f")

# === CROSS-MODALITY HEATMAP (MDPI unified look) ===
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
plt.savefig(os.path.join(save_dir, "heatmap_modalities.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_dir, "heatmap_modalities.png"), dpi=600, bbox_inches="tight")
plt.close()

# === Compute EI, WCR, Stability ===
ei_list = []
for train_modality in df['TRAINING DATA'].unique():
    subset = df[df['TRAINING DATA'] == train_modality]
    self_map = subset.loc[subset['TESTING DATA'] == train_modality, 'Map5095'].mean()
    cross_maps = subset.loc[subset['TESTING DATA'] != train_modality, 'Map5095']
    cross_mean = cross_maps.mean()
    cross_min = cross_maps.min()
    cross_std = cross_maps.std()

    ei = cross_mean / self_map if self_map else np.nan
    wcr = cross_min / self_map if self_map else np.nan

    ei_list.append({
        "TRAINING_MODALITY": train_modality,
        "SELF_MAP": self_map,
        "CROSS_MEAN": cross_mean,
        "EI": ei,
        "WCR": wcr,
        "STABILITY": cross_std
    })

ei_df = pd.DataFrame(ei_list).sort_values("EI", ascending=False)
ei_df.to_csv(os.path.join(save_dir, "elasticity_modalities_summary.csv"), index=False, float_format="%.4f")

# ===  Plot 1: Elasticity-related metrics (EI, mAPtrain, mAPcross) ===
plt.figure(figsize=(8.0, 5.0))  # slightly wider and taller
palette = sns.color_palette("viridis", 3)

metric_labels = {
    "EI": r"Relative $mAP$ Retention",
    "SELF_MAP": r"$mAP_{in}$",
    "CROSS_MEAN": r"$mAP_{out}$"
}

melted = ei_df.melt(
    id_vars=["TRAINING_MODALITY"],
    value_vars=["EI", "SELF_MAP", "CROSS_MEAN"],
    var_name="Metric",
    value_name="Value"
)
melted["Metric"] = melted["Metric"].map(metric_labels)

ax = sns.barplot(
    data=melted,
    x="TRAINING_MODALITY",
    y="Value",
    hue="Metric",
    palette=palette,
    errorbar=None,
    width=0.90,
    dodge=True
)

# precise value labels above each bar — no manual offsets
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", label_type="edge", padding=3, fontsize=9)

plt.title("")
plt.ylabel("Value")
plt.xlabel(None)
plt.ylim(0, 1.05)
ax.margins(y=0.05)

# Compact legend below the chart
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "EI_MODALITIES_barplot.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_dir, "EI_MODALITIES_barplot.png"), dpi=600, bbox_inches="tight")
plt.close()

# ===============================================================
#   PLOT 2: Robustness and stability (WCR + σ(mAPcross))
# ===============================================================

plt.figure(figsize=(7.5, 5.0))
palette = sns.color_palette("viridis", 2)

metric_labels_wcr = {
    "WCR": "Worst-case ratio",
    "STABILITY": r"Cross-Domain Variability"
}

melted_wcr = ei_df.melt(
    id_vars=["TRAINING_MODALITY"],
    value_vars=["WCR", "STABILITY"],
    var_name="Metric",
    value_name="Value"
)
melted_wcr["Metric"] = melted_wcr["Metric"].map(metric_labels_wcr)

ax = sns.barplot(
    data=melted_wcr,
    x="TRAINING_MODALITY",
    y="Value",
    hue="Metric",
    palette=palette,
    errorbar=None
)

# widen bars slightly
for bar in ax.patches:
    bar.set_width(bar.get_width() * 1.25)

plt.title("")
plt.ylabel("Value")
plt.xlabel(None)
plt.ylim(0, 1.05)
plt.subplots_adjust(bottom=0.18)

# Values above bars
grouped_wcr = melted_wcr.groupby(["TRAINING_MODALITY", "Metric"])["Value"].mean().reset_index()
unique_metrics_wcr = list(metric_labels_wcr.values())
bar_offsets = np.linspace(-0.19, 0.19, len(unique_metrics_wcr))
modalities = ei_df["TRAINING_MODALITY"].unique()

for i, modality in enumerate(modalities):
    for j, metric in enumerate(unique_metrics_wcr):
        val = grouped_wcr.loc[
            (grouped_wcr["TRAINING_MODALITY"] == modality) & (grouped_wcr["Metric"] == metric),
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

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "WCR_Stability_MODALITIES_barplot.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_dir, "WCR_Stability_MODALITIES_barplot.png"), dpi=600, bbox_inches="tight")
plt.close()

# ===============================================================
#   PLOT 2 (duplicate): Robustness and stability (WCR + σ(mAPcross))
# ===============================================================

plt.figure(figsize=(6.8, 4.8))
palette = sns.color_palette("viridis", 2)

metric_labels_wcr = {
    "WCR": "Worst-case ratio",
    "STABILITY": r"Cross-Domain Variability"
}

melted_wcr = ei_df.melt(
    id_vars=["TRAINING_MODALITY"],
    value_vars=["WCR", "STABILITY"],
    var_name="Metric",
    value_name="Value"
)
melted_wcr["Metric"] = melted_wcr["Metric"].map(metric_labels_wcr)

sns.barplot(
    data=melted_wcr,
    x="TRAINING_MODALITY",
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

# Value labels above bars (fixed offset = 0.015)
grouped_wcr = melted_wcr.groupby(["TRAINING_MODALITY", "Metric"])["Value"].mean().reset_index()
unique_metrics_wcr = list(metric_labels_wcr.values())
bar_offsets = np.linspace(-0.17, 0.17, len(unique_metrics_wcr))
modalities = ei_df["TRAINING_MODALITY"].unique()

for i, modality in enumerate(modalities):
    for j, metric in enumerate(unique_metrics_wcr):
        val = grouped_wcr.loc[
            (grouped_wcr["TRAINING_MODALITY"] == modality) & (grouped_wcr["Metric"] == metric),
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

# Legend below chart
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "WCR_Stability_MODALITIES_barplot.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_dir, "WCR_Stability_MODALITIES_barplot.png"), dpi=600, bbox_inches="tight")
plt.close()

print("\n✅ Data-centric analysis completed! Results saved in MDPI-ready format ✅")
