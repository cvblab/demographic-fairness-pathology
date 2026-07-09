"""
Figure for Medical Image Analysis: 
  (a) Demographic leakage from fused representations (|Probe AUROC - 0.5|)
  (b) Zero-shot cross-cohort transfer fairness (Delta EO Sex & Age)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Journal-quality styling ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
})

# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = {
    'Vision-Only':              '#4C72B0',
    'GroupDRO (Multimodal)':    '#DD8452',
    'Adv Debiasing (Multimodal)': '#55A868',
    'Adv Align (deterministic)': '#8172B3',
    'Adv Align (diagonal)':     '#937860',
    'Adv Align (low_rank)':     '#DA8BC3',
}

HATCHES = {
    'Vision-Only':              '',
    'GroupDRO (Multimodal)':    '',
    'Adv Debiasing (Multimodal)': '',
    'Adv Align (deterministic)': '',
    'Adv Align (diagonal)':     '',
    'Adv Align (low_rank)':     '',
}

MODELS_LEAKAGE = [
    'Vision-Only',
    'GroupDRO (Multimodal)',
    'Adv Debiasing (Multimodal)',
    'Adv Align (deterministic)',
    'Adv Align (diagonal)',
    'Adv Align (low_rank)',
]

SHORT_LABELS = {
    'Vision-Only':              'Vis-Only',
    'GroupDRO (Multimodal)':    'GroupDRO',
    'Adv Debiasing (Multimodal)': 'Adv Debias',
    'Adv Align (deterministic)': 'AdvAlign\n(det)',
    'Adv Align (diagonal)':     'AdvAlign\n(diag)',
    'Adv Align (low_rank)':     'AdvAlign\n(LR)',
}

DATASETS = ['AI4SKIN', 'ASSIST', 'TCGA-SKCM']

# ── Load leakage data ───────────────────────────────────────────────────────
leakage = {}
for ds in DATASETS:
    path = f'/workspace/src/FOMENT/step2_results_{ds}/leakage.csv'
    df = pd.read_csv(path)
    leakage[ds] = df

# ── Load transfer data ──────────────────────────────────────────────────────
transfer_df = pd.read_csv('/workspace/src/FOMENT/step3_cross_cohort_results/transfer_metrics_full.csv')

# Parse out numeric values from strings like "**0.964 (0.910 - 0.998)**"
def parse_val(s):
    s = str(s).replace('*', '').strip()
    # Take the first number before any parenthesis
    parts = s.split('(')
    return float(parts[0].strip())

# Split transfer_df into AI4SKIN (rows 0-4) and ASSIST (rows 5-9)
transfer_ai4skin = transfer_df.iloc[:5].copy()
transfer_assist  = transfer_df.iloc[5:].copy()

# ── Create figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

# ═══════════════════════════════════════════════════════════════════════════
# (a) Leakage subplot — grouped bar chart, 3 dataset groups
# ═══════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])

n_models = len(MODELS_LEAKAGE)
n_ds = len(DATASETS)
bar_width = 0.11
group_width = n_models * bar_width
x_groups = np.arange(n_ds) * (group_width + 0.25)

for j, model in enumerate(MODELS_LEAKAGE):
    vals = []
    for ds in DATASETS:
        row = leakage[ds][leakage[ds]['Model'] == model]
        if len(row) == 0:
            vals.append(0)
        else:
            probe_inter = row['Probe Intersectional'].values[0]
            vals.append(abs(probe_inter - 0.5))
    
    positions = x_groups + j * bar_width
    bars = ax_a.bar(positions, vals, width=bar_width * 0.9,
                    color=COLORS[model], edgecolor='black', linewidth=0.5,
                    hatch=HATCHES[model],
                    label=SHORT_LABELS[model])

# X-axis
ax_a.set_xticks(x_groups + group_width / 2 - bar_width / 2)
ax_a.set_xticklabels(DATASETS, fontsize=11, fontweight='bold')
ax_a.set_ylabel('|Probe AUROC − 0.5|  (↓ better)', fontsize=12)
ax_a.set_title('(a) Demographic leakage (intersectional probe)', fontsize=13, fontweight='bold', pad=10)
ax_a.axhline(y=0, color='black', linewidth=0.6)
ax_a.set_ylim(0, 0.22)
ax_a.yaxis.grid(True, alpha=0.3)
ax_a.set_axisbelow(True)

# Legend inside subplot (a)
ax_a.legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=8.5,
            handlelength=1.5, handletextpad=0.4, columnspacing=0.8)

# ═══════════════════════════════════════════════════════════════════════════
# (b) Transfer fairness subplot — grouped bars for ΔEO (Sex) and ΔEO (Age)
# ═══════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])

TRANSFER_MODELS = [
    'Vision-Only (Linear)',
    'Transfer Adv Align (Real+Vis)',
]

TRANSFER_COLORS = {
    'Vision-Only (Linear)':            '#4C72B0',
    'Transfer Adv Align (Real+Vis)':   '#8172B3',
}

TRANSFER_SHORT = {
    'Vision-Only (Linear)':            'Vis-Only',
    'Transfer Adv Align (Real+Vis)':   'AdvAlign (LR) (Real)',
}

# Datasets for transfer: AI4SKIN and ASSIST
transfer_data = {'AI4SKIN': transfer_ai4skin, 'ASSIST': transfer_assist}

# Structure: for each dataset, for each model, get DeltaEO (Sex) and DeltaEO (Age)
n_tm = len(TRANSFER_MODELS)

bar_w = 0.18
# x positions: 4 groups (AI4SKIN-Sex, AI4SKIN-Age, ASSIST-Sex, ASSIST-Age)
group_labels = ['AI4SKIN\nΔEO Sex', 'AI4SKIN\nΔEO Age', 'ASSIST\nΔEO Sex', 'ASSIST\nΔEO Age']
x_pos = np.array([0, 0.55, 1.3, 1.85])

for j, model in enumerate(TRANSFER_MODELS):
    vals = []
    for ds_name, ds_df in transfer_data.items():
        row = ds_df[ds_df['Model'] == model]
        if len(row) == 0:
            vals.extend([0, 0])
        else:
            deo_sex = parse_val(row['DeltaEO (Sex)'].values[0])
            deo_age = parse_val(row['DeltaEO (Age)'].values[0])
            vals.extend([deo_sex, deo_age])
    
    positions = x_pos + j * bar_w
    bars = ax_b.bar(positions, vals, width=bar_w * 0.85,
                    color=TRANSFER_COLORS[model], edgecolor='black', linewidth=0.5,
                    label=TRANSFER_SHORT[model])

ax_b.set_xticks(x_pos + (n_tm - 1) * bar_w / 2)
ax_b.set_xticklabels(group_labels, fontsize=9, fontweight='bold')
ax_b.set_ylabel('ΔEO  (↓ better)', fontsize=12)
ax_b.set_title('(b) Zero-shot cross-cohort fairness\n(TCGA-SKCM → target)', fontsize=13, fontweight='bold', pad=10)
ax_b.axhline(y=0, color='black', linewidth=0.6)
ax_b.yaxis.grid(True, alpha=0.3)
ax_b.set_axisbelow(True)

ax_b.legend(loc='upper left', framealpha=0.9, fontsize=8.5,
            handlelength=1.5, handletextpad=0.4)

# ── Save ────────────────────────────────────────────────────────────────────
out_dir = '/workspace/src/FOMENT/figures_custom'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'Fig_Leakage_Transfer_Combined.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f"Done! Figure saved to {out_path}")
