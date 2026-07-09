import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 22, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16})
sns.set_context("paper", font_scale=1.8)
COLORS = sns.color_palette("tab20", 20)

MODELS_TO_PLOT = [
    "Vision-Only", 
    "Adv Align (deterministic)", 
    "Adv Align (diagonal)", 
    "Adv Align (low_rank)"
]

MODEL_STYLES = {
    "Vision-Only": {'color': COLORS[0], 'marker': 's'},
    "Adv Align (deterministic)": {'color': COLORS[1], 'marker': '*'},
    "Adv Align (diagonal)": {'color': COLORS[3], 'marker': '*'},
    "Adv Align (low_rank)": {'color': COLORS[5], 'marker': '*'}
}

def load_metrics(dataset):
    path = f"/workspace/src/FOMENT/step2_results_{dataset}/metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def plot_custom_pareto():
    datasets = ['AI4SKIN', 'ASSIST', 'TCGA-SKCM']
    y_col = 'Delta EO (Sex)'
    x_col = 'AUROC'
    
    print("Generating custom Pareto frontier for Delta EO (Sex) vs Macro AUROC...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, ds in zip(axes, datasets):
        df = load_metrics(ds)
        if df is None:
            ax.set_title(f"{ds} (No data)", fontsize=24)
            continue
            
        points = {}
        for i, row in df.iterrows():
            model_raw = row['Model']
            model = model_raw.replace('$^\dagger$', '').strip()
            if model not in MODELS_TO_PLOT:
                continue
                
            x = row[x_col]
            y = row[y_col]
            points[model] = (x, y)
            
            style = MODEL_STYLES.get(model, {'color': 'gray', 'marker': 'o'})
            c = style['color']
            m = style['marker']
                
            ax.scatter(x, y, label=model if ax == axes[0] else "", s=350, c=[c], marker=m, edgecolor='k', linewidth=1.5)
            
        # Draw curved arrow from Vision-Only to Adv Align (low_rank)
        if "Vision-Only" in points and "Adv Align (low_rank)" in points:
            x_start, y_start = points["Vision-Only"]
            x_end, y_end = points["Adv Align (low_rank)"]
            
            # Determine curvature direction based on relative position
            # Usually we want it to arc nicely without intersecting the axis
            rad = 0.3 if y_end < y_start else -0.3
            
            ax.annotate("",
                        xy=(x_end, y_end), xycoords='data',
                        xytext=(x_start, y_start), textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", 
                                        connectionstyle=f"arc3,rad={rad}",
                                        color="black",
                                        lw=2.5,
                                        shrinkA=15, shrinkB=15))
            
        ax.set_title(ds, fontsize=24, fontweight='bold')
        ax.set_xlabel(rf"Performance (Macro {x_col}, $\uparrow$ better)", fontsize=20)
        if ax == axes[0]:
            ax.set_ylabel(rf"Unfairness ({y_col}, $\downarrow$ better)", fontsize=16)
        
    handles, labels = axes[0].get_legend_handles_labels()
    
    if handles:
        order_dict = {m: i for i, m in enumerate(MODELS_TO_PLOT)}
        hl_sorted = sorted(zip(handles, labels), key=lambda x: order_dict.get(x[1], 99))
        handles_sorted, labels_sorted = zip(*hl_sorted)
        fig.legend(handles_sorted, labels_sorted, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, columnspacing=1.0)
        
    plt.tight_layout()
    out_dir = "/workspace/src/FOMENT/figures_custom"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Fig_Pareto_AUROC_Sex_Curved_v2.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Done! Figure saved to {out_path}")

if __name__ == "__main__":
    plot_custom_pareto()
