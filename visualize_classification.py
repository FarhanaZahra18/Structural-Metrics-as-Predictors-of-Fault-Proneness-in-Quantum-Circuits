"""
visualize_classification.py
============================
Generates publication-quality figures for classification results.

Figures produced:
    figures/fig_roc_curves.png       — ROC curves for all three classifiers
    figures/fig_feature_importance.png — Random Forest feature importance

Requires roc_data.pkl and imp_data.pkl produced by classification.py.

Usage:
    python visualize_classification.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = [
    'num_qubits', 'depth', 'size',
    'multi_qubit_gates', 'measure_count', 'rmqg',
]
FEATURE_LABELS = {
    'num_qubits'        : 'Number of Qubits',
    'depth'             : 'Circuit Depth',
    'size'              : 'Total Gate Count',
    'multi_qubit_gates' : 'Multi-Qubit Gate Count',
    'measure_count'     : 'Measurement Gate Count',
    'rmqg'              : 'Multi-Qubit Gate Ratio',
}

# Blue shades per model — light to dark
MODEL_COLORS = {
    'Logistic Regression': '#6aadd5',
    'SVM (RBF kernel)'   : '#2e7da6',
    'Random Forest'      : '#1a4f72',
}


def load_pkl(name):
    path = os.path.join(BASE_DIR, name)
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_roc(roc_store):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (model_name, roc_data) in zip(axes, roc_store.items()):
        color = MODEL_COLORS[model_name]

        for fpr, tpr, auc in roc_data:
            ax.plot(fpr, tpr, color=color, alpha=0.25, linewidth=1)

        # mean ROC
        mean_fpr = np.linspace(0, 1, 300)
        interp_tprs = [
            np.interp(mean_fpr, fpr, tpr)
            for fpr, tpr, _ in roc_data
        ]
        mean_tpr = np.mean(interp_tprs, axis=0)
        std_tpr  = np.std(interp_tprs,  axis=0)
        mean_auc = np.mean([auc for _, _, auc in roc_data])
        std_auc  = np.std( [auc for _, _, auc in roc_data])

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2.2,
                label=f'Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}')
        ax.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color=color, alpha=0.12
        )
        ax.plot([0, 1], [0, 1], color='#aaaaaa', linewidth=1,
                linestyle='--', label='Random classifier')

        ax.set_title(model_name, fontsize=10,
                     fontweight='bold', color='#1a1a1a', pad=8)
        ax.set_xlabel('False Positive Rate', fontsize=9, color='#2a2a2a')
        ax.legend(fontsize=8, loc='lower right',
                  framealpha=0.9, edgecolor='#cccccc')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.4, color='#aaaaaa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.set_facecolor('#fafafa')
        ax.tick_params(labelsize=8, colors='#444444')

    axes[0].set_ylabel('True Positive Rate', fontsize=9, color='#2a2a2a')

    fig.suptitle(
        'Receiver Operating Characteristic Curves for Fault-Proneness Prediction\n'
        '(Stratified 5-Fold Cross-Validation, n = 82)',
        fontsize=11, fontweight='bold', color='#1a1a1a', y=1.02
    )

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_roc_curves.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


def plot_importance(imp_store):
    if 'Random Forest' not in imp_store:
        print("No Random Forest importance data found.")
        return

    importances = imp_store['Random Forest']
    labels      = [FEATURE_LABELS[f] for f in FEATURES]
    idx         = np.argsort(importances)

    sorted_labels = [labels[i] for i in idx]
    sorted_vals   = importances[idx]

    # shade bars by importance — darker = more important
    max_val = sorted_vals.max()
    colors  = [
        plt.cm.Blues(0.35 + 0.6 * (v / max_val))
        for v in sorted_vals
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(
        sorted_labels, sorted_vals,
        color=colors, edgecolor='white',
        height=0.55, linewidth=0.8
    )

    for bar, val in zip(bars, sorted_vals):
        ax.text(
            bar.get_width() + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left',
            fontsize=8.5, color='#1a1a1a'
        )

    ax.set_xlabel('Mean Decrease in Impurity',
                  fontsize=10, color='#2a2a2a')
    ax.set_title(
        'Random Forest Feature Importance for Quantum Circuit Fault-Proneness Prediction',
        fontsize=11, fontweight='bold', color='#1a1a1a', pad=10
    )
    ax.set_xlim(0, sorted_vals.max() + 0.07)
    ax.tick_params(axis='y', labelsize=9, colors='#2a2a2a')
    ax.tick_params(axis='x', labelsize=8.5, colors='#444444')
    ax.grid(axis='x', linestyle='--', linewidth=0.5,
            alpha=0.45, color='#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_facecolor('#fafafa')

    # simple colorbar-style legend
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap='Blues',
                        norm=Normalize(vmin=0, vmax=max_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Relative importance', fontsize=8.5,
                   color='#444444')
    cbar.ax.tick_params(labelsize=8, colors='#444444')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_feature_importance.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Generating classification figures...")
    try:
        roc_store = load_pkl('roc_data.pkl')
        imp_store = load_pkl('imp_data.pkl')
        plot_roc(roc_store)
        plot_importance(imp_store)
        print("Done.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python classification.py' first to generate the required .pkl files.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()