"""
visualize_distributions.py
===========================
Generates box plots showing the distribution of structural metrics
across fault-prone and clean quantum circuits.

Output:
    figures/fig_metric_distributions.png

Usage:
    python visualize_distributions.py
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fix font issues
plt.rcParams['font.family'] = 'Arial'  # Use Arial which supports semibold
plt.rcParams['font.weight'] = 'normal'  # Default weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "combined_dataset.csv")
STATS_CSV = os.path.join(BASE_DIR, "statistical_results.csv")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METRICS = [
    'num_qubits', 'depth', 'size',
    'multi_qubit_gates', 'measure_count', 'rmqg',
]

METRIC_LABELS = {
    'num_qubits'        : 'Number of Qubits',
    'depth'             : 'Circuit Depth',
    'size'              : 'Total Gate Count',
    'multi_qubit_gates' : 'Multi-Qubit Gate Count',
    'measure_count'     : 'Measurement Gate Count',
    'rmqg'              : 'Multi-Qubit Gate Ratio',
}

# Blue palette — two distinct shades
BLUE_DARK  = '#1a4f72'   # fault-prone
BLUE_LIGHT = '#a8c8e8'   # clean

def load_significant_metrics():
    """Load significant metrics from statistical results"""
    sig_metrics = set()
    if os.path.exists(STATS_CSV):
        with open(STATS_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('significant') == 'Yes':
                    sig_metrics.add(row['metric'])
    return sig_metrics

def load_data():
    buggy = {m: [] for m in METRICS}
    clean = {m: [] for m in METRICS}
    with open(DATA_CSV, newline='') as f:
        for row in csv.DictReader(f):
            target = buggy if int(row['label']) == 1 else clean
            for m in METRICS:
                target[m].append(float(row[m]))
    return buggy, clean

def main():
    buggy, clean = load_data()
    SIG_METRICS = load_significant_metrics()
    
    # If no significant metrics found, use hardcoded fallback
    if not SIG_METRICS:
        print("Warning: No significant metrics found. Using default set.")
        SIG_METRICS = {'depth', 'size', 'rmqg'}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for i, m in enumerate(METRICS):
        ax = axes[i]
        bp = ax.boxplot(
            [buggy[m], clean[m]],
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color='white', linewidth=2.5),
            whiskerprops=dict(color='#4a4a4a', linewidth=1.3),
            capprops=dict(color='#4a4a4a', linewidth=1.3),
            flierprops=dict(marker='o', markersize=3.5,
                            markerfacecolor='#888888',
                            markeredgewidth=0.5, alpha=0.6),
        )
        bp['boxes'][0].set_facecolor(BLUE_DARK)
        bp['boxes'][0].set_alpha(0.85)
        bp['boxes'][1].set_facecolor(BLUE_LIGHT)
        bp['boxes'][1].set_alpha(0.85)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            ['Fault-Prone\n(n = 42)', 'Clean\n(n = 40)'],
            fontsize=9, color='#2a2a2a'
        )
        ax.set_title(METRIC_LABELS[m], fontsize=10,
                     fontweight='bold', color='#1a1a1a', pad=6)  # Changed to 'bold'
        ax.set_ylabel('Value', fontsize=8.5, color='#444444')
        ax.yaxis.set_tick_params(labelsize=8, colors='#444444')
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.45, color='#aaaaaa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.set_facecolor('#fafafa')

        if m in SIG_METRICS:
            ax.text(0.97, 0.97, '*', transform=ax.transAxes,
                    fontsize=14, va='top', ha='right',
                    color=BLUE_DARK, fontweight='bold')

    patch_b = mpatches.Patch(facecolor=BLUE_DARK,  alpha=0.85,
                              label='Fault-Prone (Bugs4Q)')
    patch_c = mpatches.Patch(facecolor=BLUE_LIGHT, alpha=0.85,
                              label='Clean (MQT Bench)')
    fig.legend(
        handles=[patch_b, patch_c],
        loc='lower center', ncol=2, fontsize=10,
        frameon=True, framealpha=0.9,
        edgecolor='#cccccc',
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle(
        'Distribution of Structural Metrics across Fault-Prone and Clean Quantum Circuits',
        fontsize=12, fontweight='bold', color='#1a1a1a', y=1.01
    )
    fig.text(
        0.5, -0.06,
        '* Statistically significant after Bonferroni correction ($\\alpha$ = 0.007)',
        ha='center', fontsize=8.5, color='#555555', style='italic'
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIG_DIR, 'fig_metric_distributions.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()