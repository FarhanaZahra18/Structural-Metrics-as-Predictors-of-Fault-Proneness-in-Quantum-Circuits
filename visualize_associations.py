"""
visualize_associations.py
==========================
Generates bar charts showing Spearman rank correlation and
Cliff's delta effect sizes between structural metrics and
fault-proneness in quantum circuits.

Output:
    figures/fig_spearman_correlation.png
    figures/fig_effect_sizes.png

Usage:
    python visualize_associations.py
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STATS_CSV = os.path.join(BASE_DIR, "statistical_results.csv")  # FIXED: changed from rq1_rq2_results.csv
FIG_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METRIC_LABELS = {
    'num_qubits'        : 'Number of Qubits',
    'depth'             : 'Circuit Depth',
    'size'              : 'Total Gate Count',
    'multi_qubit_gates' : 'Multi-Qubit Gate Count',
    't_gate_count'      : 'T-Gate Count',
    'measure_count'     : 'Measurement Gate Count',
    'rmqg'              : 'Multi-Qubit Gate Ratio',
}

# Blue shades for significance levels
BLUE_SIG   = '#1a4f72'   # significant
BLUE_NSIG  = '#a8c8e8'   # not significant

BONFERRONI = 0.007143

def load_stats():
    """Load statistical results from CSV"""
    stats = {}
    try:
        with open(STATS_CSV, newline='') as f:
            for row in csv.DictReader(f):
                stats[row['metric']] = row
        return stats
    except FileNotFoundError:
        print(f"ERROR: {STATS_CSV} not found!")
        print("Please run 'python statistical_analysis.py' first.")
        raise

def plot_spearman(stats):
    items = [
        (m, stats[m]) for m in stats
        if m != 't_gate_count' and stats[m]['spearman_rho'] not in ('nan', '')
    ]
    items_sorted = sorted(items,
                          key=lambda x: abs(float(x[1]['spearman_rho'])),
                          reverse=True)

    labels = [METRIC_LABELS[m] for m, _ in items_sorted]
    rhos   = [float(s['spearman_rho']) for _, s in items_sorted]
    pvals  = [float(s['spearman_p'])   for _, s in items_sorted]

    colors = [
        BLUE_SIG if p < BONFERRONI else BLUE_NSIG
        for p in pvals
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(
        labels[::-1], rhos[::-1],
        color=colors[::-1],
        edgecolor='white', height=0.55,
        linewidth=0.8
    )

    ax.axvline(0, color='#333333', linewidth=0.9)
    for x in [-0.3, 0.3]:
        ax.axvline(x, color='#bbbbbb', linewidth=0.7,
                   linestyle='--', alpha=0.7)

    for bar, rho in zip(bars, rhos[::-1]):
        xpos   = bar.get_width()
        ha     = 'left'  if xpos >= 0 else 'right'
        offset = 0.012   if xpos >= 0 else -0.012
        ax.text(xpos + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{rho:.3f}', va='center', ha=ha,
                fontsize=8.5, color='#1a1a1a')

    ax.set_xlabel("Spearman's Rank Correlation Coefficient ($\\rho$)",
                  fontsize=10, color='#2a2a2a')
    ax.set_title(
        "Spearman Correlation between Structural Metrics and Fault-Proneness",
        fontsize=11, fontweight='bold', color='#1a1a1a', pad=10
    )
    ax.set_xlim(-0.68, 0.68)
    ax.tick_params(axis='y', labelsize=9, colors='#2a2a2a')
    ax.tick_params(axis='x', labelsize=8.5, colors='#444444')
    ax.grid(axis='x', linestyle='--', linewidth=0.6,
            alpha=0.45, color='#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_facecolor('#fafafa')

    patch_sig  = mpatches.Patch(facecolor=BLUE_SIG,
                                 label='Significant ($p$ < 0.007, Bonferroni-corrected)')
    patch_nsig = mpatches.Patch(facecolor=BLUE_NSIG,
                                 label='Not significant')
    ax.legend(handles=[patch_sig, patch_nsig],
              fontsize=8.5, loc='lower right',
              framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_spearman_correlation.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


def plot_cliffs_delta(stats):
    items = [
        (m, stats[m]) for m in stats
        if m != 't_gate_count'
    ]
    items_sorted = sorted(items,
                          key=lambda x: abs(float(x[1]['cliffs_delta'])),
                          reverse=True)

    labels     = [METRIC_LABELS[m]          for m, _ in items_sorted]
    deltas     = [float(s['cliffs_delta'])   for _, s in items_sorted]
    magnitudes = [s['effect_magnitude']      for _, s in items_sorted]

    # Blues by magnitude — darker = stronger effect
    mag_colors = {
        'large'     : '#1a4f72',
        'medium'    : '#2e7da6',
        'small'     : '#6aadd5',
        'negligible': '#c5dff0',
    }
    colors = [mag_colors[mag] for mag in magnitudes]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(
        labels[::-1], deltas[::-1],
        color=colors[::-1],
        edgecolor='white', height=0.55,
        linewidth=0.8
    )

    ax.axvline(0, color='#333333', linewidth=0.9)
    for thresh in [0.147, 0.33, 0.474]:
        ax.axvline( thresh, color='#bbbbbb', linewidth=0.7,
                   linestyle=':', alpha=0.7)
        ax.axvline(-thresh, color='#bbbbbb', linewidth=0.7,
                   linestyle=':', alpha=0.7)

    for bar, delta, mag in zip(bars, deltas[::-1], magnitudes[::-1]):
        xpos   = bar.get_width()
        ha     = 'left'  if xpos >= 0 else 'right'
        offset = 0.012   if xpos >= 0 else -0.012
        ax.text(xpos + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{delta:.3f}', va='center', ha=ha,
                fontsize=8.5, color='#1a1a1a')

    ax.set_xlabel("Cliff's Delta ($\\delta$)",
                  fontsize=10, color='#2a2a2a')
    ax.set_title(
        "Effect Size of Structural Metrics between Fault-Prone and Clean Circuits",
        fontsize=11, fontweight='bold', color='#1a1a1a', pad=10
    )
    ax.set_xlim(-0.85, 0.85)
    ax.tick_params(axis='y', labelsize=9, colors='#2a2a2a')
    ax.tick_params(axis='x', labelsize=8.5, colors='#444444')
    ax.grid(axis='x', linestyle='--', linewidth=0.6,
            alpha=0.45, color='#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_facecolor('#fafafa')

    patches = [
        mpatches.Patch(facecolor=c, label=f'{m.capitalize()} effect')
        for m, c in mag_colors.items()
    ]
    ax.legend(handles=patches, fontsize=8.5,
              loc='lower right',
              framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_effect_sizes.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


def main():
    try:
        stats = load_stats()
        plot_spearman(stats)
        plot_cliffs_delta(stats)
        print("\n✅ All visualizations generated successfully!")
        print(f"   - {os.path.join(FIG_DIR, 'fig_spearman_correlation.png')}")
        print(f"   - {os.path.join(FIG_DIR, 'fig_effect_sizes.png')}")
    except FileNotFoundError:
        print("\n❌ Failed to generate visualizations.")
        print("   Please run 'python statistical_analysis.py' first.")

if __name__ == "__main__":
    main()