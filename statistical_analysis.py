"""
Statistical Analysis Script
"""

import csv
import os
import math
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(BASE_DIR, "combined_dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "statistical_results.csv")
OUTPUT_TXT = os.path.join(BASE_DIR, "statistical_summary.txt")  # ← CHANGED

METRICS = [
    'num_qubits',
    'depth',
    'size',
    'multi_qubit_gates',
    't_gate_count',
    'measure_count',
    'rmqg',
]

METRIC_LABELS = {
    'num_qubits'        : 'Number of Qubits',
    'depth'             : 'Circuit Depth',
    'size'              : 'Total Gate Count',
    'multi_qubit_gates' : 'Multi-Qubit Gate Count',
    't_gate_count'      : 'T-Gate Count',
    'measure_count'     : 'Measurement Gate Count',
    'rmqg'              : 'Ratio of Multi-Qubit Gates (RMQG)',
}

# ── Cliff's delta ─────────────────────────────────────────────────────────────
def cliffs_delta(x, y):
    """
    Computes Cliff's delta effect size between two groups x and y.
    Range: [-1, 1]
    Interpretation: |d| < 0.147 negligible, < 0.33 small,
                    < 0.474 medium, >= 0.474 large
    """
    x, y = list(x), list(y)
    n = len(x) * len(y)
    more = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    return (more - less) / n

def cliffs_magnitude(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.330:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"

# ── load data ─────────────────────────────────────────────────────────────────
def load_data():
    buggy, clean = {m: [] for m in METRICS}, {m: [] for m in METRICS}
    all_vals = {m: [] for m in METRICS}
    labels = []

    with open(INPUT_CSV, newline='') as f:
        for row in csv.DictReader(f):
            label = int(row['label'])
            labels.append(label)
            for m in METRICS:
                val = float(row[m])
                all_vals[m].append(val)
                if label == 1:
                    buggy[m].append(val)
                else:
                    clean[m].append(val)

    return buggy, clean, all_vals, labels

# ── main analysis ─────────────────────────────────────────────────────────────
def main():
    buggy, clean, all_vals, labels = load_data()

    n_buggy = len(buggy['depth'])
    n_clean = len(clean['depth'])
    n_metrics = len(METRICS)
    alpha = 0.05
    alpha_bonferroni = alpha / n_metrics

    results = []

    for m in METRICS:
        b = buggy[m]
        c = clean[m]

        # descriptive stats
        b_median = round(float(np.median(b)), 4)
        c_median = round(float(np.median(c)), 4)
        b_mean   = round(float(np.mean(b)), 4)
        c_mean   = round(float(np.mean(c)), 4)
        b_std    = round(float(np.std(b)), 4)
        c_std    = round(float(np.std(c)), 4)

        # Mann-Whitney U test
        u_stat, p_val = mannwhitneyu(b, c, alternative='two-sided')
        p_val = round(p_val, 6)
        significant = "Yes" if p_val < alpha_bonferroni else "No"

        # Cliff's delta
        d = cliffs_delta(b, c)
        d_rounded = round(d, 4)
        magnitude = cliffs_magnitude(d)

        # Spearman correlation
        rho, sp_pval = spearmanr(all_vals[m], labels)
        rho = round(rho, 4)
        sp_pval = round(sp_pval, 6)

        results.append({
            'metric'              : m,
            'metric_label'        : METRIC_LABELS[m],
            'buggy_median'        : b_median,
            'clean_median'        : c_median,
            'buggy_mean'          : b_mean,
            'clean_mean'          : c_mean,
            'buggy_std'           : b_std,
            'clean_std'           : c_std,
            'mann_whitney_u'      : round(u_stat, 2),
            'p_value'             : p_val,
            'p_bonferroni'        : round(alpha_bonferroni, 6),
            'significant'         : significant,
            'cliffs_delta'        : d_rounded,
            'effect_magnitude'    : magnitude,
            'spearman_rho'        : rho,
            'spearman_p'          : sp_pval,
        })

    # ── write CSV ─────────────────────────────────────────────────────────────
    fields = [
        'metric', 'metric_label',
        'buggy_median', 'clean_median', 'buggy_mean', 'clean_mean',
        'buggy_std', 'clean_std',
        'mann_whitney_u', 'p_value', 'p_bonferroni', 'significant',
        'cliffs_delta', 'effect_magnitude',
        'spearman_rho', 'spearman_p'
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    # ── write summary ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL ANALYSIS RESULTS")
    lines.append(f"Dataset: {n_buggy} buggy circuits + {n_clean} clean circuits")
    lines.append(f"Alpha: {alpha} | Bonferroni-corrected alpha: {round(alpha_bonferroni, 6)}")
    lines.append("=" * 70)

    lines.append("\nGroup Comparison Statistics (Buggy vs. Clean):")
    lines.append("-" * 70)
    lines.append(f"{'Metric':<30} {'Buggy Med':>10} {'Clean Med':>10} "
                 f"{'p-value':>10} {'Sig?':>6} {'Effect':>12}")
    lines.append("-" * 70)

    sig_count = 0
    for r in results:
        sig = r['significant']
        if sig == "Yes":
            sig_count += 1
        lines.append(
            f"{r['metric_label']:<30} "
            f"{r['buggy_median']:>10} "
            f"{r['clean_median']:>10} "
            f"{r['p_value']:>10} "
            f"{sig:>6} "
            f"{r['effect_magnitude']:>12}"
        )

    lines.append("-" * 70)
    lines.append(f"Metrics with significant differences: {sig_count}/{n_metrics}")

    lines.append("\nAssociation Strength Metrics:")
    lines.append("-" * 70)
    lines.append(f"{'Metric':<30} {'Spearman rho':>14} {'p-value':>10} {'Cliff delta':>12} {'Magnitude':>12}")
    lines.append("-" * 70)

    sorted_by_rho = sorted(results, key=lambda x: abs(x['spearman_rho']), reverse=True)
    for r in sorted_by_rho:
        lines.append(
            f"{r['metric_label']:<30} "
            f"{r['spearman_rho']:>14} "
            f"{r['spearman_p']:>10} "
            f"{r['cliffs_delta']:>12} "
            f"{r['effect_magnitude']:>12}"
        )

    lines.append("=" * 70)
    lines.append(f"\nFull results saved to: {OUTPUT_CSV}")

    summary = "\n".join(lines)
    print(summary)

    with open(OUTPUT_TXT, 'w') as f:
        f.write(summary)

    print(f"\nSummary saved to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()