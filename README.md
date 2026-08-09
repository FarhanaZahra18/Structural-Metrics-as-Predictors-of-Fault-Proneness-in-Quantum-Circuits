# Structural-Metrics-as-Predictors-of-Fault-Proneness-in-Quantum-Circuits
## Overview

This repository contains the replication package for the empirical study investigating whether structural properties of quantum circuits can predict fault-proneness without simulation or hardware execution.

The study extracts seven structural metrics from 82 labeled quantum circuits drawn from two publicly available sources: Bugs4Q (fault-prone) and MQT Bench (clean) and evaluates their predictive utility through statistical analysis and machine learning classification.

---

## Repository Structure

```
.
├── cleaned_circuits/          # Reconstructed Bugs4Q circuits compatible with Qiskit 2.4.2
│   ├── 1/
│   │   ├── buggy_1_clean.py
│   │   └── fixed_1_clean.py
│   └── ...                    # Folders 1–42
├── mqt_clean_circuits/        # Sampled MQT Bench circuits in QASM format
├── figures/                   # All publication figures (300 DPI PNG)
│   ├── fig_metric_distributions.png
│   ├── fig_spearman_correlation.png
│   ├── fig_effect_sizes.png
│   ├── fig_roc_curves.png
│   └── fig_feature_importance.png
├── extract_metrics.py         # Metric extraction from Bugs4Q circuits
├── sample_mqt_bench.py        # Clean circuit sampling from MQT Bench
├── merge_datasets.py          # Combines both datasets into combined_dataset.csv
├── statistical_analysis.py    # Mann-Whitney U, Bonferroni, Cliff's delta, Spearman
├── classification.py          # LR, SVM, Random Forest with 5-fold CV
├── visualize_distributions.py # Box plot figures for metric distributions
├── visualize_associations.py  # Spearman and effect size bar charts
├── visualize_classification.py# ROC curves and feature importance figures
├── metrics.csv                # Extracted metrics from Bugs4Q (buggy + fixed)
├── mqt_metrics.csv            # Extracted metrics from MQT Bench circuits
├── combined_dataset.csv       # Final labeled dataset (82 circuits)
├── statistical_results.csv    # Full statistical analysis results
├── classification_results.csv # Per-fold and mean classification results
└── requirements.txt           # Python dependencies
```

---

## Datasets

**Bugs4Q** — Zhao et al. (2023), Journal of Systems and Software  
Source: https://github.com/Z-928/Bugs4Q-Framework  
42 real, manually validated Qiskit bugs collected from GitHub, StackOverflow, and Stack Exchange.

**MQT Bench** — Quetschlich et al. (2023), Quantum  
Source: https://github.com/cda-tum/mqt-bench  
Benchmarking suite providing quantum circuits across multiple abstraction levels. Circuits sampled at the algorithmic abstraction level, matched by qubit count to the Bugs4Q distribution.

---

## Metrics Extracted

| Notation | Metric | Qiskit API |
|----------|--------|------------|
| NQ | Number of Qubits | `circuit.num_qubits` |
| CD | Circuit Depth | `circuit.depth()` |
| TG | Total Gate Count | `circuit.size()` |
| MQG | Multi-Qubit Gate Count | `circuit.count_ops()` |
| TGC | T-Gate Count | `circuit.count_ops()` |
| MNG | Measurement Gate Count | `circuit.count_ops()` |
| RMQG | Ratio of Multi-Qubit Gates | MQG / TG |

Note: TGC is constant at zero across this dataset and is excluded from classification.

---

## Requirements

Python 3.10 or higher is recommended.

```
qiskit>=2.4.0
mqt.bench
scikit-learn
scipy
numpy
matplotlib
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing the Study

Run the scripts in the following order:

```bash
# Step 1: Extract metrics from Bugs4Q circuits
python extract_metrics.py

# Step 2: Sample clean circuits from MQT Bench
python sample_mqt_bench.py

# Step 3: Merge into combined dataset
python merge_datasets.py

# Step 4: Run statistical analysis
python statistical_analysis.py

# Step 5: Run classification
python classification.py

# Step 6: Generate figures
python visualize_distributions.py
python visualize_associations.py
python visualize_classification.py
```

All scripts are self-contained and require no quantum hardware access. Everything runs on a standard laptop CPU.

---

## Results Summary

**Statistical analysis (RQ1 and RQ2)**

Three metrics differ significantly between fault-prone and clean circuits after Bonferroni correction (α = 0.00714):

| Metric | p-value | Cliff's δ | Magnitude |
|--------|---------|-----------|-----------|
| Circuit Depth | < 0.001 | −0.536 | Large |
| Total Gate Count | < 0.001 | −0.524 | Large |
| RMQG | 0.007 | +0.332 | Medium |

Circuit Depth (ρ = −0.474) and Total Gate Count (ρ = −0.457) show the strongest Spearman associations with fault-proneness.

**Classification (RQ3)**

| Model | AUC | F1 |
|-------|-----|----|
| Logistic Regression | 0.848 ± 0.039 | 0.825 ± 0.083 |
| SVM (RBF kernel) | 0.847 ± 0.061 | 0.715 ± 0.068 |
| Random Forest | 0.846 ± 0.097 | 0.804 ± 0.068 |

All three classifiers converge at AUC ≈ 0.847 under stratified 5-fold cross-validation, indicating the finding is not model-dependent.

---

## License

This replication package is released for academic use. The Bugs4Q and MQT Bench datasets are subject to their respective original licenses.
