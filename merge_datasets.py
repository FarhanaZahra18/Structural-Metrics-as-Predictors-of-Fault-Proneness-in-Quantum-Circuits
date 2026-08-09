"""
Dataset Merging Script
=======================
Merges Bugs4Q buggy circuit metrics and MQT Bench clean circuit metrics
into a single combined dataset for analysis.

Input:
    - metrics.csv      : Bugs4Q buggy + fixed circuits (label 1 = buggy, 0 = fixed)
    - mqt_metrics.csv  : MQT Bench clean circuits (label 0)

Output:
    - combined_dataset.csv : 42 buggy (label=1) + 40 clean (label=0) = 82 circuits

Usage:
    python merge_datasets.py
"""

import csv
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BUGS4Q_CSV = os.path.join(BASE_DIR, "metrics.csv")
MQT_CSV    = os.path.join(BASE_DIR, "mqt_metrics.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "combined_dataset.csv")

FIELDS = [
    'source', 'id', 'num_qubits', 'depth', 'size',
    'multi_qubit_gates', 't_gate_count', 'measure_count', 'rmqg', 'label'
]

def main():
    rows = []

    # ── Bugs4Q: keep only buggy versions (label=1) ───────────────────────────
    with open(BUGS4Q_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r['version'] == 'buggy':
                rows.append({
                    'source'            : 'Bugs4Q',
                    'id'                : f"bug_{r['bug_id']}",
                    'num_qubits'        : r['num_qubits'],
                    'depth'             : r['depth'],
                    'size'              : r['size'],
                    'multi_qubit_gates' : r['multi_qubit_gates'],
                    't_gate_count'      : r['t_gate_count'],
                    'measure_count'     : r['measure_count'],
                    'rmqg'              : r['rmqg'],
                    'label'             : 1,
                })

    buggy_count = len(rows)

    # ── MQT Bench: all circuits are clean (label=0) ───────────────────────────
    with open(MQT_CSV, newline='') as f:
        for r in csv.DictReader(f):
            rows.append({
                'source'            : 'MQTBench',
                'id'                : r['filename'].replace('.qasm', ''),
                'num_qubits'        : r['num_qubits'],
                'depth'             : r['depth'],
                'size'              : r['size'],
                'multi_qubit_gates' : r['multi_qubit_gates'],
                't_gate_count'      : r['t_gate_count'],
                'measure_count'     : r['measure_count'],
                'rmqg'              : r['rmqg'],
                'label'             : 0,
            })

    clean_count = len(rows) - buggy_count

    # ── write combined CSV ────────────────────────────────────────────────────
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Dataset merge complete.")
    print(f"  Buggy circuits  (label=1) : {buggy_count}")
    print(f"  Clean circuits  (label=0) : {clean_count}")
    print(f"  Total                     : {len(rows)}")
    print(f"  Output                    : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
