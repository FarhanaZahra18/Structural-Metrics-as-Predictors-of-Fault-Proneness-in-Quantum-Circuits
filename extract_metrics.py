"""
Quantum Circuit Metric Extraction Script
=========================================
Extracts structural metrics from all 42 cleaned buggy and fixed circuits.
Metrics based on Cruz-Lemus et al. (QUATIC 2021) — ID20 in SMS.

Metrics extracted:
    - num_qubits     : Number of qubits (circuit.num_qubits)
    - depth          : Circuit depth (circuit.depth())
    - size           : Total gate count (circuit.size())
    - multi_qubit_gates : Count of 2+ qubit gates (from circuit.count_ops())
    - t_gate_count   : Count of T gates
    - measure_count  : Count of measurement gates
    - rmqg           : Ratio of multi-qubit gates = multi_qubit_gates / size

Output: metrics.csv in the same directory as this script.

Usage:
    python extract_metrics.py
"""

import os
import csv
import importlib.util
import sys
import traceback

from qiskit import QuantumCircuit

# ── configuration ─────────────────────────────────────────────────────────────
# Set this to wherever you unzipped cleaned_circuits
CLEANED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_circuits")
OUTPUT_CSV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv")

# Gates considered multi-qubit (2+ qubit)
MULTI_QUBIT_GATES = {
    'cx', 'cy', 'cz', 'ch', 'cp', 'crx', 'cry', 'crz',
    'cu', 'cu1', 'cu3', 'ccx', 'ccz', 'cswap', 'cnot',
    'ecr', 'dcx', 'iswap', 'rxx', 'ryy', 'rzz', 'rzx',
    'swap', 'ms', 'c3x', 'c4x', 'mcx', 'mcu1'
}

T_GATES = {'t', 'tdg'}

# ── helpers ───────────────────────────────────────────────────────────────────
def load_circuit(filepath: str):
    """
    Execute a clean circuit file and return the largest QuantumCircuit found.
    Returns None if no circuit is found or file errors.
    """
    spec = importlib.util.spec_from_file_location("circuit_module", filepath)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None

    circuits = [v for v in vars(mod).values() if isinstance(v, QuantumCircuit)]
    if not circuits:
        return None
    # Return the circuit with most gates (most representative)
    return max(circuits, key=lambda c: c.size())


def extract_metrics(qc: QuantumCircuit) -> dict:
    """Extract all structural metrics from a QuantumCircuit object."""
    ops = qc.count_ops()

    num_qubits  = qc.num_qubits
    depth       = qc.depth()
    size        = qc.size()

    mq_count    = sum(count for gate, count in ops.items()
                      if gate.lower() in MULTI_QUBIT_GATES)
    t_count     = sum(count for gate, count in ops.items()
                      if gate.lower() in T_GATES)
    meas_count  = ops.get('measure', 0)
    rmqg        = round(mq_count / size, 4) if size > 0 else 0.0

    return {
        'num_qubits'        : num_qubits,
        'depth'             : depth,
        'size'              : size,
        'multi_qubit_gates' : mq_count,
        't_gate_count'      : t_count,
        'measure_count'     : meas_count,
        'rmqg'              : rmqg,
    }


# ── main extraction loop ──────────────────────────────────────────────────────
def main():
    if not os.path.isdir(CLEANED_DIR):
        print(f"ERROR: cleaned_circuits folder not found at:\n  {CLEANED_DIR}")
        print("Please update CLEANED_DIR in this script to point to your cleaned_circuits folder.")
        sys.exit(1)

    fieldnames = [
        'bug_id', 'version', 'label',
        'num_qubits', 'depth', 'size',
        'multi_qubit_gates', 't_gate_count', 'measure_count', 'rmqg',
        'status'
    ]

    rows = []
    ok_count = 0
    fail_count = 0

    for bug_id in range(1, 43):
        for version in ('buggy', 'fixed'):
            filename = f"{version}_{bug_id}_clean.py"
            filepath = os.path.join(CLEANED_DIR, str(bug_id), filename)

            row = {
                'bug_id'  : bug_id,
                'version' : version,
                'label'   : 1 if version == 'buggy' else 0,  # 1=fault-prone, 0=clean
            }

            if not os.path.exists(filepath):
                row.update({'status': 'FILE_NOT_FOUND',
                            'num_qubits': '', 'depth': '', 'size': '',
                            'multi_qubit_gates': '', 't_gate_count': '',
                            'measure_count': '', 'rmqg': ''})
                fail_count += 1
            else:
                qc = load_circuit(filepath)
                if qc is None:
                    row.update({'status': 'LOAD_FAILED',
                                'num_qubits': '', 'depth': '', 'size': '',
                                'multi_qubit_gates': '', 't_gate_count': '',
                                'measure_count': '', 'rmqg': ''})
                    fail_count += 1
                else:
                    metrics = extract_metrics(qc)
                    row.update(metrics)
                    row['status'] = 'OK'
                    ok_count += 1

            rows.append(row)

    # write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExtraction complete.")
    print(f"  Successful : {ok_count}/84")
    print(f"  Failed     : {fail_count}/84")
    print(f"  Output     : {OUTPUT_CSV}")

    if ok_count > 0:
        print("\nSample output (first 5 rows):")
        print(f"{'bug_id':<8}{'version':<8}{'label':<7}{'qubits':<8}"
              f"{'depth':<8}{'size':<7}{'MQG':<6}{'T':<5}{'meas':<7}{'RMQG':<8}{'status'}")
        print("-" * 80)
        for r in rows[:5]:
            print(f"{r['bug_id']:<8}{r['version']:<8}{r['label']:<7}"
                  f"{r.get('num_qubits',''):<8}{r.get('depth',''):<8}"
                  f"{r.get('size',''):<7}{r.get('multi_qubit_gates',''):<6}"
                  f"{r.get('t_gate_count',''):<5}{r.get('measure_count',''):<7}"
                  f"{r.get('rmqg',''):<8}{r.get('status','')}")


if __name__ == "__main__":
    main()
