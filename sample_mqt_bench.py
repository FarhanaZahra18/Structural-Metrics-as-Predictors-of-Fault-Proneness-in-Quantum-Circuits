"""
MQT Bench Sampling Script v2
==============================
Fixed for current MQT Bench API (BenchmarkLevel.ALG).
Samples clean circuits matched by qubit count to Bugs4Q buggy class.
"""

import os, sys, csv
from mqt.bench import get_benchmark, BenchmarkLevel
from qiskit import qasm2

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mqt_clean_circuits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# qubit_count -> needed
QUBIT_DISTRIBUTION = {
    1: 7, 2: 19, 3: 8, 4: 2, 5: 2, 7: 1, 18: 1, 36: 1
}

# ordered by variety — used round-robin per qubit count
ALGORITHMS = [
    "dj","ghz","qaoa","qft","qftentangled","wstate",
    "grover","qpeexact","qpeinexact","graphstate",
    "realamprandom","su2random","twolocalrandom","vqe",
]

def try_get(alg, n):
    try:
        qc = get_benchmark(alg, BenchmarkLevel.ALG, n)
        if qc is None or qc.num_qubits != n:
            return None
        return qc
    except Exception:
        return None

def main():
    print("MQT Bench Sampler v2\n" + "="*40)
    all_rows = []
    total_saved = 0

    for num_q, needed in sorted(QUBIT_DISTRIBUTION.items()):
        print(f"\n{num_q} qubit(s) — need {needed}")
        collected = []
        # cycle through algorithms until we have enough
        for alg in ALGORITHMS * 3:   # *3 allows reuse with random_parameters variation
            if len(collected) >= needed:
                break
            # avoid exact duplicate alg names unless we have no choice
            if alg in [c[0] for c in collected] and len(ALGORITHMS) > needed:
                continue
            qc = try_get(alg, num_q)
            if qc is None:
                continue
            fname = f"clean_{num_q}q_{alg}_{len(collected)+1}.qasm"
            fpath = os.path.join(OUTPUT_DIR, fname)
            try:
                qs = qasm2.dumps(qc)
                with open(fpath, 'w') as f:
                    f.write(qs)
                ops   = qc.count_ops()
                mq    = sum(v for k,v in ops.items() if k in {
                    'cx','cy','cz','ch','cp','crx','cry','crz','cu','cu1','cu3',
                    'ccx','ccz','cswap','swap','ecr','dcx','iswap','rxx','ryy',
                    'rzz','rzx','c3x','c4x','mcx'})
                sz    = qc.size()
                rmqg  = round(mq/sz, 4) if sz > 0 else 0.0
                row   = {
                    'filename':fname,'algorithm':alg,'num_qubits':num_q,
                    'depth':qc.depth(),'size':sz,'multi_qubit_gates':mq,
                    't_gate_count':ops.get('t',0)+ops.get('tdg',0),
                    'measure_count':ops.get('measure',0),
                    'rmqg':rmqg,'label':0
                }
                all_rows.append(row)
                collected.append((alg, fname))
                total_saved += 1
                print(f"  ✓ {fname}  depth={qc.depth()} size={sz}")
            except Exception as e:
                print(f"  ✗ {alg} save error: {e}")

        if len(collected) < needed:
            print(f"  WARNING: only got {len(collected)}/{needed}")

    # save summary CSV
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mqt_metrics.csv")
    fields = ['filename','algorithm','num_qubits','depth','size',
              'multi_qubit_gates','t_gate_count','measure_count','rmqg','label']
    with open(csv_path,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'='*40}")
    print(f"Done. Saved {total_saved}/41 circuits.")
    print(f"QASM files → mqt_clean_circuits/")
    print(f"Metrics    → mqt_metrics.csv")

if __name__ == "__main__":
    main()