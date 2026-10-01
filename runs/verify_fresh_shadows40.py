"""Sanity check of 40-qubit shadows: shadow estimate of the fidelity of the EXACT target
(|0..0> + i|1..1>)/sqrt(2) with itself. Must be 1 within error bars if the shadows are valid.
Compares the author's 200 stored shadows with fresh ones from runs/clifford_cirq.py
(the same 200 used as held-out set by run_ghz40_holdout.py --ntrain 200 --seed 0, plus 200 new).
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np, cirq
import Helper_fun_new as helper
from qiskit.quantum_info import random_clifford
from clifford_cirq import ch_form_shadow

N = 40; ALL1 = 2**N - 1

def val(phi):
    ov = (np.conj(phi.inner_product_of_state_and_x(0)) + 1j * np.conj(phi.inner_product_of_state_and_x(ALL1))) / np.sqrt(2)
    return (2**N + 1) * abs(ov)**2 - 1

def report(name, v):
    v = np.array(v); print(f'{name}: n={len(v)}  mean {v.mean():.3f} +/- {v.std(ddof=1)/np.sqrt(len(v)):.3f}  (exact 1)  '
                           f'fraction nonzero overlap {np.mean(v > -0.5):.3f}', flush=True)
    return v

out = {}
phi_all = cirq.read_json(json_text=json.load(open(f'{ROOT}/data/phi_list_40.json')))
out['author'] = report("author's stored shadows", [val(p) for p in phi_all]).tolist()

circ = helper.GHZ_circuit(N)
rng = np.random.default_rng(4000); rng.permutation(200)          # same stream as the held-out run
t0 = time.time(); v = []
for k in range(200):
    b, phi = ch_form_shadow(circ, random_clifford(N, seed=rng), rng); v.append(val(phi))
out['fresh_same'] = report(f'fresh shadows, same 200 as the held-out run ({time.time()-t0:.0f}s)', v).tolist()

rng = np.random.default_rng(777); v2 = []
for k in range(200):
    b, phi = ch_form_shadow(circ, random_clifford(N, seed=rng), rng); v2.append(val(phi))
out['fresh_new'] = report('fresh shadows, 200 new', v2).tolist()
report('fresh, all 400', v + v2)
json.dump(out, open(f'{HERE}/results/verify_fresh_shadows40.json', 'w'))
