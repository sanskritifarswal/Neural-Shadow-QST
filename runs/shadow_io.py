"""Shadow data file format: the only thing NSQST needs from a quantum computer.

Each snapshot is (random Clifford U_i, measured bitstring b_i). We store the Clifford as its
stabilizer/destabilizer Pauli strings (Clifford.to_dict()), which exist in every Qiskit
version from 0.2x to 2.x, so a data-collection script running on modern qiskit-ibm-runtime /
qiskit-ionq can hand shadows to this (qiskit 0.42) training environment through a JSON file.

    {"num_qubits": 6, "backend": "...", "meta": {...},
     "shadows": [{"stabilizer": [...], "destabilizer": [...], "bitstring": "010110"}, ...]}

Bitstring convention: Qiskit's counts key (qubit 0 is the rightmost character), identical to
what Helper_fun.random_Clifford_shadow / Statevector.sample_counts produce.

Usage:
    python runs/shadow_io.py            # self-test: round-trip + amplitude check at N=6
"""
import json, numpy as np
from qiskit.quantum_info import Clifford


def save_shadows(path, shadows, num_qubits, backend='statevector', meta=None):
    out = {'num_qubits': num_qubits, 'backend': backend, 'meta': meta or {},
           'shadows': [{**c.to_dict(), 'bitstring': b} for c, b in shadows]}
    with open(path, 'w') as f:
        json.dump(out, f)


def load_shadows(path):
    d = json.load(open(path))
    shadows = [(Clifford.from_dict({'stabilizer': s['stabilizer'], 'destabilizer': s['destabilizer']}),
                s['bitstring']) for s in d['shadows']]
    return shadows, d


if __name__ == '__main__':
    import os, sys, tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nsqst_ext import seeded_shadow_fn, flatten_shadows, dense_phi, shadow_fidelity_estimate
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import Helper_fun as helper
    from qiskit.quantum_info import Statevector
    N = 6; circ = helper.GHZ_circuit()
    psi = np.array(Statevector.from_int(0, 2**N).evolve(circ).data)
    rng = np.random.default_rng(123)
    sh = flatten_shadows(*seeded_shadow_fn(N, rng)(circ, 400))
    p = os.path.join(tempfile.gettempdir(), 'shadows_test.json')
    save_shadows(p, sh, N, meta={'seed': 123})
    sh2, d = load_shadows(p)
    assert all(c1 == c2 and b1 == b2 for (c1, b1), (c2, b2) in zip(sh, sh2)), 'round-trip failed'
    # phi amplitudes identical after round trip
    assert np.allclose(dense_phi(*sh[0]), dense_phi(*sh2[0]))
    f, se = shadow_fidelity_estimate(psi, sh2)
    print(f'round-trip OK: {len(sh2)} shadows, {os.path.getsize(p)/1e3:.1f} kB; '
          f'shadow fidelity of target with itself = {f:.3f} +/- {se:.3f} (exact 1)')
