"""Gate-level noise models for shadow collection, built with qiskit-aer (pinned qiskit 0.42 API).

The IonQ numbers are rough averages of what IonQ publishes for each generation
(1-qubit / 2-qubit gate fidelity and SPAM). They are not calibration data; treat
them as "the right order of magnitude" and check IonQ's current spec sheet.
IonQ has all-to-all connectivity, so no SWAPs are needed; the IBM-like model uses
a linear chain so the transpiler must insert SWAPs.
"""
import numpy as np
import qiskit
from qiskit import execute, Aer
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, ReadoutError

SPECS = {
    #            1q error, 2q error, readout error, basis gates,           coupling (None = all-to-all)
    'ionq_harmony': dict(p1=5e-3,  p2=3.5e-2, ro=7e-3,  basis=['rx', 'ry', 'rz', 'rxx'], linear=False),
    'ionq_aria':    dict(p1=5e-4,  p2=4e-3,   ro=4e-3,  basis=['rx', 'ry', 'rz', 'rxx'], linear=False),   # Aria-1 spec: 99.95% / 99.6% / SPAM 99.61%
    'ionq_forte':   dict(p1=2e-4,  p2=4e-3,   ro=3e-3,  basis=['rx', 'ry', 'rz', 'rxx'], linear=False),   # Forte spec: 99.98% / 99.6%
    'ionq_tempo':   dict(p1=1e-4,  p2=1e-3,   ro=2e-3,  basis=['rx', 'ry', 'rz', 'rxx'], linear=False),   # Tempo (announced, late 2026): 99.99% / 99.9%
    'ibm_like':     dict(p1=3e-4,  p2=7e-3,   ro=1.5e-2, basis=['rz', 'sx', 'x', 'cx'],  linear=True),
    'ideal':        dict(p1=0.0,   p2=0.0,    ro=0.0,   basis=['rz', 'sx', 'x', 'cx'],  linear=True),
}


def build_noise_model(spec):
    s = SPECS[spec]
    nm = NoiseModel(basis_gates=s['basis'])
    one_q = [g for g in s['basis'] if g != 'rz']         # rz is virtual (error-free) on both platforms
    one_q = [g for g in one_q if g not in ('rxx', 'cx')]
    if s['p1'] > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(s['p1'], 1), one_q)
    if s['p2'] > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(s['p2'], 2), [g for g in s['basis'] if g in ('rxx', 'cx')])
    if s['ro'] > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - s['ro'], s['ro']], [s['ro'], 1 - s['ro']]]))
    return nm


def coupling_map(spec, n):
    return [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)] if SPECS[spec]['linear'] else None


def transpile_for(spec, circuit, seed=None):
    return qiskit.compiler.transpile(circuit, basis_gates=SPECS[spec]['basis'],
                                     coupling_map=coupling_map(spec, circuit.num_qubits),
                                     optimization_level=1, seed_transpiler=seed)


def gate_stats(tcirc):
    ops = tcirc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ('cx', 'rxx', 'cz', 'ecr', 'swap'))
    return {'depth': tcirc.depth(), 'two_qubit_gates': int(two_q), 'total_gates': int(sum(v for k, v in ops.items() if k not in ('measure', 'barrier')))}


def random_Clifford_shadow_aer(circuit, num_shadow, spec, rng=None, stats=None):
    """Drop-in replacement for Helper_fun.random_Clifford_shadow: one shot per random
    Clifford, executed on Aer with the gate-level noise model `spec`. All circuits go in a
    single job. If `stats` (a list) is given, per-circuit gate counts are appended to it."""
    rng = rng if rng is not None else np.random.default_rng()
    n = circuit.num_qubits
    cliffords = [qiskit.quantum_info.random_clifford(n, seed=rng) for _ in range(num_shadow)]
    circs = []
    for cl in cliffords:
        qc = circuit.compose(cl.to_circuit())
        qc.measure_all()   # measure before transpiling so routing SWAPs keep the virtual-qubit bit order
        qc = transpile_for(spec, qc, seed=int(rng.integers(2**31)))
        if stats is not None:
            stats.append(gate_stats(qc))
        circs.append(qc)
    nm = build_noise_model(spec)
    res = execute(circs, Aer.get_backend('qasm_simulator'), noise_model=nm, basis_gates=nm.basis_gates,
                  shots=1, seed_simulator=int(rng.integers(2**31)), optimization_level=0).result()
    results = [res.get_counts(i) for i in range(num_shadow)]
    return cliffords, results
