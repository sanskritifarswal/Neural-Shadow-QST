"""Qiskit -> Cirq conversion for Clifford circuits, replacing the unavailable `qusetta` package,
so that new stabilizer-formalism shadows can be generated for large N (40-qubit demo path).

Conventions (verified by the self-test at N = 6 against dense matrices):
  * Qiskit qubit i  <->  cirq.LineQubit(N-1-i). With this reversal a Qiskit statevector index,
    a Cirq CH-form `initial_state` integer and `inner_product_of_state_and_x(index)` all refer
    to the same computational-basis state, and the NN sample bit order used in Helper_fun*
    (int(''.join(bits), 2)) matches too.
  * Gates emitted by Clifford.to_circuit() in qiskit-terra 0.23: cx, h, s, swap, x, y, z
    (sdg, cz, sx added for safety).

Usage:
    python runs/clifford_cirq.py     # self-test
"""
import numpy as np, cirq

_ONE = {'h': cirq.H, 's': cirq.S, 'sdg': cirq.S**-1, 'x': cirq.X, 'y': cirq.Y, 'z': cirq.Z, 'sx': cirq.X**0.5}
_TWO = {'cx': cirq.CNOT, 'cz': cirq.CZ, 'swap': cirq.SWAP}


def qiskit_to_cirq_ops(qc, qubits):
    """Yield cirq operations for a Qiskit circuit made of Clifford gates (+ barriers)."""
    N = qc.num_qubits
    for inst in qc.data:
        name = inst.operation.name
        idx = [N - 1 - qc.find_bit(q).index for q in inst.qubits]
        if name in ('barrier', 'measure'):
            continue
        if name in _ONE:
            yield _ONE[name](qubits[idx[0]])
        elif name in ('rz', 'rx', 'ry'):           # Clifford only if the angle is a multiple of pi/2
            theta = float(inst.operation.params[0]); e = theta / np.pi
            assert abs(2 * e - round(2 * e)) < 1e-9, f'{name}({theta}) is not Clifford'
            yield {'rz': cirq.Z, 'rx': cirq.X, 'ry': cirq.Y}[name](qubits[idx[0]])**e   # global phase irrelevant
        elif name in _TWO:
            yield _TWO[name](qubits[idx[0]], qubits[idx[1]])
        else:
            raise ValueError(f'unsupported gate {name}')


def ch_form_shadow(prep_circuit, cliff, rng):
    """One classical shadow of the state prepared by `prep_circuit` (Qiskit) measured after
    Clifford `cliff`, simulated in Cirq's CH form. Returns (bitstring in Qiskit convention,
    phi = cirq.StabilizerStateChForm of U^dag |b>)."""
    N = prep_circuit.num_qubits
    qubits = cirq.LineQubit.range(N)
    circ = prep_circuit.compose(cliff.to_circuit())
    args = cirq.StabilizerChFormSimulationState(qubits=qubits, prng=np.random.RandomState(int(rng.integers(2**31))),
                                                initial_state=cirq.StabilizerStateChForm(num_qubits=N))
    for op in qiskit_to_cirq_ops(circ, qubits):
        cirq.act_on(op, args)
    bits = [int(args.state.measure([i], np.random.RandomState(int(rng.integers(2**31))))[0]) for i in range(N)] \
        if hasattr(args.state, 'measure') else None
    if bits is None:  # fallback: measure through act_on
        args.classical_data = cirq.ClassicalDataDictionaryStore()
        cirq.act_on(cirq.measure(*qubits, key='m'), args)
        bits = [int(b) for b in args.classical_data.records[cirq.MeasurementKey('m')][-1]]
    b = ''.join(str(x) for x in bits)               # cirq order = Qiskit MSB-first string
    phi_args = cirq.StabilizerChFormSimulationState(qubits=qubits,
                                                    initial_state=cirq.StabilizerStateChForm(num_qubits=N, initial_state=int(b, 2)))
    for op in qiskit_to_cirq_ops(cliff.adjoint().to_circuit(), qubits):
        cirq.act_on(op, phi_args)
    return b, phi_args.state


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qiskit, Helper_fun as helper
    from qiskit.quantum_info import random_clifford, Statevector
    from nsqst_ext import dense_phi
    N = 6; circ = helper.GHZ_circuit(); rng = np.random.default_rng(0)
    # 1. converter reproduces the dense statevector of prep + Clifford (up to global phase)
    for t in range(3):
        c = random_clifford(N, seed=rng); qc = circ.compose(c.to_circuit())
        sv = np.array(Statevector.from_int(0, 2**N).evolve(qc).data)
        args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(N), initial_state=cirq.StabilizerStateChForm(num_qubits=N))
        for op in qiskit_to_cirq_ops(qc, cirq.LineQubit.range(N)): cirq.act_on(op, args)
        cv = args.state.state_vector()
        assert abs(abs(np.vdot(sv, cv)) - 1) < 1e-8, 'converter mismatch'
    # 2. phi from CH form == U^dag|b> dense, for every basis index, and b is drawn from the right distribution
    worst = 0; counts = {}
    for t in range(200):
        c = random_clifford(N, seed=rng)
        b, phi = ch_form_shadow(circ, c, rng)
        dense = dense_phi(c, b)
        ch = np.array([phi.inner_product_of_state_and_x(i) for i in range(2**N)])
        worst = max(worst, float(abs(abs(np.vdot(dense, ch)) - 1)))
        counts[b] = counts.get(b, 0) + 1
    print(f'phi(CH form) vs dense U^dag|b>: max |1-|overlap|| = {worst:.1e} over 200 shadows')
    # 3. shadow fidelity estimate of the target with itself, from CH-form shadows
    psi = np.array(Statevector.from_int(0, 2**N).evolve(circ).data)
    vals = []
    rng = np.random.default_rng(1)
    for t in range(400):
        c = random_clifford(N, seed=rng); b, phi = ch_form_shadow(circ, c, rng)
        vals.append((2**N + 1) * abs(np.vdot(dense_phi(c, b), psi))**2 - 1)
    print(f'shadow fidelity (CH-form sampled outcomes) = {np.mean(vals):.3f} +/- {np.std(vals)/np.sqrt(len(vals)):.3f} (exact 1)')
    print('OK')
