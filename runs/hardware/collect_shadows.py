"""Collect random-Clifford classical shadows of the phase-shifted GHZ state on a backend.

Runs in the modern-Qiskit env (conda env `qhw`), NOT in the pinned `nsqst` env.
Output is a JSON file with one entry per shadow: the Clifford (as stabilizer/destabilizer
labels, so the old env can rebuild it with Clifford.from_dict) and the measured bitstring,
plus job metadata and QPU usage when the backend reports it.

Backends
  aer                          ideal Aer simulator
  fake:<name>                  Aer with the noise model of an IBM fake backend, e.g. fake:FakeTorino
  ibm:<name> | ibm:least_busy  real IBM QPU through qiskit-ibm-runtime (needs a saved account)
  ionq:<name>                  IonQ through qiskit-ionq, e.g. ionq:simulator (needs IONQ_API_KEY)

Example
  python collect_shadows.py --backend fake:FakeTorino --n 6 --num 200 --out shadows_fake_torino_n6.json
"""
import argparse, json, time, os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford


def ghz(n):
    c = QuantumCircuit(n); c.h(0); c.s(0)
    for i in range(n - 1): c.cx(i, i + 1)
    return c


def get_backend(name):
    if name == 'aer':
        from qiskit_aer import AerSimulator; return AerSimulator(), 'aer'
    kind, _, rest = name.partition(':')
    if kind == 'fake':
        import qiskit_ibm_runtime.fake_provider as fp
        from qiskit_aer import AerSimulator
        real = getattr(fp, rest)(); return AerSimulator.from_backend(real), 'aer'
    if kind == 'ibm':
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService()
        return (svc.least_busy(operational=True, simulator=False) if rest == 'least_busy' else svc.backend(rest)), 'ibm'
    if kind == 'ionq':
        from qiskit_ionq import IonQProvider
        return IonQProvider().get_backend(f'ionq_{rest}'), 'ionq'
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='aer'); ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--num', type=int, default=200, help='number of shadows (one shot each)')
    ap.add_argument('--cb_shots', type=int, default=0, help='also take this many computational-basis shots (for pre-training)')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    backend, kind = get_backend(a.backend)
    print('backend:', backend.name if hasattr(backend, 'name') else backend)

    cliffords = [random_clifford(a.n, seed=rng) for _ in range(a.num)]
    circs = []
    for cl in cliffords:
        qc = ghz(a.n).compose(cl.to_circuit()); qc.measure_all(); circs.append(qc)
    cb = ghz(a.n); cb.measure_all()
    tcircs = transpile(circs + ([cb] if a.cb_shots else []), backend=backend, optimization_level=1, seed_transpiler=a.seed)
    two_q = [sum(v for g, v in t.count_ops().items() if g in ('cx', 'ecr', 'cz', 'rxx', 'ms')) for t in tcircs]
    print(f'transpiled: mean 2q gates {np.mean(two_q[:a.num]):.1f}, mean depth {np.mean([t.depth() for t in tcircs[:a.num]]):.1f}')

    t0 = time.time(); usage = None; job_id = None
    if kind == 'ibm':
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        pubs = [(t, None, 1) for t in tcircs[:a.num]] + ([(tcircs[-1], None, a.cb_shots)] if a.cb_shots else [])
        job = sampler.run(pubs); job_id = job.job_id(); print('job', job_id, 'submitted; waiting...')
        result = job.result()
        bits = [next(iter(r.data.meas.get_counts())) for r in result[:a.num]]
        cb_counts = result[a.num].data.meas.get_counts() if a.cb_shots else {}
        try: usage = job.usage()          # QPU seconds billed to the account
        except Exception: usage = job.metrics().get('usage', None)
    else:
        jobs = [backend.run(t, shots=1, seed_simulator=int(rng.integers(2**31))) if kind == 'aer' else backend.run(t, shots=1) for t in tcircs[:a.num]]
        bits = [next(iter(j.result().get_counts())) for j in jobs]
        cb_counts = backend.run(tcircs[-1], shots=a.cb_shots).result().get_counts() if a.cb_shots else {}
    wall = time.time() - t0

    out = dict(n=a.n, backend=a.backend, seed=a.seed, job_id=job_id, wall_seconds=wall, qpu_usage=usage,
               mean_two_qubit_gates=float(np.mean(two_q[:a.num])), cb_counts=cb_counts,
               shadows=[dict(clifford=cl.to_dict(), bit=b) for cl, b in zip(cliffords, bits)])
    json.dump(out, open(a.out, 'w'))
    print(f'saved {a.num} shadows to {a.out}; wall {wall:.1f}s; qpu usage {usage}')


if __name__ == '__main__':
    main()
