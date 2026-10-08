"""How much quantum time / money each NSQST protocol needs, as a function of qubit number.

Gate counts come from actually transpiling GHZ + random Clifford circuits (10 random Cliffords
per N) to an IBM-style linear chain (cx/sx/rz/x) and an IonQ-style all-to-all device (rxx/rx/ry/rz).
Timing and price constants are rough public numbers (see PRICES); check them before quoting.
Writes runs/results/quantum_budget.json and runs/figures/fig12_quantum_budget.png.
"""
import sys, os, json, numpy as np
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), os.path.dirname(os.path.abspath(__file__))]
import qiskit, noise_models as nmod
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# per-gate durations (s), readout (s), repetition delay (s), price
PRICES = {
    # IBM: gate/readout times are typical Heron values; rep delay is the default 250 us; price is IBM's pay-as-you-go
    #      list price of $96 per QPU minute = $1.60/s (verify). Open Plan is free: 10 min per 28-day rolling window.
    'IBM (Heron-class)': dict(t1=3.2e-8, t2=6.8e-7, t_ro=1.5e-6, t_rep=2.5e-4, usd_per_s=1.60),
    # IonQ: gate times from Azure's Aria page (1q 135 us, 2q 600 us); prices are Azure Quantum pay-as-you-go
    #      (Aria $0.000220 / 1q gate-shot, $0.000975 / 2q gate-shot; Forte $0.0001645 / $0.001121), minimum per
    #      job $12.42 (Aria) / $25.79 (Forte) with debiasing off, $97.50 / $168.20 with it on. Forte gate times assumed = Aria.
    'IonQ Aria':         dict(t1=1.35e-4, t2=6.0e-4, t_ro=1.0e-4, t_rep=0.0, usd_1q=0.000220,  usd_2q=0.000975, min_job=12.4166),
    'IonQ Forte':        dict(t1=1.35e-4, t2=6.0e-4, t_ro=1.0e-4, t_rep=0.0, usd_1q=0.0001645, usd_2q=0.001121,  min_job=25.7899),
}
PROTOCOLS = {  # (number of shadow circuits, computational-basis shots)
    'NSQST demo, 200 iters (100 fresh shadows/iter)': (100 * 200, 0),
    'NSQST paper, 2000 iters': (100 * 2000, 0),
    'NSQST + pre-training (3000 CB shots + 200 shadows)': (200, 3000),
    'plain classical-shadow fidelity (200 shadows)': (200, 0),
}


def ghz(n):
    c = qiskit.QuantumCircuit(n); c.h(0); c.s(0)
    for i in range(n - 1): c.cx(i, i + 1)
    return c


def stats(n, spec, k=10, rng=np.random.default_rng(0)):
    out = []
    for _ in range(k):
        qc = ghz(n).compose(qiskit.quantum_info.random_clifford(n, seed=rng).to_circuit()); qc.measure_all()
        t = nmod.transpile_for(spec, qc, seed=int(rng.integers(2**31))); ops = t.count_ops()
        two = sum(v for g, v in ops.items() if g in ('cx', 'rxx')); one = sum(v for g, v in ops.items() if g in ('sx', 'x', 'rx', 'ry'))
        out.append((t.depth(), two, one))
    return np.mean(out, axis=0)


def shot_time(p, depth, two, one):
    # IBM: gates run in parallel, so time ~ depth; IonQ: gates are mostly serial on the ion chain
    if p['t_rep'] > 0: return depth * p['t2'] + p['t_ro'] + p['t_rep']
    return two * p['t2'] + one * p['t1'] + p['t_ro']


Ns = [3, 4, 6, 8, 10, 12]
res = {}
for n in Ns:
    d_ibm, two_ibm, one_ibm = stats(n, 'ibm_like'); d_ion, two_ion, one_ion = stats(n, 'ionq_aria')
    d_cb_ibm = ghz(n).depth(); 
    res[n] = dict(ibm=dict(depth=d_ibm, two=two_ibm, one=one_ibm), ionq=dict(depth=d_ion, two=two_ion, one=one_ion))
    for plat, p in PRICES.items():
        g = res[n]['ibm'] if plat.startswith('IBM') else res[n]['ionq']
        t_shadow = shot_time(p, g['depth'], g['two'], g['one']); t_cb = shot_time(p, n, n - 1, 1)
        for name, (n_sh, n_cb) in PROTOCOLS.items():
            T = n_sh * t_shadow + n_cb * t_cb
            if 'usd_per_s' in p: cost = T * p['usd_per_s']
            else: cost = max(p['min_job'], n_sh * (g['two'] * p['usd_2q'] + g['one'] * p['usd_1q']) + n_cb * ((n - 1) * p['usd_2q'] + 1 * p['usd_1q']))
            res[n].setdefault('protocols', {}).setdefault(plat, {})[name] = dict(qpu_seconds=T, usd=cost, shots=n_sh + n_cb)

json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'quantum_budget.json'), 'w'), indent=1)

print(f'{"N":>3} {"IBM 2q":>7} {"IBM depth":>9} {"IonQ 2q":>8} {"IonQ depth":>10}')
for n in Ns: print(f'{n:>3} {res[n]["ibm"]["two"]:7.1f} {res[n]["ibm"]["depth"]:9.1f} {res[n]["ionq"]["two"]:8.1f} {res[n]["ionq"]["depth"]:10.1f}')
print('\nN = 6:')
for plat in PRICES:
    for name, v in res[6]['protocols'][plat].items():
        print(f'  {plat:18s} {name:52s} shots {v["shots"]:7d}  QPU {v["qpu_seconds"]:8.1f} s  ~${v["usd"]:9.2f}')

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(Ns, [res[n]['ibm']['two'] for n in Ns], 'o-', label='IBM-style linear chain (CX, with SWAPs)')
ax[0].plot(Ns, [res[n]['ionq']['two'] for n in Ns], 's-', label='IonQ-style all-to-all (RXX)')
ax[0].set(xlabel='qubits', ylabel='two-qubit gates per shadow circuit', title='Cost of one random-Clifford shadow'); ax[0].legend(fontsize=7)
for i, plat in enumerate(['IBM (Heron-class)', 'IonQ Aria']):
    for name, (n_sh, n_cb) in PROTOCOLS.items():
        ax[1 + i].semilogy(Ns, [res[n]['protocols'][plat][name]['qpu_seconds'] for n in Ns], 'o-', label=name, ms=3)
    ax[1 + i].set(xlabel='qubits', ylabel='estimated QPU seconds', title=f'{plat}: QPU time per protocol')
    ax[1 + i].axhline(600, color='gray', ls='--', lw=.8); ax[1 + i].text(Ns[-1], 450, 'free IBM Open Plan: 10 min per 28 days', fontsize=6, color='gray', ha='right')
ax[1].legend(fontsize=5.5, loc='center right')
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'fig12_quantum_budget.png'), dpi=160)
