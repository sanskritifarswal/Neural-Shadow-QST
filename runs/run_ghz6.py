"""Script version of NSQST_GHZ_demo.ipynb (6-qubit phase-shifted GHZ, noiseless).
Same hyperparameters as the notebook; saves curves + final state to runs/results/.
Usage: python runs/run_ghz6.py [seed] [max_iters] [tag] [lamb]
lamb > 0: shadows are sampled from the state after an n-qubit depolarizing channel of
strength lamb (uses the repo's random_Clifford_shadow_dep_n), mimicking a noisy device.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from qiskit.quantum_info import Statevector
import nqs_models, utils
from exact_solvers import GenericExactState
import Helper_fun as helper
import NSQST_Trainer as nsqst_trainer

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
max_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
tag = sys.argv[3] if len(sys.argv) > 3 else f'ghz6_seed{seed}'
lamb = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
if lamb > 0:
    helper.random_Clifford_shadow = lambda c, num_shadow: helper.random_Clifford_shadow_dep_n(c, num_shadow, lamb)
out = os.path.join(os.path.dirname(__file__), 'results')

torch.manual_seed(seed); np.random.seed(seed)

N = 6
circ = helper.GHZ_circuit()
init = np.zeros(2**N, 'complex128'); init[0] = 1
target_dense = np.array(Statevector(init).evolve(circ).data)

nqs = nqs_models.TransformerWF(num_sites=N, num_layers=2, internal_dimension=8,
                               num_heads=4, dropout=0.0)
opt = torch.optim.Adam(nqs.parameters(), lr=1e-2)
trainer = nsqst_trainer.NSQST_Trainer(
    nqs_model=nqs, circuit=circ, optimizer=opt, batch_size=100,
    target_state=GenericExactState(utils.Complex(target_dense)),
    max_iters=max_iters, num_samples=5000, K=1, state_file_name=None)

t0 = time.time()
exact, loss = trainer.train()
dt = time.time() - t0

exact = np.array(exact, dtype=float); loss = np.array(loss, dtype=float)
np.savetxt(f'{out}/{tag}_exact.txt', exact); np.savetxt(f'{out}/{tag}_loss.txt', loss)
with torch.no_grad():
    psi = nqs.full_state().state  # GenericExactState -> utils.Complex
    psi = psi.real.numpy().astype(complex) + 1j * psi.imag.numpy()
np.save(f'{out}/{tag}_final_state.npy', psi)
np.save(f'{out}/ghz6_target_state.npy', target_dense)
json.dump({'seed': seed, 'lamb': lamb, 'iters': max_iters, 'runtime_s': dt,
           'final_infid': float(exact[-1]), 'min_infid': float(exact.min()),
           'mean_last20': float(exact[-20:].mean())},
          open(f'{out}/{tag}_summary.json', 'w'), indent=1)
print('DONE', tag, 'runtime', dt, 'final', exact[-1], 'min', exact.min())
