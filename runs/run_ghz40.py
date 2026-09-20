"""Script version of NSQST_with_pre_training_demo.ipynb (40-qubit phase-shifted GHZ).
Loads the pre-trained amplitude network + 200 stored Clifford shadows, trains the phase
network. Does NOT overwrite the author's files in data/.
Usage: python runs/run_ghz40.py [seed] [max_iters]
"""
import sys, os, time, json
ROOT = os.path.join(os.path.dirname(__file__), '..'); sys.path.insert(0, ROOT)
import numpy as np, torch, cirq
import nqs_models
import Helper_fun_new as helper
import NSQST_Pre_Trainer_GHZ as shadow_trainer

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
max_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
tag = f'ghz40_seed{seed}'; out = os.path.join(os.path.dirname(__file__), 'results')
torch.manual_seed(seed); np.random.seed(seed)

N = 40
amp = nqs_models.TransformerWF(num_sites=N, num_layers=3, internal_dimension=8,
                               num_heads=4, dropout=0.0, phase_mode=0)
amp.load_state_dict(torch.load(f'{ROOT}/data/param_GHZ_pre_amp_N40'))
phase = nqs_models.TransformerWF_phase(num_sites=N, num_layers=3, internal_dimension=8,
                                       num_heads=4, dropout=0.0, phase_mode=2)
opt = torch.optim.Adam(phase.parameters(), lr=1e-2)

t0 = time.time()
phi_list = cirq.read_json(json_text=json.load(open(f'{ROOT}/data/phi_list_40.json')))
print('loaded', len(phi_list), 'shadows in', time.time() - t0, 's', flush=True)

trainer = shadow_trainer.ShadowTomographyTrainer(
    N=N, nqs_model_amp=amp, nqs_model_phase=phase, phi_list=phi_list, optimizer=opt,
    max_iters=max_iters, num_samples=5000, batch_size=200, K=1,
    state_file_name_amp=None, state_file_name_phase=None)
t0 = time.time()
exact, loss = trainer.train()
dt = time.time() - t0
exact = np.array(exact, dtype=float).ravel(); loss = np.array(loss, dtype=float).ravel()
np.savetxt(f'{out}/{tag}_exact.txt', exact); np.savetxt(f'{out}/{tag}_loss.txt', loss)
with torch.no_grad():
    s = amp.sample(5000); u, c = torch.unique(s, dim=0, return_counts=True)
    json.dump({'seed': seed, 'iters': max_iters, 'runtime_s': dt,
               'final_infid': float(exact[-1]), 'min_infid': float(exact.min()),
               'mean_last20': float(exact[-20:].mean()),
               'amp_net_unique_samples': [(''.join(map(str, r.tolist())), int(k)) for r, k in zip(u, c)]},
              open(f'{out}/{tag}_summary.json', 'w'), indent=1)
print('DONE', tag, 'runtime', dt, 'final', exact[-1], 'min', exact.min())
