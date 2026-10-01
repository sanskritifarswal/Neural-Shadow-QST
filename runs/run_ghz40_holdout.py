"""40-qubit NSQST with pre-training: held-out shadow evaluation and shadow-count sweep.

Last week's finding: with the same 200 shadows reused every iteration, the reported loss is
biased ~ -0.025 below the exact infidelity (overfitting to the shadow set). This script
 (a) trains the phase net on n_train shadows and evaluates the shadow estimate on a
     DISJOINT held-out set, alongside the exact infidelity;
 (b) sweeps n_train in {25, 50, 100, 200}.
Held-out shadows: the 200 - n_train unused author shadows when n_train < 200; for n_train = 200
a fresh set of 200 is generated with runs/clifford_cirq.py (replaces qusetta).

Usage: python runs/run_ghz40_holdout.py --ntrain 100 [--seed 0] [--iters 200]
"""
import argparse, os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument('--ntrain', type=int, default=100)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--nhold', type=int, default=200, help='fresh held-out shadows when ntrain == 200')
ap.add_argument('--threads', type=int, default=2)
args = ap.parse_args()
import numpy as np, torch, cirq
torch.set_num_threads(args.threads)
import nqs_models, utils
import Helper_fun_new as helper
import NSQST_Pre_Trainer_GHZ as shadow_trainer
from qiskit.quantum_info import random_clifford
from clifford_cirq import ch_form_shadow

tag = f'ghz40_hold_n{args.ntrain}_seed{args.seed}'; out = os.path.join(HERE, 'results')
class _Tee:
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()
sys.stdout = sys.stderr = _Tee(f'{out}/{tag}.log')
torch.manual_seed(args.seed); np.random.seed(args.seed); rng = np.random.default_rng(4000 + args.seed)

N = 40
amp = nqs_models.TransformerWF(num_sites=N, num_layers=3, internal_dimension=8, num_heads=4, dropout=0.0, phase_mode=0)
amp.load_state_dict(torch.load(f'{ROOT}/data/param_GHZ_pre_amp_N40'))
phase = nqs_models.TransformerWF_phase(num_sites=N, num_layers=3, internal_dimension=8, num_heads=4, dropout=0.0, phase_mode=2)
opt = torch.optim.Adam(phase.parameters(), lr=1e-2)

t0 = time.time()
phi_all = cirq.read_json(json_text=json.load(open(f'{ROOT}/data/phi_list_40.json')))
perm = rng.permutation(len(phi_all))
train = [phi_all[i] for i in perm[:args.ntrain]]
if args.ntrain < len(phi_all):
    hold = [phi_all[i] for i in perm[args.ntrain:]]; hold_src = f'{len(hold)} unused author shadows'
else:
    circ = helper.GHZ_circuit(N); hold = []
    for k in range(args.nhold):
        b, phi = ch_form_shadow(circ, random_clifford(N, seed=rng), rng); hold.append(phi)
    hold_src = f'{len(hold)} fresh shadows (clifford_cirq.py)'
print(f'train {len(train)}  held-out: {hold_src}  ({time.time()-t0:.0f}s)', flush=True)


def shadow_estimate(phi_list, num_samples=5000):
    """Same estimator as ShadowTomographyTrainer._step (no grad): returns fidelity estimate."""
    with torch.no_grad():
        s = amp.sample(num_samples); u, c = torch.unique(s, dim=0, return_counts=True)
        w = c.type(torch.double) / num_samples
        vals = []
        for phi in phi_list:
            ca, w2, u2 = helper.samples_to_clifford_amplitudes(phi, u, w, N)
            ph = phase.amplitudes(u2, return_polar=False)
            mean = (utils.Complex(np.sqrt(w2)) * (utils.Complex(ca) * ph.conjugate())).sum(dim=0)
            vals.append((2**N + 1) * float(utils.Complex.abs(mean)**2) - 1)
    vals = np.array(vals); return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


class T(shadow_trainer.ShadowTomographyTrainer):
    def train(self):
        ex, lo, ho = [], [], []
        for i in range(self.max_iters):
            cost = self._step(i=i)
            a1 = self.nqs_amp.amplitudes(torch.tensor([1] * N)).real.detach().numpy()
            a0 = self.nqs_amp.amplitudes(torch.tensor([0] * N)).real.detach().numpy()
            amp1 = a1 * self.nqs_phase.amplitudes(torch.tensor([1] * N)).detach().numpy()
            amp0 = a0 * self.nqs_phase.amplitudes(torch.tensor([0] * N)).detach().numpy()
            infid = float(1 - np.abs(amp1 / np.sqrt(2) + amp0 / np.sqrt(2) * 1j)**2)
            ex.append(infid); lo.append(float(cost))
            if i % 20 == 0 or i == self.max_iters - 1:
                fh, se = shadow_estimate(hold); ho.append((i, 1 - fh, se))
                print(f'it {i:4d}  train-loss {float(cost):+.4f}  held-out infid est {1-fh:+.4f} +/- {se:.4f}  exact {infid:.4f}', flush=True)
        return ex, lo, ho

trainer = T(N=N, nqs_model_amp=amp, nqs_model_phase=phase, phi_list=train, optimizer=opt, max_iters=args.iters,
            num_samples=5000, batch_size=len(train), K=1, state_file_name_amp=None, state_file_name_phase=None)
t0 = time.time(); ex, lo, ho = trainer.train(); dt = time.time() - t0
ex = np.array(ex); lo = np.array(lo)
np.savetxt(f'{out}/{tag}_exact.txt', ex); np.savetxt(f'{out}/{tag}_loss.txt', lo); np.savetxt(f'{out}/{tag}_holdout.txt', np.array(ho))
fh, se = shadow_estimate(hold)
summary = {'tag': tag, 'ntrain': args.ntrain, 'nhold': len(hold), 'hold_src': hold_src, 'seed': args.seed, 'iters': args.iters,
           'runtime_s': dt, 'final_exact_infid': float(ex[-1]), 'mean_last20_exact': float(ex[-20:].mean()),
           'mean_last20_trainloss': float(lo[-20:].mean()), 'final_holdout_infid_est': 1 - fh, 'final_holdout_se': se,
           'train_bias': float(lo[-20:].mean() - ex[-20:].mean()), 'holdout_bias': float((1 - fh) - ex[-1])}
json.dump(summary, open(f'{out}/{tag}_summary.json', 'w'), indent=1)
print('DONE', json.dumps(summary), flush=True)
