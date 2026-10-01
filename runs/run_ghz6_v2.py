"""6-qubit phase-shifted GHZ NSQST, version 2: seeded, with three training modes.

  --mode plain     original NSQST (5000 self-samples, 100 fresh shadows / iteration) but
                   bit-reproducible (seeded Cliffords + seeded outcomes, no random.seed()).
  --mode exact     same, but the Monte-Carlo average over self-samples is replaced by the exact
                   sum over all 64 basis states  -> removes the sampling side of mode collapse.
  --mode pretrain  NSQST with pre-training (as the 40-qubit demo): amplitude net fit to
                   computational-basis measurements first, then the phase net is trained with
                   NSQST on a FIXED set of --nshadow shadows.

Noise: --lamb L  global depolarizing on the measured state;  --pcx P  Aer depolarizing error
of probability P on every CNOT of the transpiled measurement circuit (+ GHZ prep).

Outputs in runs/results/: <tag>_exact.txt, <tag>_loss.txt, <tag>_final_state.npy,
<tag>_summary.json (includes a held-out shadow estimate of the final fidelity from 1000 fresh
shadows evaluated with exact overlaps).
"""
import argparse, os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--mode', choices=['plain', 'exact', 'pretrain'], default='plain')
ap.add_argument('--lamb', type=float, default=0.0)
ap.add_argument('--pcx', type=float, default=0.0)
ap.add_argument('--nshadow', type=int, default=100, help='shadows per iteration (plain/exact) or fixed set size (pretrain)')
ap.add_argument('--nmeas', type=int, default=3000, help='computational-basis shots for pre-training')
ap.add_argument('--noisy_pretrain', action='store_true', help='pre-training shots also pass through the noise (hardware-realistic)')
ap.add_argument('--tag', default=None)
ap.add_argument('--threads', type=int, default=2)
args = ap.parse_args()

import numpy as np, torch
torch.set_num_threads(args.threads)
from qiskit.quantum_info import Statevector
import nqs_models, utils
from exact_solvers import GenericExactState
import Helper_fun as helper
from nsqst_ext import (seeded_shadow_fn, SeededTrainer, dense_state, pretrain_amplitudes, DensePhi,
                       PreTrainedTrainer, dense_phi, shadow_fidelity_estimate, flatten_shadows)

tag = args.tag or f'v2_{args.mode}_seed{args.seed}' + (f'_lamb{args.lamb}' if args.lamb else '') + (f'_pcx{args.pcx}' if args.pcx else '') + ('_np' if args.noisy_pretrain else '')
out = os.path.join(HERE, 'results'); os.makedirs(out, exist_ok=True)
if os.path.exists(f'{out}/{tag}_summary.json') and not os.environ.get('NSQST_FORCE'):
    print('skip (already done):', tag); sys.exit(0)

class _Tee:                      # copy stdout/stderr to runs/results/<tag>.log
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()
sys.stdout = sys.stderr = _Tee(f'{out}/{tag}.log')
torch.manual_seed(args.seed); np.random.seed(args.seed)
rng = np.random.default_rng(1000 + args.seed)        # shadows / outcomes
rng_pre = np.random.default_rng(2000 + args.seed)    # computational-basis shots for pre-training
rng_hold = np.random.default_rng(3000 + args.seed)   # held-out shadows for final evaluation

N = 6
circ = helper.GHZ_circuit()
target_dense = np.array(Statevector.from_int(0, 2**N).evolve(circ).data)

noise_model = None
if args.pcx > 0:
    import qiskit.providers.aer.noise as noise
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(args.pcx, 2), ['cx'])
shadow_fn = seeded_shadow_fn(N, rng, lamb=args.lamb, noise_model=noise_model)

t0 = time.time(); extra = {}
if args.mode in ('plain', 'exact'):
    nqs = nqs_models.TransformerWF(num_sites=N, num_layers=2, internal_dimension=8, num_heads=4, dropout=0.0)
    opt = torch.optim.Adam(nqs.parameters(), lr=1e-2)
    trainer = SeededTrainer(nqs_model=nqs, circuit=circ, optimizer=opt, batch_size=args.nshadow,
                            target_state=GenericExactState(utils.Complex(target_dense)),
                            max_iters=args.iters, num_samples=5000, K=1, state_file_name=None,
                            shadow_fn=shadow_fn, exact_weights=(args.mode == 'exact'))
    exact, loss = trainer.train()
    psi = dense_state(nqs)
    n_shadows_total = args.nshadow * args.iters
else:
    amp = nqs_models.TransformerWF(num_sites=N, num_layers=2, internal_dimension=8, num_heads=4, dropout=0.0, phase_mode=0)
    print('pre-training amplitude net on', args.nmeas, 'computational-basis shots', flush=True)
    pre_source = target_dense
    if args.noisy_pretrain and args.lamb > 0:      # depolarized computational-basis distribution
        p = (1 - args.lamb) * np.abs(target_dense)**2 + args.lamb / 2**N
        pre_source = np.sqrt(p)                    # only |.|^2 is used for sampling
    elif args.noisy_pretrain and noise_model is not None:   # Aer: GHZ prep with noisy CNOTs, Z-basis shots
        from qiskit import execute, Aer
        qc = circ.copy(); qc.measure_all()
        cnt = execute(qc, Aer.get_backend('qasm_simulator'), basis_gates=noise_model.basis_gates, noise_model=noise_model,
                      shots=args.nmeas, seed_simulator=int(rng_pre.integers(2**31))).result().get_counts(0)
        p = np.zeros(2**N)
        for b, c in cnt.items(): p[int(b, 2)] = c
        pre_source = np.sqrt(p / p.sum())
    pre_hist = pretrain_amplitudes(amp, pre_source, rng_pre, n_meas=args.nmeas)
    with torch.no_grad():
        p_amp = amp.full_state().state.norm2().numpy()
    extra['pretrain_nll_final'] = pre_hist[-1]
    extra['pretrain_p000000'] = float(p_amp[0]); extra['pretrain_p111111'] = float(p_amp[-1])
    print(f'  amplitude net: p(000000)={p_amp[0]:.3f} p(111111)={p_amp[-1]:.3f} other={1-p_amp[0]-p_amp[-1]:.3e}', flush=True)
    cl, res = shadow_fn(circ, args.nshadow)
    train_shadows = flatten_shadows(cl, res)
    phi_list = [DensePhi(dense_phi(c, b)) for c, b in train_shadows]
    phase = nqs_models.TransformerWF_phase(num_sites=N, num_layers=2, internal_dimension=8, num_heads=4, dropout=0.0, phase_mode=2)
    opt = torch.optim.Adam(phase.parameters(), lr=1e-2)
    trainer = PreTrainedTrainer(N=N, nqs_model_amp=amp, nqs_model_phase=phase, phi_list=phi_list, optimizer=opt,
                                max_iters=args.iters, num_samples=5000, batch_size=len(phi_list), K=1,
                                state_file_name_amp=None, state_file_name_phase=None, target_dense=target_dense)
    exact, loss = trainer.train()
    psi = trainer.dense_state()
    n_shadows_total = args.nshadow
    f_train, se_train = shadow_fidelity_estimate(psi, train_shadows)
    extra['train_shadow_fid'] = f_train; extra['train_shadow_fid_se'] = se_train
dt = time.time() - t0

# held-out evaluation: 1000 fresh noiseless shadows, exact overlaps (this is what one would
# report on hardware, where the exact infidelity is unavailable)
hold_fn = seeded_shadow_fn(N, rng_hold)
f_hold, se_hold = shadow_fidelity_estimate(psi, flatten_shadows(*hold_fn(circ, 1000)))

exact = np.array(exact, float); loss = np.array(loss, float)
np.savetxt(f'{out}/{tag}_exact.txt', exact); np.savetxt(f'{out}/{tag}_loss.txt', loss)
np.save(f'{out}/{tag}_final_state.npy', psi)
below = np.where(exact < 0.3)[0]
summary = {'tag': tag, 'mode': args.mode, 'seed': args.seed, 'iters': args.iters, 'lamb': args.lamb, 'pcx': args.pcx,
           'nshadow': args.nshadow, 'shadows_total': n_shadows_total, 'runtime_s': dt, 'noisy_pretrain': args.noisy_pretrain,
           'final_infid': float(exact[-1]), 'min_infid': float(exact.min()), 'mean_last20': float(exact[-20:].mean()),
           'escape_iter': int(below[0]) if len(below) else None,
           'p000000': float(abs(psi[0])**2), 'p111111': float(abs(psi[-1])**2),
           'holdout_shadow_fid': f_hold, 'holdout_shadow_fid_se': se_hold, **extra}
json.dump(summary, open(f'{out}/{tag}_summary.json', 'w'), indent=1)
print('DONE', json.dumps(summary), flush=True)
