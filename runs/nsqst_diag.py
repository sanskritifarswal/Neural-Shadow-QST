"""NSQST for the 6-qubit phase-shifted GHZ state with diagnostics and pluggable noise.

Two modes
  full        one TransformerWF (amplitude + phase) trained from scratch, like NSQST_GHZ_demo
  pretrained  amplitude net pre-trained on computational-basis shots of the target, then frozen;
              only a TransformerWF_phase is trained with NSQST (same idea as the 40-qubit demo)

Noise on the shadow measurements
  none            exact statevector sampling (Helper_fun.random_Clifford_shadow)
  dep:<lambda>    global depolarizing channel on the state (Helper_fun.random_Clifford_shadow_dep_n)
  ionq_harmony / ionq_aria / ionq_forte / ibm_like / ideal   gate-level Aer noise (noise_models.py)

Per iteration it records: shadow loss, exact infidelity, p(000000), p(111111), entropy of the model
distribution, number of unique samples, gradient norm, spread of the per-shadow estimates, time,
and the cumulative number of quantum shots the protocol would have used.

Usage: python runs/nsqst_diag.py --mode pretrained --noise ionq_aria --seed 0 --iters 200
"""
import sys, os, time, json, argparse
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from qiskit.quantum_info import Statevector
import nqs_models, utils
from nqs_models import NeuralQuantumState
from exact_solvers import GenericExactState
import Helper_fun as helper
import noise_models as nmod

N = 6
ALL0 = torch.zeros(N, dtype=torch.uint8); ALL1 = torch.ones(N, dtype=torch.uint8)


class CombinedNQS(NeuralQuantumState):
    """psi(s) = amp(s) * exp(i*phase(s)); samples come from the amplitude net."""
    def __init__(self, amp, phase):
        super().__init__(); self.amp = amp; self.phase = phase; self.num_sites = amp.num_sites
    def sample(self, n): return self.amp.sample(n)
    def amplitudes(self, samples, return_polar=False):
        with torch.no_grad():
            la = self.amp.amplitudes(samples, return_polar=True)
        lp = self.phase.amplitudes(samples, return_polar=True)
        out = utils.Complex(la.real + lp.real, la.imag + lp.imag)
        return out if return_polar else out.exp()


def target_state():
    circ = helper.GHZ_circuit()
    init = np.zeros(2**N, 'complex128'); init[0] = 1
    return circ, np.array(Statevector(init).evolve(circ).data)


def pretrain_amplitudes(amp, target_dense, num_shots=3000, steps=300, lr=1e-2, rng=None, log=print, counts=None):
    """Maximum likelihood on computational-basis measurements of the target: either `num_shots`
    simulated ones, or the measured `counts` dict {bitstring: count} from a device."""
    rng = rng if rng is not None else np.random.default_rng()
    if counts:
        idx = np.repeat([int(b, 2) for b in counts], list(counts.values())); num_shots = len(idx)
    else:
        probs = np.abs(target_dense)**2
        idx = rng.choice(2**N, size=num_shots, p=probs / probs.sum())
    shots = torch.tensor([[int(b) for b in format(i, f'0{N}b')] for i in idx], dtype=torch.uint8)
    opt = torch.optim.Adam(amp.parameters(), lr=lr)
    for k in range(steps):
        nll = -2 * amp.amplitudes(shots, return_polar=True).real.mean()   # -mean log |psi(s)|^2
        opt.zero_grad(); nll.backward(); opt.step()
        if k % 50 == 0 or k == steps - 1:
            with torch.no_grad():
                p0 = (amp.amplitudes(ALL0[None], return_polar=True).real.exp()**2).item()
                p1 = (amp.amplitudes(ALL1[None], return_polar=True).real.exp()**2).item()
            log(f'pretrain step {k}: nll {nll.item():.4f}  p(0..0)={p0:.3f} p(1..1)={p1:.3f}')
    return num_shots


def make_shadow_fn(noise, rng, stats):
    if noise == 'none':
        return lambda c, k: helper.random_Clifford_shadow(c, k)
    if noise.startswith('dep:'):
        lamb = float(noise.split(':')[1])
        return lambda c, k: helper.random_Clifford_shadow_dep_n(c, k, lamb)
    return lambda c, k: nmod.random_Clifford_shadow_aer(c, k, noise, rng=rng, stats=stats)


def train(nqs, params, circuit, target, shadow_fn, iters, batch_size, num_samples, lr, log=print, shots_before=0):
    opt = torch.optim.Adam(params, lr=lr)
    exact_target = GenericExactState(utils.Complex(target))
    hist = {k: [] for k in ['loss', 'exact', 'p0', 'p1', 'entropy', 'n_unique', 'grad_norm', 'pred_std', 'frac_zero_overlap', 'iter_time', 'shots']}
    for it in range(iters):
        t0 = time.time()
        with torch.no_grad():
            samples = nqs.sample(num_samples)
            uniq, counts = torch.unique(samples, dim=0, return_counts=True)
            weights = counts.type(torch.double) / num_samples
        cliffords, results = shadow_fn(circuit, batch_size)
        preds, loss, n_zero = [], 0, 0
        for cl, res in zip(cliffords, results):
            with torch.no_grad():
                mat = cl.adjoint().to_matrix()
                bit = next(iter(res))
                Ub = mat[:, int(bit, 2)]
                cl_log, w_new, u_new = helper.samples_to_log_clifford_amplitudes(Ub, uniq, weights)
                if len(w_new) == 1 and float(w_new[0]) == 1.0 and len(uniq) > 1: n_zero += 1
                nn_log = nqs.amplitudes(u_new, return_polar=True)
                frac = (utils.Complex(cl_log) - nn_log).exp()
                frac_c = frac.conjugate()
                mean = (utils.Complex(w_new) * frac).sum(dim=0)
                preds.append(float((2**N + 1) * (utils.Complex.abs(mean)**2).numpy() - 1))
            la = nqs.amplitudes(u_new, return_polar=True)
            t1r = (w_new * (frac_c.real * la.real - frac_c.imag * la.imag)).sum(dim=0)
            t1i = (w_new * (frac_c.real * la.imag + frac_c.imag * la.real)).sum(dim=0)
            loss = loss - 2 * (2**N + 1) * (t1r * mean.real - t1i * mean.imag) / batch_size
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 5))
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            full = nqs.full_state()
            infid = float((1 - exact_target.fidelity_to(full)).numpy())
            p = (utils.Complex.abs(full.state)**2).numpy()
            ent = float(-(p[p > 1e-12] * np.log(p[p > 1e-12])).sum())
        preds = np.array(preds)
        for k, v in [('loss', 1 - preds.mean()), ('exact', infid), ('p0', float(p[0])), ('p1', float(p[-1])), ('entropy', ent),
                     ('n_unique', int(len(uniq))), ('grad_norm', gn), ('pred_std', float(preds.std())),
                     ('frac_zero_overlap', n_zero / batch_size), ('iter_time', time.time() - t0),
                     ('shots', shots_before + (it + 1) * batch_size)]:
            hist[k].append(v)
        log(f'it {it:4d} loss {hist["loss"][-1]:+.3f} exact {infid:.4f} p0 {p[0]:.3f} p1 {p[-1]:.3f} S {ent:.2f} |g| {gn:.2f} {hist["iter_time"][-1]:.1f}s')
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='pretrained', choices=['full', 'pretrained'])
    ap.add_argument('--noise', default='none'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=200); ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--samples', type=int, default=5000); ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--pre_shots', type=int, default=3000); ap.add_argument('--reuse', action='store_true', help='draw shadows once and reuse every iteration')
    ap.add_argument('--shadow_file', default=None, help='JSON from runs/hardware/collect_shadows.py; implies --reuse and uses its CB counts for pre-training')
    ap.add_argument('--pre_ideal', action='store_true', help='with --shadow_file: pre-train on simulated noiseless CB shots instead of the measured counts')
    ap.add_argument('--pre_threshold', type=float, default=0.0, help='with --shadow_file: drop measured CB bitstrings whose frequency is below this fraction before pre-training (crude readout-error filter)')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    tag = a.tag or f'diag_{a.mode}_{a.noise.replace(":", "")}_seed{a.seed}' + ('_reuse' if a.reuse else '')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'); os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    import random; random.seed(a.seed)
    helper.random_Clifford_shadow = lambda c, k, _r=rng: _rcs(c, k, _r)
    def _rcs(circuit, k, r):  # seeded version of Helper_fun.random_Clifford_shadow
        cls = [__import__('qiskit').quantum_info.random_clifford(N, seed=r) for _ in range(k)]
        res = []
        for cl in cls:
            sv = Statevector.from_int(0, 2**N).evolve(circuit.compose(cl.to_circuit()))
            sv.seed(int(r.integers(2**31))); res.append(sv.sample_counts(1))
        return cls, res

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hardware'))
    circuit, target = target_state()
    stats = []; shots_before = 0
    if a.mode == 'full':
        amp = nqs_models.TransformerWF(num_sites=N, num_layers=2, internal_dimension=8, num_heads=4, dropout=0.0)
        nqs, params = amp, list(amp.parameters())
    else:
        amp = nqs_models.TransformerWF(num_sites=N, num_layers=2, internal_dimension=8, num_heads=4, dropout=0.0, phase_mode=0)
        if a.shadow_file and not a.pre_ideal:
            import load_hw_shadows as _l; _cb = _l.load(a.shadow_file)[2]['cb_counts']
            if a.pre_threshold > 0:
                tot = sum(_cb.values()); _cb = {b: c for b, c in _cb.items() if c / tot >= a.pre_threshold}
                print(f'pre-training keeps {len(_cb)} bitstrings / {sum(_cb.values())} shots after threshold {a.pre_threshold}')
            shots_before = pretrain_amplitudes(amp, target, num_shots=a.pre_shots, rng=rng, counts=_cb)
        else:
            shots_before = pretrain_amplitudes(amp, target, num_shots=a.pre_shots, rng=rng)
        for p_ in amp.parameters(): p_.requires_grad_(False)
        phase = nqs_models.TransformerWF_phase(num_sites=N, num_layers=2, internal_dimension=8, num_heads=4, dropout=0.0, phase_mode=2)
        nqs, params = CombinedNQS(amp, phase), list(phase.parameters())
    shadow_fn = make_shadow_fn(a.noise, rng, stats)
    if a.shadow_file:
        import load_hw_shadows
        shadow_fn = load_hw_shadows.as_shadow_fn(a.shadow_file); a.reuse = True
    if a.reuse:
        fixed = shadow_fn(circuit, a.batch)
        shadow_fn = lambda c, k: fixed
    t0 = time.time()
    hist = train(nqs, params, circuit, target, shadow_fn, a.iters, a.batch, a.samples, a.lr, shots_before=shots_before)
    if a.reuse: hist['shots'] = [shots_before + a.batch] * a.iters
    hist.update(dict(mode=a.mode, noise=a.noise, seed=a.seed, iters=a.iters, batch=a.batch, samples=a.samples, pre_shots=shots_before,
                     reuse=a.reuse, runtime_s=time.time() - t0, final_exact=hist['exact'][-1], mean_last20=float(np.mean(hist['exact'][-20:])),
                     gate_stats={k: float(np.mean([s[k] for s in stats])) for k in stats[0]} if stats else None))
    with torch.no_grad():
        st = nqs.full_state().state; np.save(f'{out}/{tag}_final_state.npy', st.real.numpy().astype(complex) + 1j * st.imag.numpy())
    json.dump(hist, open(f'{out}/{tag}.json', 'w'))
    print('DONE', tag, 'final', hist['final_exact'], 'last20', hist['mean_last20'], 'runtime', hist['runtime_s'])
