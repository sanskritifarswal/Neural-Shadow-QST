"""Diagnostic figures from runs/results/diag_*.json -> runs/figures/fig12..fig16"""
import os, glob, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'); F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
plt.rcParams.update({'figure.dpi': 160, 'font.size': 8.5})
NOISE_ORDER = ['none', 'dep:0.1', 'dep:0.3', 'ionq_forte', 'ionq_aria', 'ionq_harmony', 'ibm_like', 'fake_torino']
LABEL = {'none': 'noiseless', 'dep:0.1': 'depolarizing 0.1', 'dep:0.3': 'depolarizing 0.3', 'ionq_forte': 'IonQ Forte-like', 'ionq_aria': 'IonQ Aria-like',
         'ionq_harmony': 'IonQ Harmony-like', 'ibm_like': 'IBM-like (linear chain)', 'fake_torino': 'FakeTorino (real calibration)'}
import re
def parse_log(path):
    """Partial run: rebuild the history from the per-iteration log lines."""
    name = os.path.basename(path)[:-4]
    suffix = '_preideal' if '_preideal' in name else '_thresh10' if '_thresh10' in name else '_thresh' if '_thresh' in name else ''
    name = name.replace(suffix, '', 1) if suffix else name
    m = re.match(r'mode(full|pretrained)noise(.+?)seed(\d+)', name)
    mode, noise, seed = m.group(1), m.group(2).replace('dep_', 'dep:').rstrip('_'), int(m.group(3))
    tag = 'diag_' + mode + '_' + noise.replace(':', '') + suffix + f'_seed{seed}'
    h = {k: [] for k in ['loss', 'exact', 'p0', 'p1', 'entropy', 'grad_norm', 'iter_time']}
    for line in open(path):
        mm = re.match(r'it\s+\d+ loss ([+-][\d.]+) exact ([\d.]+) p0 ([\d.]+) p1 ([\d.]+) S ([\d.]+) \|g\| ([\d.]+) ([\d.]+)s', line)
        if mm:
            for k, v in zip(h, mm.groups()): h[k].append(float(v))
    if not h['exact']: return None
    n = len(h['exact'])
    h.update(n_unique=[np.nan] * n, pred_std=[np.nan] * n, frac_zero_overlap=[np.nan] * n, shots=[np.nan] * n,
             mode=mode, noise=noise, seed=seed, iters=n, partial=True, final_exact=h['exact'][-1], mean_last20=float(np.mean(h['exact'][-20:])),
             gate_stats=None, file=tag + '.json')
    return h
runs = [json.load(open(f)) | {'file': os.path.basename(f), 'partial': False} for f in sorted(glob.glob(f'{R}/diag_*.json'))]
done = {r['file'] for r in runs}
for lg in sorted(glob.glob(f'{R}/mode*.log')):
    h = parse_log(lg)
    if h and h['file'] not in done: runs.append(h)
pre = [r for r in runs if r['mode'] == 'pretrained']; full = [r for r in runs if r['mode'] == 'full']
sm = lambda x, k=5: np.convolve(x, np.ones(k) / k, mode='valid')

# fig7: noise sweep (pre-trained), exact infidelity + loss bias
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
keys = [k for k in NOISE_ORDER if any(r['noise'] == k for r in pre)]
cols = plt.cm.viridis(np.linspace(0, .9, len(keys)))
for k, c in zip(keys, cols):
    rs = [r for r in pre if r['noise'] == k]
    L = min(len(r['exact']) for r in rs); ex = np.mean([r['exact'][:L] for r in rs], axis=0); lo = np.mean([r['loss'][:L] for r in rs], axis=0)
    ax[0].semilogy(sm(ex), color=c, label=f'{LABEL[k]} (n={len(rs)}{", partial" if any(r["partial"] for r in rs) else ""})'); ax[1].plot(sm(lo), color=c)
    ax[2].errorbar(keys.index(k), np.mean([np.mean(r['exact'][-50:]) for r in rs]), yerr=np.std([np.mean(r['exact'][-50:]) for r in rs]) if len(rs) > 1 else 0, fmt='o', color=c)
    ax[2].plot(keys.index(k), np.mean([np.mean(r['loss'][-50:]) for r in rs]), 'x', color=c, ms=8)
ax[0].set(title='Exact infidelity (5-iter moving avg), pre-trained amplitudes', xlabel='iteration', ylabel='exact infidelity'); ax[0].legend(fontsize=6)
ax[1].set(title='Loss = shadow estimate of infidelity (noisy device data)', xlabel='iteration', ylabel='loss'); ax[1].axhline(0, color='k', lw=.5)
ax[2].set(title='Last 50 iterations: exact (dot) vs loss (x)', ylabel='infidelity'); ax[2].set_xticks(range(len(keys))); ax[2].set_xticklabels([LABEL[k] for k in keys], rotation=35, ha='right', fontsize=6.5)
ax[2].axhline(0, color='k', lw=.5)
fig.tight_layout(); fig.savefig(f'{F}/fig13_noise_sweep_pretrained.png'); plt.close(fig)

# fig8: collapse diagnostics, from-scratch vs pre-trained
fig, ax = plt.subplots(2, 3, figsize=(13, 6))
for r in full:
    lab = f"from scratch, seed {r['seed']}"
    ax[0, 0].plot(r['p0'], label=lab + ' p(000000)'); ax[0, 0].plot(r['p1'], '--', label=lab + ' p(111111)')
    ax[0, 1].plot(r['entropy'], label=lab); ax[0, 2].plot(r['n_unique'], label=lab)
    ax[1, 0].semilogy(r['grad_norm'], label=lab); ax[1, 1].plot(r['frac_zero_overlap'], label=lab); ax[1, 2].semilogy(r['exact'], label=lab)
for r in [x for x in pre if x['noise'] == 'none'][:1]:
    lab = 'pre-trained amplitudes'
    ax[0, 0].plot(r['p0'], 'k', label=lab + ' p(000000)'); ax[0, 0].plot(r['p1'], 'k--', label=lab + ' p(111111)')
    ax[0, 1].plot(r['entropy'], 'k', label=lab); ax[0, 2].plot(r['n_unique'], 'k', label=lab)
    ax[1, 0].semilogy(r['grad_norm'], 'k', label=lab); ax[1, 1].plot(r['frac_zero_overlap'], 'k', label=lab); ax[1, 2].semilogy(r['exact'], 'k', label=lab)
ax[0, 0].set(title='Probability of the two GHZ branches', ylabel='probability'); ax[0, 0].axhline(.5, color='gray', lw=.5); ax[0, 0].legend(fontsize=5.5)
ax[0, 1].set(title='Entropy of the model distribution (GHZ = ln 2 = 0.69)', ylabel='nats'); ax[0, 1].axhline(np.log(2), color='gray', lw=.5); ax[0, 1].legend(fontsize=6)
ax[0, 2].set(title='Distinct bitstrings among 5000 samples', ylabel='count')
ax[1, 0].set(title='Gradient norm before clipping', xlabel='iteration'); ax[1, 1].set(title='Fraction of shadows with no overlap with any sample', xlabel='iteration')
ax[1, 2].set(title='Exact infidelity', xlabel='iteration')
fig.tight_layout(); fig.savefig(f'{F}/fig14_collapse_diagnostics.png'); plt.close(fig)

# fig9: per-shadow estimate spread and time per iteration vs noise
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for k, c in zip(keys, cols):
    rs = [r for r in pre if r['noise'] == k]
    rs2 = [r for r in rs if not r['partial']]
    if rs2: ax[0].plot(sm(np.mean([r['pred_std'] for r in rs2], axis=0)), color=c, label=LABEL[k])
    ax[1].bar(keys.index(k), np.mean([np.mean(r['iter_time']) for r in rs]), color=c)
ax[0].set(title='Std of the 100 per-shadow fidelity estimates', xlabel='iteration', ylabel='std'); ax[0].legend(fontsize=6)
ax[1].set(title='Classical time per iteration (laptop, 7 jobs in parallel)', ylabel='seconds'); ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels([LABEL[k] for k in keys], rotation=35, ha='right', fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{F}/fig15_shadow_spread_and_time.png'); plt.close(fig)

# fig10: runs on collected (hardware-format) shadows
hw = [r for r in pre if 'torino' in r['file']]
HW = {'diag_pretrained_fake_torino_seed0.json': 'pre-train on the measured (noisy) basis counts',
      'diag_pretrained_fake_torino_thresh_seed0.json': 'same, drop bitstrings under 2%',
      'diag_pretrained_fake_torino_thresh10_seed0.json': 'same, drop bitstrings under 10%',
      'diag_pretrained_fake_torino_preideal_seed0.json': 'pre-train on noiseless basis counts (control)'}
if hw:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for r in hw:
        lab = HW.get(r['file'], r['file'])
        l, = ax[0].plot(r['exact'], label=lab); ax[1].plot(r['loss'], '.', ms=2, color=l.get_color(), label=lab)
    ax[0].set(title='FakeTorino shadows (200, reused): exact infidelity to the ideal GHZ', xlabel='iteration', ylabel='exact infidelity'); ax[0].legend(fontsize=6.5); ax[0].set_ylim(bottom=0)
    ax[1].set(title='loss reported during training', xlabel='iteration', ylabel='loss'); ax[1].axhline(0, color='k', lw=.5)
    fig.tight_layout(); fig.savefig(f'{F}/fig16_fake_torino_run.png'); plt.close(fig)

print(f'{"run":48s} {"final":>7s} {"last50":>7s} {"loss50":>7s} {"min":>7s} {"s/it":>5s} 2q-gates')
for r in runs:
    print(f'{r["file"][:-5]:48s} {r["final_exact"]:7.4f} {np.mean(r["exact"][-50:]):7.4f} {np.mean(r["loss"][-50:]):+7.3f} {min(r["exact"]):7.4f} {np.mean(r["iter_time"]):5.1f} {r["gate_stats"]["two_qubit_gates"] if r.get("gate_stats") else "-"}')
