"""Make all figures from runs/results/*.txt -> runs/figures/*.png"""
import os, glob, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), 'results'); F = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(F, exist_ok=True)
plt.rcParams.update({'figure.dpi': 160, 'font.size': 9})
COL = ['#1f77b4', '#d95f02', '#1b9e77', '#7570b3']

def curves(prefix):
    out = []
    for f in sorted(glob.glob(f'{R}/{prefix}*_exact.txt')):
        out.append((os.path.basename(f)[:-10], np.loadtxt(f), np.loadtxt(f.replace('_exact', '_loss'))))
    return out

def panel(runs, title, fname, ref=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, (tag, ex, lo) in enumerate(runs):
        c = COL[i % 4]
        ax[0].plot(lo, '.', ms=2.5, color=c, alpha=.45)
        ax[0].plot(ex, '-', lw=1.4, color=c, label=tag)
        ax[1].semilogy(np.clip(ex, 1e-4, None), '-', lw=1.2, color=c, label=tag)
    if ref is not None:
        ax[0].plot(ref, 'k--', lw=1, label="author's saved run"); ax[1].semilogy(ref, 'k--', lw=1, label="author's saved run")
    ax[0].axhline(0, color='gray', lw=.5)
    ax[0].set(xlabel='Iteration', ylabel='Infidelity', title=title + '\nlines: exact, dots: shadow estimate (loss)')
    ax[1].set(xlabel='Iteration', ylabel='Exact infidelity (log)', title='Exact infidelity, log scale')
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{F}/{fname}'); plt.close(fig)

r6 = [r for r in curves('ghz6_seed')]
if r6: panel(r6, '6-qubit phase-shifted GHZ, NSQST (100 shadows/iter)', 'fig1_ghz6_training.png')
r40 = curves('ghz40_seed')
if r40:
    ref = np.loadtxt(os.path.join(R, '..', '..', 'data', 'exact_GHZ_pre_phase_N40'))
    panel(r40, '40-qubit GHZ, NSQST w/ pre-training (200 fixed shadows)', 'fig2_ghz40_training.png', ref=ref)
rn = curves('ghz6_noise')
if rn: panel(curves('ghz6_seed0') + rn, '6-qubit GHZ: noiseless vs depolarized shadows', 'fig3_ghz6_noise.png')

# estimator quality: loss - exact
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, runs, t in [(ax[0], r6, 'N=6 (fresh shadows each iter)'), (ax[1], r40, 'N=40 (same 200 shadows reused)')]:
    for i, (tag, ex, lo) in enumerate(runs):
        a.plot(lo - ex, '.', ms=3, color=COL[i], alpha=.6, label=f'{tag}: mean {np.mean(lo-ex):+.3f}, std {np.std(lo-ex):.3f}')
    a.axhline(0, color='k', lw=.6); a.set(xlabel='Iteration', ylabel='estimate - exact', title=t); a.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{F}/fig4_estimator_error.png'); plt.close(fig)

# learned state vs target, N=6
tf = f'{R}/ghz6_target_state.npy'
if os.path.exists(tf):
    tgt = np.load(tf); fs = sorted(glob.glob(f'{R}/ghz6_*_final_state.npy'))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2)); w = .8 / (len(fs) + 1); x = np.arange(64)
    ax[0].bar(x - .4, np.abs(tgt)**2, w, color='k', label='target')
    for i, f in enumerate(fs):
        p = np.load(f); p = p / np.linalg.norm(p)
        ax[0].bar(x - .4 + (i + 1) * w, np.abs(p)**2, w, color=COL[i % 4], label=os.path.basename(f)[:-16])
        rel = np.angle(p[63] / p[0]); ax[1].bar(i + 1, rel / np.pi, color=COL[i % 4])
    ax[1].bar(0, np.angle(tgt[63] / tgt[0]) / np.pi, color='k')
    ax[1].set_xticks(range(len(fs) + 1)); ax[1].set_xticklabels(['target'] + [os.path.basename(f)[5:-16] for f in fs], rotation=20, fontsize=7)
    ax[0].set(xlabel='basis state index', ylabel='probability', title='Learned |amplitude|^2 (N=6)'); ax[0].legend(fontsize=6.5)
    ax[1].set(ylabel='relative phase arg(a_111111 / a_000000) / pi', title='Learned GHZ relative phase')
    fig.tight_layout(); fig.savefig(f'{F}/fig5_ghz6_learned_state.png'); plt.close(fig)

for f in sorted(glob.glob(f'{R}/*_summary.json')):
    d = json.load(open(f)); d.pop('amp_net_unique_samples', None); print(os.path.basename(f), d)
