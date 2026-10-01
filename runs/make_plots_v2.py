"""Week-2 figures from runs/results/v2_* and ghz40_hold_* -> runs/figures/fig6..fig10 + tables."""
import os, glob, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'); F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(F, exist_ok=True)
plt.rcParams.update({'figure.dpi': 160, 'font.size': 9})
C = {'plain': '#1f77b4', 'exact': '#d95f02', 'pretrain': '#1b9e77'}
LAB = {'plain': 'plain NSQST (self-sampled, 100 fresh shadows/it)', 'exact': 'exact-sum over 64 states (100 fresh shadows/it)',
       'pretrain': 'pre-trained amplitudes + phase net (200 fixed shadows)'}

def S(pattern):
    out = []
    for f in sorted(glob.glob(f'{R}/{pattern}_summary.json')):
        d = json.load(open(f)); d['_ex'] = np.loadtxt(f.replace('_summary.json', '_exact.txt')); d['_lo'] = np.loadtxt(f.replace('_summary.json', '_loss.txt')); out.append(d)
    return out

base = {m: [d for d in S(f'v2_{m}_seed?') if d['lamb'] == 0 and d['pcx'] == 0 and d['iters'] == 200 and (m != 'pretrain' or d['nshadow'] == 200)] for m in C}

# ---- fig6: 10-seed curves per mode + convergence summary
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for k, m in enumerate(C):
    for d in base[m]:
        ax[k].plot(d['_ex'], lw=1, alpha=.8, color=C[m])
    conv = sum(d['mean_last20'] < 0.3 for d in base[m]); n = len(base[m])
    esc = [d['escape_iter'] for d in base[m] if d['escape_iter'] is not None]
    ax[k].set(title=f'{m}: {conv}/{n} seeds converged\nescape iter median {np.median(esc) if esc else "-"}', xlabel='iteration', ylim=(-0.02, 1.02))
    ax[k].axhline(0.5, color='gray', lw=.5, ls='--')
ax[0].set_ylabel('exact infidelity'); fig.suptitle('6-qubit phase-shifted GHZ, 10 seeds per mode, 200 iterations', y=1.02)
fig.tight_layout(); fig.savefig(f'{F}/fig6_modes_10seeds.png', bbox_inches='tight'); plt.close(fig)

# ---- fig7: mode-collapse diagnostic: final p(000000), p(111111) per run
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for k, m in enumerate(C):
    for d in base[m]:
        ax.scatter(d['p000000'], d['p111111'], color=C[m], s=28, alpha=.8, label=m if d is base[m][0] else None)
ax.plot([0, 1], [1, 0], 'k:', lw=.6); ax.scatter([.5], [.5], marker='*', s=120, color='k', label='target', zorder=5)
ax.set(xlabel='final p(000000)', ylabel='final p(111111)', title='Where each run ended (corners = one branch only = infidelity 0.5)')
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{F}/fig7_branch_weights.png'); plt.close(fig)

# ---- fig8: noise re-do (exact and pretrain modes)
noise = [d for m in ('exact', 'pretrain') for d in S(f'v2_{m}_seed?_*') if (d['lamb'] > 0 or d['pcx'] > 0) and not (d.get('noisy_pretrain') and d['pcx'] > 0)]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for k, m in enumerate(('exact', 'pretrain')):
    ref = [d for d in base[m] if d['seed'] < 3]
    for d in ref: ax[k].plot(d['_ex'], color='gray', lw=.9, alpha=.6, label='noiseless' if d is ref[0] else None)
    for key, col, ls in (('lamb', '#d62728', '-'), ('pcx', '#9467bd', '--')):
        for val in sorted({d[key] for d in noise if d['mode'] == m and d[key] > 0}):
            rs = [d for d in noise if d['mode'] == m and d[key] == val]
            for d in rs: ax[k].plot(d['_ex'], color=col, lw=.9, ls=ls, alpha=.85,
                                    label=(f'depolarizing λ={val}' if key == 'lamb' else f'CNOT error p={val}') if d is rs[0] else None)
    ax[k].set(title=f'{m} mode', xlabel='iteration', ylim=(-0.02, 1.02)); ax[k].legend(fontsize=6.5)
ax[0].set_ylabel('exact infidelity'); fig.suptitle('Noise on the measured state (seeds 0-2)', y=1.02)
fig.tight_layout(); fig.savefig(f'{F}/fig8_noise_redo.png', bbox_inches='tight'); plt.close(fig)

# ---- fig9: 6-qubit pretrain: final infidelity vs number of fixed shadows; train vs held-out estimate
sw = [d for d in S('v2_pretrain_seed?') + S('v2_pretrain_seed?_n*') if d['lamb'] == 0 and d['pcx'] == 0 and d['iters'] == 200]
ns = sorted({d['nshadow'] for d in sw})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for n in ns:
    rs = [d for d in sw if d['nshadow'] == n]
    ax[0].scatter([n] * len(rs), [d['mean_last20'] for d in rs], color=C['pretrain'], s=22)
    ax[1].scatter([n] * len(rs), [d['train_shadow_fid'] - (1 - d['final_infid']) for d in rs], color='#d62728', s=22, label='train shadows' if n == ns[0] else None)
    ax[1].scatter([n] * len(rs), [d['holdout_shadow_fid'] - (1 - d['final_infid']) for d in rs], color='#1f77b4', s=22, marker='s', label='1000 held-out shadows' if n == ns[0] else None)
ax[0].set(xscale='log', yscale='log', xlabel='number of fixed shadows', ylabel='exact infidelity (mean of last 20 it)', title='N=6 pre-trained NSQST: accuracy vs shadow budget')
ax[1].axhline(0, color='k', lw=.6); ax[1].set(xscale='log', xlabel='number of fixed shadows', ylabel='shadow fidelity estimate − exact fidelity', title='Estimator bias: training set vs held-out set'); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{F}/fig9_shadow_budget_n6.png'); plt.close(fig)

# ---- fig10: long run
lg = S('v2_plain_seed0_long2000')
if lg:
    d = lg[0]; fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(d['_lo'], '.', ms=1.5, alpha=.35, color=C['plain'], label='shadow estimate'); ax[0].plot(d['_ex'], color='k', lw=1, label='exact')
    ax[0].set(xlabel='iteration', ylabel='infidelity', title=f'plain NSQST, seed 0, {len(d["_ex"])} iterations'); ax[0].legend(fontsize=7)
    ax[1].semilogy(np.clip(d['_ex'], 1e-4, None), color='k', lw=1); ax[1].set(xlabel='iteration', ylabel='exact infidelity (log)', title=f'mean of last 100: {d["_ex"][-100:].mean():.3f}, min {d["_ex"].min():.3f}')
    fig.tight_layout(); fig.savefig(f'{F}/fig10_long2000.png'); plt.close(fig)

# ---- fig11: 40-qubit held-out
h40 = []
for f in sorted(glob.glob(f'{R}/ghz40_hold_n*_summary.json')):
    d = json.load(open(f)); d['_ho'] = np.loadtxt(f.replace('_summary.json', '_holdout.txt')); d['_ex'] = np.loadtxt(f.replace('_summary.json', '_exact.txt')); d['_lo'] = np.loadtxt(f.replace('_summary.json', '_loss.txt')); h40.append(d)
if h40:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for d in sorted(h40, key=lambda x: x['ntrain']):
        l, = ax[0].plot(d['_ex'], lw=1, label=f"n_train={d['ntrain']}")
        ax[0].plot(d['_lo'], '.', ms=1.5, alpha=.3, color=l.get_color())
        ax[0].errorbar(d['_ho'][:, 0], d['_ho'][:, 1], yerr=d['_ho'][:, 2], fmt='s', ms=3, color=l.get_color(), alpha=.9)
        ax[1].scatter(d['ntrain'], d['train_bias'], color='#d62728', s=28, label='training shadows' if d is h40[0] else None)
        ax[1].scatter(d['ntrain'], d['holdout_bias'], color='#1f77b4', s=28, marker='s', label='held-out shadows' if d is h40[0] else None)
    ax[0].set(xlabel='iteration', ylabel='infidelity', title='N=40: lines exact, dots training-shadow loss, squares held-out estimate', ylim=(-0.08, 0.3)); ax[0].legend(fontsize=7)
    ax[1].axhline(0, color='k', lw=.6); ax[1].set(xscale='log', xlabel='training shadows', ylabel='estimate − exact infidelity', title='Bias of the shadow estimate (N=40)'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{F}/fig11_ghz40_holdout.png'); plt.close(fig)

# ---- tables (markdown) for the log
def fmt(x): return '-' if x is None else (f'{x:.3f}' if isinstance(x, float) else str(x))
lines = ['| mode | seeds | converged (<0.3) | escape iteration (median, range) | final infidelity, converged runs (median) | held-out shadow infidelity of final state (median) |', '|---|---|---|---|---|---|']
for m in C:
    rs = base[m]; conv = [d for d in rs if d['mean_last20'] < 0.3]; esc = [d['escape_iter'] for d in rs if d['escape_iter'] is not None]
    lines.append(f"| {m} | {len(rs)} | {len(conv)}/{len(rs)} | {fmt(float(np.median(esc))) if esc else '-'} ({min(esc) if esc else '-'}–{max(esc) if esc else '-'}) | "
                 f"{fmt(float(np.median([d['mean_last20'] for d in conv]))) if conv else '-'} | {fmt(float(np.median([1-d['holdout_shadow_fid'] for d in conv]))) if conv else '-'} |")
lines += ['', '| noise | mode | seeds | final infidelity (mean of last 20), per seed |', '|---|---|---|---|']
for m in ('exact', 'pretrain'):
    for key, name in (('lamb', 'global depolarizing λ'), ('pcx', 'CNOT depolarizing p')):
        for val in sorted({d[key] for d in noise if d['mode'] == m and d[key] > 0}):
            rs = sorted([d for d in noise if d['mode'] == m and d[key] == val], key=lambda x: x['seed'])
            lines.append(f"| {name}={val} | {m} | {len(rs)} | {', '.join(fmt(d['mean_last20']) for d in rs)} |")
lines += ['', '| fixed shadows (N=6 pretrain) | seeds | exact infidelity (mean last 20), per seed | train-set estimate − exact fid | held-out estimate − exact fid |', '|---|---|---|---|---|']
for n in ns:
    rs = sorted([d for d in sw if d['nshadow'] == n], key=lambda x: x['seed'])
    tb = ', '.join(f"{d['train_shadow_fid']-(1-d['final_infid']):+.3f}" for d in rs); hb = ', '.join(f"{d['holdout_shadow_fid']-(1-d['final_infid']):+.3f}" for d in rs)
    lines.append(f"| {n} | {len(rs)} | {', '.join(fmt(d['mean_last20']) for d in rs)} | {tb} | {hb} |")
if h40:
    lines += ['', '| N=40 training shadows | held-out set | exact infidelity (mean last 20) | training-loss bias | held-out bias |', '|---|---|---|---|---|']
    for d in sorted(h40, key=lambda x: x['ntrain']):
        lines.append(f"| {d['ntrain']} | {d['hold_src']} | {fmt(d['mean_last20_exact'])} | {d['train_bias']:+.3f} | {d['holdout_bias']:+.3f} ± {d['final_holdout_se']:.3f} |")
if lg:
    d = lg[0]; lines += ['', f"Long run (plain, seed 0, {d['iters']} it): escape at {d['escape_iter']}, mean of last 100 = {d['_ex'][-100:].mean():.3f}, min = {d['min_infid']:.3f}, runtime {d['runtime_s']/60:.0f} min"]
open(f'{R}/tables_week2.md', 'w').write('\n'.join(lines)); print('\n'.join(lines))
