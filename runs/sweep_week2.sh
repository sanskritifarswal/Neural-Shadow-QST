#!/bin/bash
# Week-2 sweep. 6 jobs in parallel, 2 torch threads each. Each job writes runs/results/<tag>.log itself
# and is skipped if its summary already exists (safe to re-run after an interruption).
cd "$(dirname "$0")/.." || exit 1
source ~/opt/anaconda3/etc/profile.d/conda.sh && conda activate nsqst
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
J=runs/results/jobs_week2.txt; : > $J
run() { echo "runs/run_ghz6_v2.py --threads 2 $*" >> $J; }

# 1. mode comparison, 10 seeds each
for s in 0 1 2 3 4 5 6 7 8 9; do run --mode plain --seed $s; done
for s in 0 1 2 3 4 5 6 7 8 9; do run --mode exact --seed $s; done
for s in 0 1 2 3 4 5 6 7 8 9; do run --mode pretrain --seed $s --nshadow 200; done
# 2. noise re-do on the pre-trained mode (the only mode that converges reliably)
for l in 0.1 0.3 0.5; do for s in 0 1 2 3 4; do run --mode pretrain --seed $s --nshadow 200 --lamb $l; done; done
for s in 0 1 2; do run --mode pretrain --seed $s --nshadow 200 --lamb 0.3 --noisy_pretrain; done
for p in 0.01 0.05; do for s in 0 1 2; do run --mode pretrain --seed $s --nshadow 200 --pcx $p; run --mode pretrain --seed $s --nshadow 200 --pcx $p --noisy_pretrain; done; done
# 3. shadow budget
for n in 25 50 100 400; do for s in 0 1 2; do run --mode pretrain --seed $s --nshadow $n --tag v2_pretrain_seed${s}_n${n}; done; done

echo "$(wc -l < $J) jobs"; date
xargs -P 6 -L 1 python -u < $J > /dev/null 2>&1
echo ALL DONE; date
