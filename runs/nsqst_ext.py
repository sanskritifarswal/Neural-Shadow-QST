"""Extensions to the NSQST trainer used by runs/run_ghz6_v2.py.

Nothing in the original repo is modified. This module provides:

  seeded_shadow_fn(...)      reproducible Clifford shadows (seeded random_clifford + seeded
                             bitstring sampling), optionally through an n-qubit depolarizing
                             channel (lamb) or an Aer gate-level noise model (cx depolarizing).
  SeededTrainer              NSQST_Trainer with (a) no random.seed() call, (b) a pluggable
                             shadow function, (c) an optional *exact-sum* mode that replaces
                             the 5000 self-samples by all 2^N basis states with exact weights.
  pretrain_amplitudes(...)   NNQST-style pre-training of the amplitude network on
                             computational-basis measurements of the target (the step the
                             40-qubit demo relies on but whose code is not in the repo).
  DensePhi / PreTrainedTrainer
                             phase-only NSQST (the repo's ShadowTomographyTrainer) driven by
                             dense stabilizer states instead of Cirq CH-form objects, with the
                             exact infidelity computed from the full 2^N vector.
  shadow_fidelity_estimate() unbiased shadow estimate of fidelity from held-out shadows using
                             exact dense overlaps (no NN sampling).
"""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
import numpy as np, torch
import qiskit
from qiskit.quantum_info import Statevector, random_clifford
import utils
from exact_solvers import GenericExactState
import Helper_fun as helper
import NSQST_Trainer as nsqst_trainer
import NSQST_Pre_Trainer_GHZ as pre_trainer


# ----------------------------------------------------------------------------- shadows
def _bitstr(idx, N):
    return format(int(idx), f'0{N}b')


def seeded_shadow_fn(N, rng, lamb=0.0, noise_model=None, aer_seed=None):
    """Return f(circuit, num_shadow) -> (cliffords, results) with the same output format as
    Helper_fun.random_Clifford_shadow, but fully seeded by `rng` (np.random.Generator).
    lamb > 0      : sample from (1-lamb) rho + lamb I/2^N  (global depolarizing, as
                    Helper_fun.random_Clifford_shadow_dep_n)
    noise_model   : qiskit-aer NoiseModel applied to the transpiled circuit (gate-level noise,
                    as Helper_fun.random_Clifford_shadow_noisy)
    """
    def f(circuit, num_shadow):
        cliffords = [random_clifford(N, seed=rng) for _ in range(num_shadow)]
        results = []
        if noise_model is not None:
            from qiskit import execute, Aer
            backend = Aer.get_backend('qasm_simulator')
            for cliff in cliffords:
                qc = circuit.compose(qiskit.compiler.transpile(cliff.to_circuit(),
                                                               basis_gates=['rx', 'ry', 'rz', 'cx']))
                qc.measure_all()
                res = execute(qc, backend, basis_gates=noise_model.basis_gates, noise_model=noise_model,
                              shots=1, memory=False, seed_simulator=int(rng.integers(2**31))).result()
                results.append(res.get_counts(0))
            return cliffords, results
        for cliff in cliffords:
            sv = Statevector.from_int(0, 2**N).evolve(circuit.compose(cliff.to_circuit()))
            p = np.abs(sv.data)**2
            if lamb > 0:
                p = (1 - lamb) * p + lamb / 2**N
            p = p / p.sum()
            b = rng.choice(2**N, p=p)
            results.append({_bitstr(b, N): 1})
        return cliffords, results
    return f


def dense_phi(cliff, bitstring):
    """Stabilizer state U^dag |b> as a dense vector (matches NSQST_Trainer's `Ub`)."""
    return cliff.adjoint().to_matrix()[:, int(bitstring, 2)]


def shadow_fidelity_estimate(psi_dense, shadows):
    """Unbiased shadow estimate of <psi|rho|psi> from (clifford, bitstring) pairs, using the
    exact overlap <phi_i|psi> (no NN sampling). Returns (mean, std_err)."""
    N = int(np.log2(len(psi_dense)))
    vals = []
    for cliff, b in shadows:
        phi = dense_phi(cliff, b)
        vals.append((2**N + 1) * abs(np.vdot(phi, psi_dense))**2 - 1)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


def flatten_shadows(cliffords, results):
    return [(c, next(iter(r.keys()))) for c, r in zip(cliffords, results)]


# ----------------------------------------------------------------------------- trainer
class SeededTrainer(nsqst_trainer.NSQST_Trainer):
    """NSQST_Trainer with reproducible shadows and an optional exact-sum mode.

    exact_weights=True: instead of drawing 5000 samples from the model, use all 2^N basis
    states with weights |psi(s)|^2. The estimator is otherwise identical, so this isolates
    the effect of self-sampling (mode collapse) from everything else.
    """
    def __init__(self, *a, shadow_fn=None, exact_weights=False, **k):
        super().__init__(*a, **k)
        self.shadow_fn = shadow_fn or helper.random_Clifford_shadow
        self.exact_weights = exact_weights
        self.shadow_log = []          # (clifford, bitstring) pairs used, per iteration
        if exact_weights:
            self._all_states = utils.all_cb_states(self.num_qubits)

    def _step(self, i=0):
        with torch.no_grad():
            if self.exact_weights:
                unique_samples = self._all_states
                weights = self.nqs.full_state().state.norm2().type(torch.double)
                weights = weights / weights.sum()
            else:
                NN_samples = self.nqs.sample(self.num_samples)
                unique_samples, counts = torch.unique(NN_samples, dim=0, return_counts=True)
                weights = counts.type(torch.double) / self.num_samples
            prediction_list = []

        cliffords_batch, results_batch = self.shadow_fn(self.circuit, self.batch_size)
        self.shadow_log.append(flatten_shadows(cliffords_batch, results_batch))

        loss_nsqst = 0
        for cliff, res in zip(cliffords_batch, results_batch):
            with torch.no_grad():
                mat = cliff.adjoint().to_matrix()
                for bit, count in res.items():
                    Ub = mat[:, int(bit, 2)]
                    Clifford_sample_log_amplitudes, weights_new, unique_samples_new = \
                        helper.samples_to_log_clifford_amplitudes(Ub, unique_samples, weights)
                    NN_sample_log_amplitudes_no_grad = self.nqs.amplitudes(unique_samples_new, return_polar=True)
                    amp_fraction = (utils.Complex(Clifford_sample_log_amplitudes) - NN_sample_log_amplitudes_no_grad).exp()
                    amp_fraction_conj = amp_fraction.conjugate()
                    mean = (utils.Complex(weights_new) * amp_fraction).sum(dim=0)
                    overlap = utils.Complex.abs(mean)**2
                    prediction_list.append((2**self.num_qubits + 1) * overlap.detach().numpy() - 1)
            log_amplitudes = self.nqs.amplitudes(unique_samples_new, return_polar=True)
            term_1_log_moduli = amp_fraction_conj.real * log_amplitudes.real
            term_1_phases = amp_fraction_conj.imag * log_amplitudes.imag
            term_1_real = (weights_new * (term_1_log_moduli - term_1_phases)).sum(dim=0)
            term_1_imag = (weights_new * (amp_fraction_conj.real * log_amplitudes.imag +
                                          amp_fraction_conj.imag * log_amplitudes.real)).sum(dim=0)
            loss_real = term_1_real * mean.real - term_1_imag * mean.imag
            loss_nsqst = loss_nsqst - 2 * (2**self.num_qubits + 1) * loss_real / self.batch_size

        torch.nn.utils.clip_grad_norm_(self.nqs.parameters(), 5)
        loss_nsqst.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            cost_fun = 1 - helper.median_of_mean(prediction_list, self.K)
        return cost_fun

    def train(self, log_every=10):
        infidelity_list, cost_fun_list = [], []
        for i in range(self.max_iters):
            cost_fun = self._step(i=i)
            infidelity = float((1 - self.target_state.fidelity_to(self.nqs.full_state())).detach().numpy())
            infidelity_list.append(infidelity); cost_fun_list.append(float(cost_fun))
            if i % log_every == 0 or i == self.max_iters - 1:
                print(f'it {i:5d}  loss(shadow est) {cost_fun:+.4f}  exact infid {infidelity:.4f}', flush=True)
        return infidelity_list, cost_fun_list


def dense_state(nqs):
    with torch.no_grad():
        s = nqs.full_state().state
        return s.real.numpy().astype(complex) + 1j * s.imag.numpy()


# ----------------------------------------------------------------------------- pre-training
def sample_computational_basis(target_dense, n_meas, rng):
    N = int(np.log2(len(target_dense)))
    p = np.abs(target_dense)**2; p = p / p.sum()
    idx = rng.choice(2**N, size=n_meas, p=p)
    return torch.tensor([[int(c) for c in _bitstr(i, N)] for i in idx], dtype=torch.uint8)


def pretrain_amplitudes(nqs_amp, target_dense, rng, n_meas=3000, epochs=300, lr=1e-2, log_every=50):
    """Maximum-likelihood fit of |psi(s)|^2 to computational-basis samples of the target
    (standard NNQST amplitude training). Returns list of NLL per epoch."""
    samples = sample_computational_basis(target_dense, n_meas, rng)
    opt = torch.optim.Adam(nqs_amp.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        logamp = nqs_amp.amplitudes(samples, return_polar=True)   # log|psi| + i*0
        nll = -(2 * logamp.real).mean()                            # -mean log p(s)
        opt.zero_grad(); nll.backward(); opt.step()
        hist.append(float(nll))
        if ep % log_every == 0 or ep == epochs - 1:
            print(f'  pretrain ep {ep:4d}  NLL {float(nll):.4f}', flush=True)
    return hist


class DensePhi:
    """Minimal stand-in for Cirq's StabilizerStateChForm used by NSQST_Pre_Trainer_GHZ."""
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.complex128)
    def inner_product_of_state_and_x(self, index):
        return self.vec[index]


class PreTrainedTrainer(pre_trainer.ShadowTomographyTrainer):
    """Phase-only NSQST on a fixed shadow set with exact infidelity from the dense state."""
    def __init__(self, *a, target_dense=None, **k):
        super().__init__(*a, **k)
        self.target_dense = target_dense
        self._all_states = utils.all_cb_states(self.N)

    def dense_state(self):
        with torch.no_grad():
            a = self.nqs_amp.amplitudes(self._all_states, return_polar=False)
            ph = self.nqs_phase.amplitudes(self._all_states, return_polar=False)
            psi = a * ph
            return psi.real.numpy().astype(complex) + 1j * psi.imag.numpy()

    def train(self, log_every=10):
        infidelity_list, cost_fun_list = [], []
        for i in range(self.max_iters):
            cost_fun = self._step(i=i)
            psi = self.dense_state()
            infid = float(1 - abs(np.vdot(self.target_dense, psi))**2)
            infidelity_list.append(infid); cost_fun_list.append(float(cost_fun))
            if i % log_every == 0 or i == self.max_iters - 1:
                print(f'it {i:5d}  loss(shadow est) {cost_fun:+.4f}  exact infid {infid:.4f}', flush=True)
        return infidelity_list, cost_fun_list
