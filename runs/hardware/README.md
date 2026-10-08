# Running NSQST data collection on a real quantum computer

NSQST only needs one thing from the quantum computer: a list of pairs
(random Clifford circuit that was applied, bitstring that was measured),
plus a few thousand plain computational-basis shots if we pre-train the amplitudes.
Everything else (the neural network, the gradient, the fidelity estimate) is classical.
So the plan is: collect that list on the device, save it as JSON, and feed it to the trainer.

The original repo is pinned to Qiskit 0.42 (2023). That version cannot talk to IBM's
current cloud, so data collection lives in a second conda environment:

| env | Python | packages | used for |
|---|---|---|---|
| `nsqst` | 3.10 | torch 2.0, qiskit 0.42.1, cirq 1.2 | training (original code + `runs/`) |
| `qhw` | 3.12 | qiskit 2.x, qiskit-ibm-runtime, qiskit-aer, qiskit-ionq | talking to devices (`runs/hardware/`) |

## 1. Create an IBM Quantum account (you have to do this yourself)

1. Go to https://quantum.cloud.ibm.com and sign up (an IBMid, free). The old
   quantum-computing.ibm.com site was retired in 2025; everything is on IBM Cloud now.
2. The free **Open Plan** gives up to 10 minutes of QPU time per 28-day rolling window
   (IBM's docs also mention an opt-in promotion of 180 extra minutes over 12 months for
   active Open Plan users, as of March 2026). That is plenty: the whole
   "pre-training + 200 shadows" protocol at 6 qubits is estimated at about 1 s of QPU
   time (see `runs/quantum_budget.py`).
3. In the dashboard create an **instance** on the Open Plan and copy two things:
   the **API key** and the instance **CRN** (looks like `crn:v1:bluemix:public:quantum-computing:us-east:...`).
4. Save them once on this laptop, from the `qhw` environment (never paste the key into a
   file that goes to GitHub):

```bash
~/opt/anaconda3/envs/qhw/bin/python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(token='PASTE_API_KEY', instance='PASTE_CRN', set_as_default=True, overwrite=True)"
```

   This is the form in IBM's "Save your access credentials" guide (the channel defaults to
   `ibm_quantum_platform`; `instance` can be left out and IBM picks one). It writes
   `~/.qiskit/qiskit-ibm.json`; don't edit that file by hand. Check it works:

```bash
~/opt/anaconda3/envs/qhw/bin/python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; s=S(); print([b.name for b in s.backends()]); print(s.least_busy(simulator=False, operational=True).name)"
```

## 2. Connect VS Code

1. Install the **Python** and **Jupyter** extensions from Microsoft.
2. Open this folder in VS Code (`File > Open Folder > Neural-Shadow-QST-master`).
3. `Cmd+Shift+P` > **Python: Select Interpreter** > pick
   `~/opt/anaconda3/envs/qhw/bin/python` for hardware scripts
   (or `.../envs/nsqst/bin/python` for training scripts). The two envs are already
   registered as Jupyter kernels, so notebooks can pick "qhw" or "nsqst" from the kernel menu.
4. Optional: IBM's **Qiskit Code Assistant** VS Code extension (search "Qiskit" in the
   extensions panel) gives Qiskit autocompletion and uses the same saved account.
5. Running `collect_shadows.py` from the VS Code terminal with the `qhw` interpreter is
   the "connection": `QiskitRuntimeService()` reads the saved account, submits the job,
   and waits for the result. Jobs can also be watched at https://quantum.cloud.ibm.com/workloads.

## 3. Collect shadows

Simulator with a real device's noise model (no account needed, good for testing):

```bash
cd runs/hardware && ~/opt/anaconda3/envs/qhw/bin/python collect_shadows.py --backend fake:FakeTorino --n 6 --num 200 --cb_shots 3000 --out shadows_fake_torino_n6.json
```

Real IBM machine (after step 1):

```bash
cd runs/hardware && ~/opt/anaconda3/envs/qhw/bin/python collect_shadows.py --backend ibm:least_busy --n 6 --num 200 --cb_shots 3000 --out shadows_ibm_n6.json
```

The script submits everything as one SamplerV2 job (200 one-shot circuits plus one
3000-shot circuit), waits, and records `job.usage()` (the QPU seconds IBM charged) in the JSON.
IonQ: `--backend ionq:simulator` needs the `IONQ_API_KEY` environment variable; IonQ
hardware is paid (through IonQ, AWS Braket or Azure), see the budget script.

## 4. Train on the collected data (back in the `nsqst` env)

```bash
~/opt/anaconda3/envs/nsqst/bin/python -u runs/nsqst_diag.py --mode pretrained --shadow_file runs/hardware/shadows_fake_torino_n6.json --iters 200
```

Pre-training uses the measured computational-basis counts from the file, and the 200
shadows are reused every iteration (same as the 40-qubit demo).

## Bit ordering

Qiskit's `measure_all` strings and the old trainer both put qubit 0 on the right, so the
bitstrings can be used directly. `load_hw_shadows.py` rebuilds the Clifford objects with
`Clifford.from_dict`. Checked: with the ideal Aer backend, none of the sampled bitstrings
has zero probability under the exact state.
