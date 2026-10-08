"""Load shadows collected by collect_shadows.py into the pinned env's trainer format.
Returns (cliffords, results) exactly like Helper_fun.random_Clifford_shadow, so you can do
    helper.random_Clifford_shadow = lambda circuit, k: load_hw_shadows.as_shadow_fn('file.json')(circuit, k)
"""
import json
from qiskit.quantum_info import Clifford


def load(path):
    d = json.load(open(path))
    cls = [Clifford.from_dict(s['clifford']) for s in d['shadows']]
    res = [{s['bit']: 1} for s in d['shadows']]
    return cls, res, d


def as_shadow_fn(path):
    cls, res, _ = load(path)
    def fn(circuit, k):
        assert k <= len(cls), f'file has only {len(cls)} shadows'
        return cls[:k], res[:k]
    return fn
