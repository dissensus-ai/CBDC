"""M2 (additive) hook: same record fields and conventions as the fitted pair."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

from extra_models import ADDITIVE_MODEL, additive_model, make_additive_hook  # noqa: E402
from identity_signal import cfg_factory  # noqa: E402
from replicate import run_replicate  # noqa: E402


def test_additive_is_no_interaction_boosting():
    m = additive_model(7)
    assert m.interaction_cst == "no_interactions"
    assert m.class_weight == "balanced" and m.random_state == 7


def test_hook_writes_record_fields_and_leaves_pair_alone():
    kw = dict(n_train=900, n_test=1000, k_star=500)
    seeds = (0, 700_020, 700_020 + 10_000_003)
    base = run_replicate(*seeds, cfg_factory=cfg_factory(1.0), **kw)
    rec = run_replicate(*seeds, cfg_factory=cfg_factory(1.0),
                        score_hook=make_additive_hook(), **kw)
    assert rec["status"] == "OK", rec.get("error")
    add = rec["models"].pop(ADDITIVE_MODEL)
    d4 = rec.pop(f"delta_miss_T2_minus_T4_{ADDITIVE_MODEL}")
    d3 = rec.pop(f"delta_miss_T2_minus_T3_{ADDITIVE_MODEL}")
    assert rec == base                      # confirmatory fields untouched
    n_pos = rec["N_positive_test"]
    for tier in ("T2", "T3", "T4"):
        assert add[tier]["MissedPer10k"] == 10.0 * (n_pos - add[tier]["TP"])
    assert d4 == add["T2"]["MissedPer10k"] - add["T4"]["MissedPer10k"]
    assert d3 == add["T2"]["MissedPer10k"] - add["T3"]["MissedPer10k"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} extra-model tests passed")
