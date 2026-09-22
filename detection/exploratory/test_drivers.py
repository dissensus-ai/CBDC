"""Driver-level tests: run spec contents and summary layout, end to end at toy
size on DEV seeds (700030-700031 for E9, 700040-700041 for E10)."""

import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_signal_scale  # noqa: E402
import run_two_stage  # noqa: E402
from addendum_lock import SPEC_KEYS  # noqa: E402
from seed_guard import TEST_SEED_OFFSET  # noqa: E402


def _spec(mod, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main(["--seed-base", "700030", "--R", "3", "--print-spec", *extra])
    return json.loads(buf.getvalue())


def test_print_spec_defaults_match_confirmatory_sizes():
    for mod, arm, stream in ((run_two_stage, "E9", False),
                             (run_signal_scale, "E10", True)):
        spec = _spec(mod)
        assert tuple(spec) == SPEC_KEYS
        assert spec["arm"] == arm
        assert (spec["n_train"], spec["n_test"], spec["k_star"]) == \
            (8000, 10000, 500)                      # confirmatory D2/D1/D3
        assert spec["test_seed_offset"] == TEST_SEED_OFFSET
        assert spec["identity_rng_stream"] is stream
        assert set(spec["environment"]) == {"python", "numpy", "scipy",
                                            "sklearn", "pandas",
                                            "requirements_sha256"}
    e9 = _spec(run_two_stage)
    assert e9["flags"]["stage1_train_scores"] == "oof"
    assert e9["models"] == ["gboost", "logit"]
    e10 = _spec(run_signal_scale, "--models", "gboost", "logit")
    assert e10["models"] == ["gboost", "logit"]
    assert e10["grid"] == [0.25 * i for i in range(9)]


def test_e9_summary_leads_with_absolute_table():
    with tempfile.TemporaryDirectory() as d:
        run_two_stage.main(["--seed-base", "700030", "--R", "2",
                            "--n-train", "800", "--n-test", "1000",
                            "--kprime-grid", "500", "1000", "10000",
                            "--bootstrap-B", "50", "--workers", "2",
                            "--out-dir", d])
        with open(os.path.join(d, "two_stage_summary.json")) as f:
            s = json.load(f)
    keys = list(s)
    assert keys[0] == "primary_absolute_missed_vs_identity_lookups"
    cells = s[keys[0]]["cells"]
    prim = [c for c in cells if c["is_primary_cell"]]
    assert len(prim) == 1 and prim[0]["model"] == "gboost"
    assert [p["IdentityLookupsPer10k"] for p in prim[0]["curve"]] == \
        [500.0, 1000.0, 10000.0]
    assert "recovery_secondary" in s and s["run_spec"]["n_train"] == 800
    for rec in s["recovery_secondary"].values():
        assert rec["role"] == "secondary_annotation"
        assert "reference_gain" in rec


def test_e10_summary_leads_with_both_crossing_tables():
    with tempfile.TemporaryDirectory() as d:
        run_signal_scale.main(["--seed-base", "700040", "--R", "2",
                               "--n-train", "600", "--n-test", "800",
                               "--lambda-grid", "0", "1", "2",
                               "--bootstrap-B", "50", "--workers", "2",
                               "--out-dir", d])
        with open(os.path.join(d, "signal_scale_summary.json")) as f:
            s = json.load(f)
    keys = list(s)
    assert keys[:3] == ["gain_threshold_crossing_T2_minus_T4",
                        "gain_threshold_crossing_T2_minus_T3",
                        "per_lambda_delta"]
    for c in keys[:2]:
        tab = s[c]
        assert tab["primary"] == "crossing_mean"
        assert set(tab["by_model"]) == {"gboost", "logit", "additive"}
        for entries in tab["by_model"].values():
            assert [e["tau"] for e in entries] == [0.0, 1.0]
            for e in entries:
                if e["flag_mean"] != "not_reached":
                    p = e["params_at_crossing_mean"]
                    assert p["lambda"] == e["crossing_mean"]
                    assert "watchlist_tpr" in p and "segment" in p
                else:
                    assert e["params_at_crossing_mean"] is None
                for side in ("lo", "hi"):
                    lim = e["bootstrap"]["mean"][side]
                    assert (lim["params"] is None) == (lim["value"] is None)
    assert s["primary_estimand"] == "crossing_mean"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} driver tests passed")
