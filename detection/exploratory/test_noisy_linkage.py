"""Noisy-linkage arm tests.

    python3 -m pytest test_noisy_linkage.py

The scoring rules are pinned on hand-built partitions where the right count
is obvious by inspection; the operator is pinned by the eps=0 identity (the
arm must collapse to the oracle pipeline exactly) and by determinism.
"""

import os

import numpy as np
import pandas as pd
import pytest

import noisy_linkage as nl
import run_noisy_linkage as run
from dgp import default_config, generate
from endpoint import missed_per_10k
from features import TIER_COLS, build_entity_features, build_wallet_features


@pytest.fixture(scope="module")
def world():
    cfg = default_config(700501)
    cfg.n_entities = 400
    data = generate(cfg)
    wf = build_wallet_features(data)
    ef = build_entity_features(data, wf)
    return data, wf, ef


def _link(wallet_cluster, wallet_entity):
    wc = np.asarray(wallet_cluster)
    return nl.Linkage(wc, np.asarray(wallet_entity), int(wc.max()) + 1, {})


# ---------------------------------------------------------------------------
# eps = 0 reproduces the oracle exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_mode", nl.SPLIT_MODES)
@pytest.mark.parametrize("merge_mode", nl.MERGE_MODES)
@pytest.mark.parametrize("attr_rule", nl.ATTR_RULES)
def test_eps_zero_is_oracle(world, split_mode, merge_mode, attr_rule):
    data, wf, ef = world
    wx = nl.counterparty_incidence(data["transactions"], data["wallets"])
    link = nl.corrupt_linkage(data["wallets"], eps_split=0.0, eps_merge=0.0,
                              seed=1, split_mode=split_mode,
                              merge_mode=merge_mode, wallet_x=wx)
    ent_idx = data["wallets"].entity_id.str.slice(1).astype(int).to_numpy()
    assert link.n_clusters == len(data["entities"])
    np.testing.assert_array_equal(link.wallet_cluster, ent_idx)

    obs, ctab = nl.observed_entity_features(data, wf, link, attr_rule)
    assert (obs.entity_id.str.slice(1) == ef.entity_id.str.slice(1)).all()
    cols = TIER_COLS["T4"] + ["is_launderer"]
    pd.testing.assert_frame_equal(obs[cols].reset_index(drop=True),
                                  ef[cols].reset_index(drop=True),
                                  check_exact=True)


def test_eps_zero_scoring_matches_endpoint(world):
    data, _, ef = world
    link = nl.corrupt_linkage(data["wallets"], eps_split=0.0, eps_merge=0.0,
                              seed=3)
    y = data["entities"].is_launderer.to_numpy()
    rng = np.random.default_rng(0)
    s = rng.random(len(y))
    s[:20] = 0.5                                   # force some ties
    tb = np.arange(len(y))
    for k in (1, 10, 50, 200):
        ref = missed_per_10k(y, s, tb, k)
        assert nl.score_coverage(y, link, s, k)["MissedPer10k"] == ref
        assert nl.score_conservative(y, link, s, k)["MissedPer10k"] == ref


# ---------------------------------------------------------------------------
# operator: determinism, nesting, realized errors, labels never consulted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_mode", nl.SPLIT_MODES)
@pytest.mark.parametrize("merge_mode", nl.MERGE_MODES)
def test_deterministic_given_seed(world, split_mode, merge_mode):
    data, _, _ = world
    wx = nl.counterparty_incidence(data["transactions"], data["wallets"])
    kw = dict(eps_split=0.2, eps_merge=0.2, split_mode=split_mode,
              merge_mode=merge_mode, wallet_x=wx)
    a = nl.corrupt_linkage(data["wallets"], seed=11, **kw)
    b = nl.corrupt_linkage(data["wallets"], seed=11, **kw)
    c = nl.corrupt_linkage(data["wallets"], seed=12, **kw)
    np.testing.assert_array_equal(a.wallet_cluster, b.wallet_cluster)
    assert a.stats == b.stats
    assert not np.array_equal(a.wallet_cluster, c.wallet_cluster)


def test_errors_actually_happen_and_grow(world):
    data, _, _ = world
    prev_s = prev_m = -1.0
    for e in (0.05, 0.1, 0.3):
        s = nl.corrupt_linkage(data["wallets"], eps_split=e, eps_merge=0.0,
                               seed=5).stats
        m = nl.corrupt_linkage(data["wallets"], eps_split=0.0, eps_merge=e,
                               seed=5).stats
        assert s["frac_entities_in_merged_cluster"] == 0.0
        assert m["frac_entities_split"] == 0.0
        assert s["frac_entities_split"] > prev_s
        assert m["frac_entities_in_merged_cluster"] > prev_m
        prev_s = s["frac_entities_split"]
        prev_m = m["frac_entities_in_merged_cluster"]


def test_split_sets_nested_in_eps(world):
    """Common random numbers: every wallet detached at 0.1 is detached at 0.3."""
    data, _, _ = world
    ent = data["wallets"].entity_id.str.slice(1).astype(int).to_numpy()

    def detached(e):
        link = nl.corrupt_linkage(data["wallets"], eps_split=e, eps_merge=0.0,
                                  seed=9)
        # a wallet is detached iff its cluster differs from its entity's
        # lowest-index wallet's cluster
        first = pd.Series(link.wallet_cluster).groupby(ent).transform("first")
        return link.wallet_cluster != first.to_numpy()

    lo, hi = detached(0.1), detached(0.3)
    assert lo.sum() > 0 and hi.sum() > lo.sum()
    assert not (lo & ~hi).any()


def test_labels_never_consulted(world):
    data, _, _ = world
    flipped = {**data, "entities": data["entities"].assign(
        is_launderer=1 - data["entities"].is_launderer)}
    a = nl.corrupt_linkage(data["wallets"], eps_split=0.2, eps_merge=0.2,
                           seed=4)
    b = nl.corrupt_linkage(flipped["wallets"], eps_split=0.2, eps_merge=0.2,
                           seed=4)
    np.testing.assert_array_equal(a.wallet_cluster, b.wallet_cluster)


def test_bipartition_makes_exactly_two_fragments(world):
    data, _, _ = world
    link = nl.corrupt_linkage(data["wallets"], eps_split=1.0, eps_merge=0.0,
                              seed=2, split_mode="bipartition")
    per = pd.DataFrame({"e": link.wallet_entity, "c": link.wallet_cluster})
    nw = per.groupby("e").size()
    nc = per.groupby("e").c.nunique()
    assert (nc[nw > 1] == 2).all() and (nc[nw == 1] == 1).all()


# ---------------------------------------------------------------------------
# cluster table: labels and attribute rules on a hand-built merge
# ---------------------------------------------------------------------------

def _toy_entities():
    return pd.DataFrame({
        "entity_id": ["E0", "E1", "E2"],
        "is_launderer": [0, 1, 0],
        "kyc_tier": [2, 1, 0],
        "account_age_days": [900.0, 100.0, 400.0],
        "prior_sar_count": [0, 3, 1],
        "jurisdiction_risk": [0.1, 0.5, 0.9],
        "on_watchlist": [0, 0, 1],
    })


def test_cluster_label_any_member_and_attr_rules():
    ents = _toy_entities()
    # E0 has 3 wallets, E1 has 1, E2 has 1; cluster 0 = E0(3) + E1(1),
    # cluster 1 = E2
    link = _link([0, 0, 0, 0, 1], [0, 0, 0, 1, 2])
    mx = nl.cluster_table(ents, link, "max_risk")
    assert mx.is_launderer.tolist() == [1, 0]           # any member illicit
    assert mx.loc[0, "kyc_tier"] == 1
    assert mx.loc[0, "account_age_days"] == 100.0
    assert mx.loc[0, "prior_sar_count"] == 3
    assert mx.loc[0, "jurisdiction_risk"] == 0.5
    assert mx.loc[0, "on_watchlist"] == 0
    maj = nl.cluster_table(ents, link, "majority")
    assert maj.loc[0, "kyc_tier"] == 2 and maj.loc[0, "prior_sar_count"] == 0
    assert maj.is_launderer.tolist() == [1, 0]
    fw = nl.cluster_table(ents, link, "first_wallet")
    assert fw.loc[0, "account_age_days"] == 900.0
    for t in (mx, maj, fw):                              # singleton = itself
        assert t.loc[1, "jurisdiction_risk"] == 0.9
        assert t.loc[1, "on_watchlist"] == 1


# ---------------------------------------------------------------------------
# scoring rules on hand-built partitions
# ---------------------------------------------------------------------------

def test_merged_cluster_with_two_illicit_entities():
    # entities: E0 illicit (2 wallets), E1 illicit (1 wallet), E2 legit,
    # E3 legit. Cluster 0 = E0 + E1 (false merge), cluster 1 = E2, cluster 2
    # = E3. n = 4, N_pos = 2.
    y = np.array([1, 1, 0, 0])
    link = _link([0, 0, 0, 1, 2], [0, 0, 1, 2, 3])
    s = np.array([0.9, 0.1, 0.2])
    cov = nl.score_coverage(y, link, s, k=1)
    con = nl.score_conservative(y, link, s, k=1)
    assert cov["TP"] == 2                  # one alert covers both entities
    assert con["TP"] == 1                  # at most one per cluster
    assert cov["MissedPer10k"] == 0.0
    assert con["MissedPer10k"] == 10_000.0 * 1 / 4


def test_split_fragments_reviewed_twice_count_once():
    # E0 illicit, 3 wallets split into two fragments (clusters 0 and 1);
    # E1 illicit, cluster 2; E2 legit, cluster 3.
    y = np.array([1, 1, 0])
    link = _link([0, 0, 1, 2, 3], [0, 0, 0, 1, 2])
    s = np.array([0.9, 0.8, 0.1, 0.2])     # both fragments of E0 on top
    for fn in (nl.score_coverage, nl.score_conservative):
        r = fn(y, link, s, 2)
        assert r["TP"] == 1                # second fragment wasted budget
        r3 = fn(y, link, s, 4)
        assert r3["TP"] == 2


def test_conservative_credits_new_entity_after_dedup():
    # cluster 0 = E0(2 wallets) + E1(1 wallet), both illicit; cluster 1 =
    # fragment of E0 only. Reviewing cluster 0 credits E0 (dominant member);
    # cluster 1 then adds nothing. Reversed order: fragment credits E0 first,
    # merged cluster then credits E1.
    y = np.array([1, 1])
    link = _link([0, 0, 0, 1], [0, 0, 1, 0])
    assert nl.score_conservative(y, link, np.array([0.9, 0.8]), 2)["TP"] == 1
    assert nl.score_conservative(y, link, np.array([0.8, 0.9]), 2)["TP"] == 2
    assert nl.score_coverage(y, link, np.array([0.9, 0.8]), 2)["TP"] == 2


def test_coverage_min_frac():
    # E0 illicit with 4 wallets: cluster 0 holds 1, cluster 1 holds 3
    y = np.array([1, 0])
    link = _link([0, 1, 1, 1, 2], [0, 0, 0, 0, 1])
    s = np.array([0.9, 0.1, 0.5])
    assert nl.score_coverage(y, link, s, 1)["TP"] == 1          # any
    assert nl.score_coverage(y, link, s, 1, min_frac=0.5)["TP"] == 0
    assert nl.score_coverage(y, link, s, 3, min_frac=0.5)["TP"] == 1


def test_conservative_never_exceeds_coverage(world):
    data, _, _ = world
    y = data["entities"].is_launderer.to_numpy()
    rng = np.random.default_rng(1)
    for seed in range(5):
        link = nl.corrupt_linkage(data["wallets"], eps_split=0.3,
                                  eps_merge=0.3, seed=seed)
        s = rng.random(link.n_clusters)
        for k in (5, 20, 80):
            assert (nl.score_conservative(y, link, s, k)["TP"]
                    <= nl.score_coverage(y, link, s, k)["TP"])


def test_budget_capped_at_cluster_count():
    y = np.array([1, 1, 0])
    link = _link([0, 0, 0], [0, 1, 2])     # everything merged into one
    r = nl.score_coverage(y, link, np.array([0.5]), 5)
    assert r["k_eff"] == 1 and r["TP"] == 2


# ---------------------------------------------------------------------------
# seed guard
# ---------------------------------------------------------------------------

def test_seed_guard_accepts_dev_range():
    pairs = run.check_seeds(700501, 3)
    assert pairs[0] == (700501, 700501 + run.SEED_OFFSET)


@pytest.mark.parametrize("base,R", [
    (700000, 3),          # below DEV
    (700998, 5),          # runs off the top of DEV
    (900001, 1),          # burned pilot
    (2026080501, 1),      # confirmatory
    (20260707, 1),
])
def test_seed_guard_refuses_without_lock(base, R):
    with pytest.raises(run.SeedGuardError):
        run.check_seeds(base, R)


def test_seed_guard_lock(tmp_path):
    lock = tmp_path / "addendum.txt"
    lock.write_text("seed_base: 810001\n")
    assert run.check_seeds(810001, 52, str(lock))[0][0] == 810001
    with pytest.raises(run.SeedGuardError):          # lock names another base
        run.check_seeds(810002, 52, str(lock))
    with pytest.raises(run.SeedGuardError):          # missing file
        run.check_seeds(810001, 52, str(tmp_path / "nope"))
    lock.write_text("seed_base: 2026080501\n")
    with pytest.raises(run.SeedGuardError):          # spent even under a lock
        run.check_seeds(2026080501, 1, str(lock))
    lock.write_text("seed_base: 4294967290\n")
    with pytest.raises(run.SeedGuardError):          # sklearn ceiling
        run.check_seeds(4294967290, 10, str(lock))


def test_seed_guard_refuses_spent_test_stream(tmp_path):
    # a clean train seed whose TEST stream lands on the confirmatory base
    base = 2026080501 - run.SEED_OFFSET
    lock = tmp_path / "a.txt"
    lock.write_text(str(base))
    with pytest.raises(run.SeedGuardError, match="spent"):
        run.check_seeds(base, 1, str(lock))


def test_driver_refuses_non_dev_seed(tmp_path):
    with pytest.raises(SystemExit) as e:
        run.main(["--seed-base", "123", "--R", "1", "--out-dir",
                  str(tmp_path / "o")])
    assert e.value.code == 2
    assert not os.path.exists(tmp_path / "o")


def test_grid():
    g = run.eps_grid((0, 0.1, 0.2), "axes")
    assert g[0] == (0.0, 0.0) and len(g) == 7 and (0.2, 0.2) in g
    assert len(run.eps_grid((0, 0.1, 0.2), "full")) == 9


# ---------------------------------------------------------------------------
# integration: the oracle cell IS the confirmatory replicate
# ---------------------------------------------------------------------------

def test_oracle_cell_matches_confirmatory_replicate():
    from replicate import run_replicate
    tr, te = 700502, 700502 + run.SEED_OFFSET
    kw = dict(n_train=600, n_test=600, k_star=500)
    ref = run_replicate(0, tr, te, **kw)
    mine = run.run_linkage_replicate(0, tr, te, cells=[(0.0, 0.0)], **kw)
    assert ref["status"] == mine["status"] == "OK"
    cell = mine["cells"][0]["models"]
    for m in run.MODELS:
        for t in run.TIERS:
            for rule in run.RULES:
                assert (cell[m][t][rule]["MissedPer10k"]
                        == ref["models"][m][t]["MissedPer10k"])
