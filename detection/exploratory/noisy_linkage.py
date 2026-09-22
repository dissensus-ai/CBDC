"""Noisy wallet->entity linkage: corruption operator, cluster tables, scoring.

EXPLORATORY. Nothing produced with this module is a reported result until a
protocol addendum freezes the choices listed in DESIGN_NOISY_LINKAGE.md.

The confirmatory pipeline treats linkage as an oracle: T2 aggregates over the
TRUE wallet set of each entity. This module replaces the oracle with an
observed partition produced by corrupting the true one, then scores the
observed clusters while keeping every label and every miss count on TRUE
entities (protocol v3 section 7, erratum E5).

Pipeline for one world:

    corrupt_linkage(wallets, ...)      -> observed wallet->cluster map
    cluster_table(...)                 -> one row per cluster: pooled identity
                                          attributes + any-member label
    observed_entity_features(...)      -> T2/T3/T4 features on clusters, via
                                          the unchanged features.py code
    score_coverage / score_conservative -> TP and MissedPer10k per TRUE entity

At eps_s = eps_m = 0 every step reproduces the oracle exactly (tested): the
observed partition is the true one, clusters are ordered as entities, pooled
attributes of a one-entity cluster are that entity's attributes, and both
scoring rules collapse to endpoint.missed_per_10k.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

from features import build_entity_features  # noqa: E402
from endpoint import select_alerts  # noqa: E402

SPLIT_MODES = ("wallet", "bipartition")
MERGE_MODES = ("uniform", "counterparty")
ATTR_RULES = ("max_risk", "majority", "first_wallet")

# Stream tag so the corruption RNG never coincides with the DGP's own stream
# for the same world seed ("LINK" in ASCII).
LINKAGE_STREAM_TAG = 0x4C494E4B

ATTR_COLS = ("kyc_tier", "account_age_days", "prior_sar_count",
             "jurisdiction_risk", "on_watchlist")


def _ids_to_int(ids) -> np.ndarray:
    """'W17' / 'E3' / 'C9' -> 17 / 3 / 9. Numeric keys, never string order."""
    return pd.Series(ids).str.slice(1).astype(np.int64).to_numpy()


@dataclass
class Linkage:
    """Observed partition of one world's wallets.

    wallet_cluster[i]  observed cluster index of wallet i (wallet table order)
    wallet_entity[i]   TRUE entity index of wallet i
    n_clusters         clusters are 0..n_clusters-1, ordered by their lowest
                       wallet index, so at eps=0 cluster j == entity j
    stats              realized error rates (what eps actually did)
    """
    wallet_cluster: np.ndarray
    wallet_entity: np.ndarray
    n_clusters: int
    stats: dict


def _canonical(labels: np.ndarray, wallet_idx: np.ndarray) -> np.ndarray:
    """Relabel an arbitrary partition 0..C-1 by each block's lowest wallet
    index. Makes cluster ids independent of how the partition was built."""
    order = np.argsort(wallet_idx, kind="stable")
    first_seen = {}
    for lab in labels[order]:
        if lab not in first_seen:
            first_seen[lab] = len(first_seen)
    return np.fromiter((first_seen[x] for x in labels), dtype=np.int64,
                       count=len(labels))


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def counterparty_incidence(transactions, wallets) -> sparse.csr_matrix:
    """Wallet x external-counterparty 0/1 matrix (rows in wallet-table order).

    The DGP has no direct transfers between entities -- every entity's flows
    go to or from the shared external pool -- so a shared external
    counterparty is the only graph proximity two entities can have. This is
    the input to the homophilous merge variant.
    """
    w_index = {w: i for i, w in enumerate(wallets.wallet_id)}
    tx = transactions
    pairs = []
    for own_col, other_col in (("src", "dst"), ("dst", "src")):
        other = tx[other_col]
        m = other.str.startswith("X") & tx[own_col].isin(w_index)
        pairs.append(pd.DataFrame({
            "w": tx.loc[m, own_col].map(w_index).to_numpy(),
            "x": other[m].str.slice(1).astype(np.int64).to_numpy(),
        }))
    p = pd.concat(pairs).drop_duplicates()
    n_x = int(p.x.max()) + 1 if len(p) else 1
    return sparse.csr_matrix(
        (np.ones(len(p), dtype=np.float32), (p.w.to_numpy(), p.x.to_numpy())),
        shape=(len(wallets), n_x))


def corrupt_linkage(wallets, *, eps_split, eps_merge, seed,
                    split_mode="wallet", merge_mode="uniform",
                    wallet_x=None, chunk=512) -> Linkage:
    """Corrupt the true wallet->entity partition. Labels are never consulted.

    Order: splits first, on the true partition; merges second, on the
    post-split partition.

    Split, mode "wallet" (default): every wallet independently, with
    probability eps_split, is detached from its entity into its own singleton
    cluster. A one-wallet entity is unaffected (it is already a singleton).
    Mode "bipartition" (protocol v2 section 6): every multi-wallet entity
    independently, with probability eps_split, is cut into two non-empty
    fragments at a uniformly random cut of a random wallet permutation.

    Merge: every post-split cluster independently, with probability
    eps_merge, becomes an initiator and is merged with one partner cluster
    drawn from the post-split partition excluding itself. Mode "uniform"
    (default): partner uniform. Mode "counterparty": partner drawn with
    probability proportional to the number of external counterparties it
    shares with the initiator; an initiator sharing none falls back to
    uniform. Merges compose transitively (union-find), so an observed
    cluster can contain three or more true entities, and a merge can rejoin
    two fragments of the same entity.

    Common random numbers: for a fixed world seed every uniform is drawn for
    every unit whether or not it is used, from two independent child streams.
    Split sets are therefore nested in eps_split, and for a fixed eps_split
    the initiator set is nested in eps_merge. Cells of an eps grid differ
    only in how many units cross the threshold, which makes paired cross-cell
    contrasts far less noisy than independent draws would.
    """
    if not (0.0 <= eps_split <= 1.0 and 0.0 <= eps_merge <= 1.0):
        raise ValueError("error rates must lie in [0, 1]")
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}")
    if merge_mode not in MERGE_MODES:
        raise ValueError(f"merge_mode must be one of {MERGE_MODES}")

    ss = np.random.SeedSequence([int(seed), LINKAGE_STREAM_TAG])
    rng_split, rng_merge = (np.random.default_rng(s) for s in ss.spawn(2))

    w_idx = _ids_to_int(wallets.wallet_id)
    e_idx = _ids_to_int(wallets.entity_id)
    n_w = len(wallets)

    # ---- splits: a fragment is labelled (entity, fragment-number) ----------
    frag = np.zeros(n_w, dtype=np.int64)
    if split_mode == "wallet":
        u = rng_split.random(n_w)
        detached = u < eps_split
        # a detached wallet gets a fragment number unique within its entity
        frag[detached] = 1 + np.arange(int(detached.sum()))
    else:
        order = np.argsort(e_idx, kind="stable")
        ents, start = np.unique(e_idx[order], return_index=True)
        counts = np.diff(np.append(start, n_w))
        u = rng_split.random(len(ents))
        for j, (s0, c) in enumerate(zip(start, counts)):
            perm = rng_split.permutation(c)       # drawn for every entity
            cut = 1 + int(rng_split.integers(max(1, c - 1)))
            if c > 1 and u[j] < eps_split:
                members = order[s0:s0 + c]
                frag[members[perm[cut:]]] = 1
    pre = _canonical(e_idx * (n_w + 1) + frag, w_idx)
    n_pre = int(pre.max()) + 1

    # ---- merges on the post-split partition -------------------------------
    u_c = rng_merge.random(n_pre)
    v_c = rng_merge.random(n_pre)
    initiators = np.flatnonzero(u_c < eps_merge) if n_pre > 1 else []
    parent = list(range(n_pre))
    partners = {}
    if len(initiators):
        if merge_mode == "uniform":
            for c in initiators:
                p = int(v_c[c] * (n_pre - 1))
                partners[c] = p + (p >= c)
        else:
            if wallet_x is None:
                raise ValueError("counterparty merges need wallet_x")
            w2c = sparse.csr_matrix(
                (np.ones(n_w, dtype=np.float32), (pre, np.arange(n_w))),
                shape=(n_pre, n_w))
            cx = (w2c @ wallet_x).tocsr()
            cx.data[:] = 1.0
            for b0 in range(0, len(initiators), chunk):
                block = initiators[b0:b0 + chunk]
                shared = (cx[block] @ cx.T).toarray()
                for r, c in enumerate(block):
                    w = shared[r]
                    w[c] = 0.0
                    tot = w.sum()
                    if tot <= 0:
                        p = int(v_c[c] * (n_pre - 1))
                        partners[c] = p + (p >= c)
                    else:
                        cdf = np.cumsum(w) / tot
                        partners[c] = int(min(np.searchsorted(
                            cdf, v_c[c], side="right"), n_pre - 1))
        for c in initiators:
            a, b = _find(parent, c), _find(parent, partners[c])
            if a != b:
                parent[max(a, b)] = min(a, b)
    root = np.array([_find(parent, c) for c in range(n_pre)], dtype=np.int64)
    obs = _canonical(root[pre], w_idx)
    n_obs = int(obs.max()) + 1

    # ---- realized error rates ---------------------------------------------
    pair = pd.DataFrame({"e": e_idx, "c": obs})
    ents_per_cluster = pair.drop_duplicates().groupby("c").e.nunique()
    clusters_per_ent = pair.drop_duplicates().groupby("e").c.nunique()
    merged_c = set(ents_per_cluster.index[ents_per_cluster > 1])
    ent_in_merged = pair[pair.c.isin(merged_c)].e.unique()
    n_e = len(clusters_per_ent)
    stats = {
        "n_entities": int(n_e),
        "n_wallets": int(n_w),
        "n_clusters_postsplit": n_pre,
        "n_clusters": n_obs,
        "n_merge_initiators": int(len(initiators)),
        "frac_entities_split": float((clusters_per_ent > 1).mean()),
        "frac_entities_in_merged_cluster": float(len(ent_in_merged) / n_e),
        "max_entities_per_cluster": int(ents_per_cluster.max()),
        "mean_clusters_per_entity": float(clusters_per_ent.mean()),
    }
    return Linkage(obs, e_idx, n_obs, stats)


def cluster_table(entities, link: Linkage, attr_rule="max_risk") -> pd.DataFrame:
    """One row per observed cluster, in the entity-table schema.

    Label: a cluster is illicit iff ANY member wallet belongs to an illicit
    true entity (erratum E5 freeze: any-member). Training on this label is
    what a supervisor learning from past case outcomes on resolved clusters
    would see: a cluster that led to a laundering case is a positive.

    Identity attributes live on true entities. The rule for a cluster that
    spans several (semi-oracle attachment -- no identity is "discovered"):

      max_risk     (default; protocol v3 section 7.1 freeze candidate)
                   riskiest member value per attribute: minimum kyc_tier,
                   minimum account_age_days, maximum prior_sar_count,
                   maximum jurisdiction_risk, on_watchlist = any member.
      majority     attributes of the true entity contributing the most
                   wallets to the cluster (ties: lowest entity index).
      first_wallet attributes of the entity owning the cluster's lowest-index
                   wallet -- the earliest-issued credential, i.e. the record
                   a resolver keyed on its first wallet would inherit.

    All three return the entity's own attributes for a one-entity cluster.
    """
    if attr_rule not in ATTR_RULES:
        raise ValueError(f"attr_rule must be one of {ATTR_RULES}")
    ent = entities.reset_index(drop=True)
    e_row = pd.Series(np.arange(len(ent)),
                      index=_ids_to_int(ent.entity_id))
    rows = e_row.loc[link.wallet_entity].to_numpy()
    m = pd.DataFrame({"c": link.wallet_cluster, "row": rows,
                      "order": np.arange(len(rows))})
    attrs = ent.loc[m.row, list(ATTR_COLS) + ["is_launderer"]].reset_index(
        drop=True)
    m = pd.concat([m, attrs], axis=1)
    g = m.groupby("c", sort=True)

    out = pd.DataFrame(index=pd.RangeIndex(link.n_clusters, name="c"))
    out["is_launderer"] = g.is_launderer.max().astype(int)
    if attr_rule == "max_risk":
        out["kyc_tier"] = g.kyc_tier.min()
        out["account_age_days"] = g.account_age_days.min()
        out["prior_sar_count"] = g.prior_sar_count.max()
        out["jurisdiction_risk"] = g.jurisdiction_risk.max()
        out["on_watchlist"] = g.on_watchlist.max()
    else:
        if attr_rule == "majority":
            cnt = m.groupby(["c", "row"]).size().rename("n").reset_index()
            cnt = cnt.sort_values(["c", "n", "row"], ascending=[True, False,
                                                                True])
            pick = cnt.drop_duplicates("c").set_index("c").row
        else:
            pick = m.sort_values("order").drop_duplicates("c").set_index(
                "c").row
        src = ent.loc[pick.sort_index().to_numpy(), list(ATTR_COLS)]
        for col in ATTR_COLS:
            out[col] = src[col].to_numpy()
    out = out.reset_index(drop=True)
    out.insert(0, "entity_id", [f"C{j}" for j in range(link.n_clusters)])
    return out[["entity_id", "is_launderer", *ATTR_COLS]]


def observed_entity_features(data, wallet_feats, link: Linkage,
                             attr_rule="max_risk"):
    """T2-T4 features on observed clusters, via unchanged features.py code.

    features.build_entity_features only ever sees `entity_id` columns; handing
    it the observed partition under that name makes every T2 aggregate a
    cluster aggregate. Per-wallet (T1) features do not depend on linkage and
    are reused from the oracle build with the cluster id substituted.

    Returns (cluster feature frame, cluster table).
    """
    ctab = cluster_table(data["entities"], link, attr_rule)
    cid = np.array([f"C{j}" for j in link.wallet_cluster])
    obs_wallets = pd.DataFrame({"wallet_id": data["wallets"].wallet_id.values,
                                "entity_id": cid})
    obs = {"entities": ctab, "wallets": obs_wallets,
           "transactions": data["transactions"], "config": data["config"]}
    wf = wallet_feats.drop(columns=["entity_id"]).merge(obs_wallets,
                                                        on="wallet_id")
    return build_entity_features(obs, wf), ctab


# ---------------------------------------------------------------------------
# scoring on TRUE entities
# ---------------------------------------------------------------------------

def _check_index(y, link):
    # y_entity[i] must be the label of true entity E<i> (the DGP's order)
    if link.wallet_entity.max() >= len(y):
        raise ValueError("y_entity must be indexed by true entity number")


def _selected(scores, k):
    """Top-k cluster indices, ties to the lower cluster index. Budget is
    capped at the cluster count (only binds under extreme merging); the
    caller records the effective budget."""
    k_eff = min(k, len(scores))
    return select_alerts(scores, np.arange(len(scores)), k_eff), k_eff


def score_coverage(y_entity, link: Linkage, scores, k, min_frac=0.0):
    """Entity-coverage rule (erratum E5 'optimistic'; protocol v3 7.3).

    Review the k highest-scoring clusters. A true entity is detected if the
    reviewed clusters contain at least one of its wallets and at least
    `min_frac` of them (default 0.0: any wallet suffices). One cluster alert
    can therefore detect several true entities (a merge), and an entity split
    across reviewed fragments is counted once.

    Returns dict(TP, MissedPer10k, k_eff). MissedPer10k is per TRUE entity.
    """
    y = np.asarray(y_entity)
    n = len(y)
    _check_index(y, link)
    sel, k_eff = _selected(scores, k)
    reviewed = np.isin(link.wallet_cluster, sel)
    tot = np.bincount(link.wallet_entity, minlength=n)
    hit = np.bincount(link.wallet_entity, weights=reviewed, minlength=n)
    covered = (hit >= 1) & (hit >= min_frac * tot)
    tp = int(y[covered].sum())
    return {"TP": tp, "MissedPer10k": 10_000.0 * (int(y.sum()) - tp) / n,
            "k_eff": int(k_eff)}


def score_conservative(y_entity, link: Linkage, scores, k):
    """Conservative cluster-level rule (erratum E5, with split dedup).

    Walk the k reviewed clusters in rank order. Each credits AT MOST ONE true
    illicit entity: the not-yet-credited illicit member contributing the most
    wallets to that cluster (ties: lower entity index). A cluster whose
    illicit members are all already credited -- e.g. the second reviewed
    fragment of a split entity -- credits nothing but still used budget.

    Greedy in review order, so TP_conservative <= the maximum bipartite
    matching of reviewed clusters to illicit entities, and <= TP_coverage.

    Returns dict(TP, MissedPer10k, k_eff).
    """
    y = np.asarray(y_entity)
    n = len(y)
    _check_index(y, link)
    sel, k_eff = _selected(scores, k)
    illicit_w = y[link.wallet_entity] == 1
    comp = pd.DataFrame({"c": link.wallet_cluster[illicit_w],
                         "e": link.wallet_entity[illicit_w]})
    comp = comp[comp.c.isin(sel)]
    by_c = {c: g.e.value_counts() for c, g in comp.groupby("c")}
    credited = set()
    for c in sel:
        vc = by_c.get(c)
        if vc is None:
            continue
        cand = vc[~vc.index.isin(credited)]
        if len(cand):
            top = cand[cand == cand.max()].index.min()
            credited.add(int(top))
    tp = len(credited)
    return {"TP": tp, "MissedPer10k": 10_000.0 * (int(y.sum()) - tp) / n,
            "k_eff": int(k_eff)}
