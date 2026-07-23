"""Feature extraction for the four surveillance capability tiers.

T1  structure-only:      per-WALLET features on unlinked pseudonyms.
T2  + pseudonymous link: entity-level aggregates over the linked wallet
                          cluster (num wallets, internal flow, cross-wallet
                          pass-through) — achievable with ZK linkage.
T3  + identity attrs:    T2 + KYC tier, account age, prior SARs, jurisdiction.
T4  + watchlist:         T3 + watchlist bit.

T1 has no linkage, so its classifier operates on wallets; the entity-level
score used for evaluation is the max over the entity's wallet scores
(ground-truth aggregation at eval time only — the detector never sees it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FAST_LAG_DAYS = 2.0          # "fast pass-through" window
NO_INFLOW_LAG = 30.0         # lag fill for wallets with no prior inflow
NEAR_THR_LO, NEAR_THR_HI = 0.70, 1.00   # near-threshold band (x reporting thr)


def _wallet_side(tx, col):
    return {w: g for w, g in tx.groupby(col, sort=False)}


def build_wallet_features(data) -> pd.DataFrame:
    """Per-wallet (pseudonym) structural features — the T1 feature space."""
    tx = data["transactions"]
    wallets = data["wallets"]
    thr = data["config"].reporting_threshold

    by_dst = _wallet_side(tx, "dst")
    by_src = _wallet_side(tx, "src")

    rows = []
    for w in wallets.wallet_id:
        gin = by_dst.get(w)
        gout = by_src.get(w)
        in_t = gin.t.to_numpy() if gin is not None else np.array([])
        in_a = gin.amount.to_numpy() if gin is not None else np.array([])
        out_t = gout.t.to_numpy() if gout is not None else np.array([])
        out_a = gout.amount.to_numpy() if gout is not None else np.array([])

        total_in, total_out = in_a.sum(), out_a.sum()
        # lag from each outgoing tx to the most recent prior inflow
        if len(out_t) and len(in_t):
            idx = np.searchsorted(in_t, out_t, side="right") - 1
            lags = np.where(idx >= 0, out_t - in_t[np.clip(idx, 0, None)],
                            NO_INFLOW_LAG)
        else:
            lags = np.full(len(out_t), NO_INFLOW_LAG)
        fast_mask = lags < FAST_LAG_DAYS
        frac_fast_value = (out_a[fast_mask].sum() / total_out
                           if total_out > 0 else 0.0)
        med_lag = float(np.median(lags)) if len(lags) else NO_INFLOW_LAG

        all_a = np.concatenate([in_a, out_a])
        all_t = np.sort(np.concatenate([in_t, out_t]))
        near_thr = ((all_a >= NEAR_THR_LO * thr) & (all_a < NEAR_THR_HI * thr))
        gaps = np.diff(all_t)
        burst = (gaps.std() / gaps.mean()
                 if len(gaps) > 2 and gaps.mean() > 0 else 0.0)

        rows.append({
            "wallet_id": w,
            "n_in": len(in_a), "n_out": len(out_a),
            "total_in": total_in, "total_out": total_out,
            "mean_in": in_a.mean() if len(in_a) else 0.0,
            "mean_out": out_a.mean() if len(out_a) else 0.0,
            "max_in": in_a.max() if len(in_a) else 0.0,
            "max_out": out_a.max() if len(out_a) else 0.0,
            "in_degree": gin.src.nunique() if gin is not None else 0,
            "out_degree": gout.dst.nunique() if gout is not None else 0,
            "flow_ratio": total_out / (total_in + 1.0),
            "frac_fast_value": frac_fast_value,
            "median_out_lag": med_lag,
            "near_thr_frac": near_thr.mean() if len(all_a) else 0.0,
            "burstiness": burst,
            "amount_cv": (all_a.std() / all_a.mean()
                          if len(all_a) > 2 and all_a.mean() > 0 else 0.0),
        })
    wf = pd.DataFrame(rows).merge(wallets, on="wallet_id")
    return wf


T1_WALLET_COLS = [
    "n_in", "n_out", "total_in", "total_out", "mean_in", "mean_out",
    "max_in", "max_out", "in_degree", "out_degree", "flow_ratio",
    "frac_fast_value", "median_out_lag", "near_thr_frac", "burstiness",
    "amount_cv",
]


def build_entity_features(data, wallet_feats: pd.DataFrame) -> pd.DataFrame:
    """Entity-level feature matrix. Columns are grouped into tiers by the
    TIER_COLS mapping below."""
    tx = data["transactions"]
    wallets = data["wallets"]
    entities = data["entities"]
    thr = data["config"].reporting_threshold

    w2e = dict(zip(wallets.wallet_id, wallets.entity_id))
    tx = tx.assign(
        src_e=tx.src.map(w2e),            # NaN for external wallets
        dst_e=tx.dst.map(w2e),
    )
    internal = tx[tx.src_e.notna() & (tx.src_e == tx.dst_e)]
    ext_in = tx[tx.dst_e.notna() & (tx.src_e != tx.dst_e)]
    ext_out = tx[tx.src_e.notna() & (tx.src_e != tx.dst_e)]

    # linkage aggregates of wallet-level structure
    agg = wallet_feats.groupby("entity_id").agg(
        num_wallets=("wallet_id", "count"),
        w_frac_fast_mean=("frac_fast_value", "mean"),
        w_frac_fast_max=("frac_fast_value", "max"),
        w_min_out_lag=("median_out_lag", "min"),
        w_burst_mean=("burstiness", "mean"),
        w_near_thr_mean=("near_thr_frac", "mean"),
        w_flow_ratio_max=("flow_ratio", "max"),
    )

    ein = ext_in.groupby("dst_e").amount.agg(["sum", "count", "max"]).rename(
        columns={"sum": "ent_total_in", "count": "ent_n_in", "max": "ent_max_in"})
    eout = ext_out.groupby("src_e").amount.agg(["sum", "count"]).rename(
        columns={"sum": "ent_total_out", "count": "ent_n_out"})
    ivol = internal.groupby("src_e").amount.sum().rename("internal_vol")
    ectr = pd.concat([
        ext_in.groupby("dst_e").src.nunique(),
        ext_out.groupby("src_e").dst.nunique(),
    ], axis=1).fillna(0).sum(axis=1).rename("n_ext_counterparties")

    # cross-wallet chain pass-through: value leaving the entity externally
    # within FAST_LAG_DAYS of an INTERNAL inflow to the sending wallet,
    # normalized by total internal volume — the layering signal proper.
    int_by_dst = _wallet_side(internal, "dst")
    chain_fast = {}
    for e, g in ext_out.groupby("src_e", sort=False):
        num = 0.0
        for w, gw in g.groupby("src", sort=False):
            gi = int_by_dst.get(w)
            if gi is None:
                continue
            it = gi.t.to_numpy()
            ot, oa = gw.t.to_numpy(), gw.amount.to_numpy()
            idx = np.searchsorted(it, ot, side="right") - 1
            lag = np.where(idx >= 0, ot - it[np.clip(idx, 0, None)], np.inf)
            num += oa[lag < FAST_LAG_DAYS].sum()
        chain_fast[e] = num
    chain = pd.Series(chain_fast, name="chain_fast_value")

    near = tx[tx.src_e.notna() | tx.dst_e.notna()].copy()
    near["is_near"] = ((near.amount >= NEAR_THR_LO * thr)
                       & (near.amount < NEAR_THR_HI * thr))
    ent_of = near.src_e.fillna(near.dst_e)
    near_frac = near.groupby(ent_of).is_near.mean().rename("ent_near_thr_frac")

    ef = agg.join([ein, eout, ivol, ectr, chain, near_frac]).fillna(0.0)
    total_vol = ef.ent_total_in + ef.ent_total_out + ef.internal_vol
    ef["internal_share"] = ef.internal_vol / (total_vol + 1.0)
    ef["ent_flow_ratio"] = ef.ent_total_out / (ef.ent_total_in + 1.0)
    ef["chain_fast_share"] = ef.chain_fast_value / (ef.internal_vol + 1.0)

    ef = ef.reset_index().rename(columns={"index": "entity_id"})
    ef = ef.merge(entities, on="entity_id", how="right").fillna(0.0)
    return ef


T2_COLS = [
    "num_wallets", "w_frac_fast_mean", "w_frac_fast_max", "w_min_out_lag",
    "w_burst_mean", "w_near_thr_mean", "w_flow_ratio_max",
    "ent_total_in", "ent_n_in", "ent_max_in", "ent_total_out", "ent_n_out",
    "internal_vol", "n_ext_counterparties", "chain_fast_value",
    "ent_near_thr_frac", "internal_share", "ent_flow_ratio",
    "chain_fast_share",
]
T3_COLS = T2_COLS + [
    "kyc_tier", "account_age_days", "prior_sar_count", "jurisdiction_risk",
]
T4_COLS = T3_COLS + ["on_watchlist"]

TIER_COLS = {"T2": T2_COLS, "T3": T3_COLS, "T4": T4_COLS}
