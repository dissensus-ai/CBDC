"""Data loaders: the interface where real AML data replaces the synthetic DGP.

Every loader returns the SAME dict the synthetic generator produces, so
degeneracy_audit.py / detection_experiment.py / equivalence_test.py run
unchanged on real data:

    {
      "entities":     DataFrame[entity_id, is_launderer, kyc_tier,
                                account_age_days, prior_sar_count,
                                jurisdiction_risk, on_watchlist],
      "wallets":      DataFrame[wallet_id, entity_id],
      "transactions": DataFrame[src, dst, amount, t],
      "config":       object with .reporting_threshold (float),
    }

Identity columns that a dataset genuinely lacks should be filled with
zeros AND the affected tiers (T3/T4) reported as not-evaluable on that
dataset — do not impute fake identity signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dgp import DGPConfig, generate


@dataclass
class RealDataConfig:
    reporting_threshold: float = 10_000.0
    seed: int = 20260707


def load_synthetic(cfg: DGPConfig | None = None) -> dict:
    """The synthetic DGP behind the standard interface."""
    return generate(cfg or DGPConfig())


def load_amlworld(tx_csv_path: str, accounts_csv_path: str,
                  cfg: RealDataConfig | None = None) -> dict:
    """AMLworld / IBM synthetic AML (Altman et al. 2023, NeurIPS) — the
    design pack's PRIMARY real dataset.

    Get the data: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
    (HI-Small is the right starting size; ~5M tx.)

    Mapping to the standard schema:
      * transactions: from the tx CSV — src = "From Bank"+"Account",
        dst = "To Bank"+"Account", amount = "Amount Received" (single
        currency slice, or convert), t = timestamp in fractional days
        from the dataset start.
      * wallets: AMLworld accounts are the wallets. entity_id: AMLworld
        models entities behind accounts via the bank+entity structure in
        the accounts file; where the release only exposes accounts, use
        account = entity (1 wallet each) and say so — T2's linkage axis
        then comes from the generator's known account->entity map if the
        run used the full generator output.
      * is_launderer: entity has >= 1 account touching a laundering-
        labeled transaction ("Is Laundering" == 1).
      * identity columns: AMLworld has no KYC/watchlist axes — fill 0 and
        mark T3/T4 as not-evaluable (see module docstring).
    """
    raise NotImplementedError(
        "Real-data adapter: download AMLworld (see docstring), then "
        "implement the column mapping above. The rest of the pipeline "
        "runs unchanged.")


def load_elliptic(nodes_csv_path: str, edges_csv_path: str,
                  classes_csv_path: str,
                  cfg: RealDataConfig | None = None) -> dict:
    """Elliptic Bitcoin dataset (Weber et al. 2019) — the design pack's
    real-data CROSS-CHECK.

    Get the data: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    (203k tx nodes, 234k edges, 2% illicit / 21% licit / 77% unknown.)

    Mapping notes:
      * Elliptic nodes are TRANSACTIONS, not wallets: build the wallet
        graph from the edge list, or run the tier ablation at node level
        with time-step-disjoint folds (train past -> test future, per
        Alarab & Prakoonwit) instead of entity-disjoint folds.
      * Linkage tier (T2): Elliptic has no native entity axis — construct
        a PROXY entity map via co-spend / common-input-ownership
        clustering and label it a proxy in every table it feeds.
      * Drop 'unknown'-class nodes from train AND test (do not treat
        unknown as licit).
      * Identity columns: none exist — T3/T4 not evaluable; Elliptic
        only cross-checks the T1 -> T2 step.
    """
    raise NotImplementedError(
        "Real-data adapter: download Elliptic (see docstring), then "
        "implement the mapping above.")
