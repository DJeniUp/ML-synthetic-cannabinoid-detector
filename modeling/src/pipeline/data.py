"""Data loading, cleaning, deduplication, and temporal split.

Cache behaviour: if train_path and test_path both exist they are returned immediately.
Otherwise the raw CSV is used (fetching from ChEMBL if that is also absent).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def _fetch_from_chembl(cfg: dict[str, Any]) -> pd.DataFrame:
    from chembl_webresource_client.new_client import new_client  # type: ignore

    activity = new_client.activity
    res = activity.filter(
        target_chembl_id=cfg["target_id"],
        pchembl_value__isnull=False,
        standard_type__in=cfg["activity_types"],
    ).only(
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_type",
            "standard_value",
            "pchembl_value",
            "document_chembl_id",
            "document_year",
        ]
    )
    raw = pd.DataFrame(list(res))
    raw_path = cfg["data"]["raw_path"]
    os.makedirs(Path(raw_path).parent, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    print(f"Fetched {len(raw)} records from ChEMBL → {raw_path}")
    return raw


def _clean_and_split(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["pchembl_value"] = pd.to_numeric(raw["pchembl_value"], errors="coerce")
    raw["document_year"] = pd.to_numeric(raw["document_year"], errors="coerce")
    raw = raw.dropna(subset=["canonical_smiles", "pchembl_value", "document_year"])

    clean = raw.groupby(
        ["molecule_chembl_id", "canonical_smiles"], as_index=False
    ).agg(
        pchembl_value=("pchembl_value", "mean"),
        year=("document_year", "min"),
    )
    clean["year"] = clean["year"].astype(int)

    cutoff = cfg["cutoff_year"]
    train_df = clean[clean["year"] <= cutoff].copy()
    test_df = clean[clean["year"] > cutoff].copy()

    os.makedirs(Path(cfg["data"]["train_path"]).parent, exist_ok=True)
    train_df.to_csv(cfg["data"]["train_path"], index=False)
    test_df.to_csv(cfg["data"]["test_path"], index=False)
    print(
        f"Split: train={len(train_df)} (≤{cutoff}), test={len(test_df)} (>{cutoff}). "
        f"Saved to {cfg['data']['train_path']} and {cfg['data']['test_path']}"
    )
    return train_df, test_df


def load_or_fetch(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df).

    Resolution order:
    1. Both CSVs cached on disk → read and return.
    2. Raw CSV on disk → clean/dedup/split.
    3. Neither → fetch from ChEMBL, then clean/dedup/split.
    """
    train_path = cfg["data"]["train_path"]
    test_path = cfg["data"]["test_path"]
    raw_path = cfg["data"]["raw_path"]

    if Path(train_path).exists() and Path(test_path).exists():
        print(f"Loading cached splits from {train_path} and {test_path}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    if Path(raw_path).exists():
        print(f"Raw CSV found at {raw_path}. Cleaning and splitting …")
        raw = pd.read_csv(raw_path)
    else:
        print("No cached data found. Fetching from ChEMBL (this may take a minute) …")
        raw = _fetch_from_chembl(cfg)

    return _clean_and_split(raw, cfg)


def load_full_clean(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return all deduplicated molecules (union of train+test). Used by retrain.py."""
    raw_path = cfg["data"]["raw_path"]
    if not Path(raw_path).exists():
        _fetch_from_chembl(cfg)
    raw = pd.read_csv(raw_path)
    raw["pchembl_value"] = pd.to_numeric(raw["pchembl_value"], errors="coerce")
    raw["document_year"] = pd.to_numeric(raw["document_year"], errors="coerce")
    raw = raw.dropna(subset=["canonical_smiles", "pchembl_value", "document_year"])
    clean = raw.groupby(
        ["molecule_chembl_id", "canonical_smiles"], as_index=False
    ).agg(
        pchembl_value=("pchembl_value", "mean"),
        year=("document_year", "min"),
    )
    clean["year"] = clean["year"].astype(int)
    return clean
