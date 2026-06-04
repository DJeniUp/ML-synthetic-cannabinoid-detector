"""Walk-forward retraining.

For each cutoff year in [start, end), trains on molecules published in
year <= cutoff and tests on the single next year (year == cutoff + 1).
This simulates periodic retraining as new synthetic cannabinoids appear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.config import load_config
from pipeline.data import load_full_clean
from pipeline.trainer import Trainer

START_YEAR = 2010
END_YEAR = 2024  # last cutoff; tests on END_YEAR+1 data


def _run_one(
    full_df: pd.DataFrame, cutoff: int, trainer_cfg: dict
) -> dict | None:
    train_df = full_df[full_df["year"] <= cutoff].copy()
    test_df = full_df[full_df["year"] == cutoff + 1].copy()

    if len(train_df) < 50 or len(test_df) < 10:
        return None

    trainer = Trainer(trainer_cfg)
    X_train, y_train = trainer.featurize(train_df)
    X_test, y_test = trainer.featurize(test_df)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)

    return {
        "cutoff_year": cutoff,
        "test_year": cutoff + 1,
        "train_size": len(X_train),
        "test_size": len(X_test),
        **metrics,
    }


def main() -> None:
    cfg = load_config()
    print("Loading full cleaned dataset …")
    full_df = load_full_clean(cfg)

    results = []
    for cutoff in range(START_YEAR, END_YEAR + 1):
        row = _run_one(full_df, cutoff, cfg)
        if row is None:
            print(f"  cutoff {cutoff}: skipped (too few molecules in train or test year).")
            continue
        results.append(row)
        print(
            f"  cutoff {cutoff} | test {row['test_year']} "
            f"| train={row['train_size']} test={row['test_size']} "
            f"| RMSE={row['rmse']:.3f} R²={row['r2']:.3f} "
            f"| miss={row['dangerous_miss_rate_pct']:.1f}%"
        )

    if not results:
        print("No valid walk-forward windows found.")
        return

    results_df = pd.DataFrame(results)
    out_path = Path(__file__).parent / "models" / "walkforward_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    _plot(results_df)


def _plot(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Walk-forward retraining: CB1 cannabinoid detector", fontsize=14)

    axes[0, 0].plot(df["cutoff_year"], df["rmse"], marker="o")
    axes[0, 0].set_title("RMSE on next-year test set")
    axes[0, 0].set_xlabel("Cutoff year")
    axes[0, 0].set_ylabel("RMSE")

    axes[0, 1].plot(df["cutoff_year"], df["r2"], marker="o", color="green")
    axes[0, 1].set_title("R² on next-year test set")
    axes[0, 1].set_xlabel("Cutoff year")
    axes[0, 1].set_ylabel("R²")

    axes[1, 0].plot(df["cutoff_year"], df["dangerous_miss_rate_pct"], marker="o", color="red")
    axes[1, 0].set_title("Dangerous miss rate (%) — active predicted inactive")
    axes[1, 0].set_xlabel("Cutoff year")
    axes[1, 0].set_ylabel("Miss rate (%)")

    axes[1, 1].plot(df["cutoff_year"], df["train_size"], marker="o", label="Train", color="steelblue")
    axes[1, 1].plot(df["cutoff_year"], df["test_size"], marker="s", label="Test", color="orange")
    axes[1, 1].set_title("Dataset sizes over time")
    axes[1, 1].set_xlabel("Cutoff year")
    axes[1, 1].set_ylabel("Number of molecules")
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = Path(__file__).parent / "models" / "walkforward_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
