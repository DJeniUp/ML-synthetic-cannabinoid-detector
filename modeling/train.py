"""Entry point: load config → fetch/load data → train → evaluate → save."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.config import load_config
from pipeline.data import load_or_fetch
from pipeline.trainer import Trainer


def main() -> None:
    cfg = load_config()
    train_df, test_df = load_or_fetch(cfg)

    print(f"\nTrain: {len(train_df)} molecules | Test: {len(test_df)} molecules")

    trainer = Trainer(cfg)
    metrics = trainer.run(train_df, test_df)

    print("\n=== Final Metrics ===")
    print(f"  RMSE:              {metrics['rmse']:.4f}")
    print(f"  MAE:               {metrics['mae']:.4f}")
    print(f"  R²:                {metrics['r2']:.4f}")
    print(f"  Dangerous misses:  {metrics['dangerous_miss_count']} / {metrics['n_active']} "
          f"({metrics['dangerous_miss_rate_pct']:.1f}%)")
    print(f"  Danger-aware RMSE: {metrics['danger_aware_rmse']:.4f}")


if __name__ == "__main__":
    main()
