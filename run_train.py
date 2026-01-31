"""Train models and persist to disk. Run from project root."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, MODELS_DIR
from src.train import train_and_evaluate


def main():
    data_path = DATA_DIR / "matches.csv"
    if not data_path.exists():
        print("No data/matches.csv found.")
        sys.exit(1)
    print("Training on", data_path)
    metrics, best_model, feature_names, meta = train_and_evaluate(
        csv_path=data_path,
        optimize=True,
        save_dir=MODELS_DIR,
    )
    print("Best model:", meta["model_name"])
    for m in metrics:
        print(f"  {m['model']}: accuracy={m['accuracy']:.4f}, roc_auc={m['roc_auc']:.4f}, log_loss={m['log_loss']:.4f}")
    print("Models saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
