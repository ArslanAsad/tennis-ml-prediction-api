"""
Offline feature store builder for fast inference.

Uses the feature pipeline:
- compute_elo_ratings
- add_player_and_surface_features
- add_h2h_features

Outputs:
- player_features.joblib
- h2h_features.joblib
"""

import joblib
from collections import defaultdict
from pathlib import Path

# Project root for imports
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MODELS_DIR
from src.data_loader import load_and_clean
from src.features import (
    compute_elo_ratings,
    add_player_and_surface_features,
    add_h2h_features,
)

# Paths
DATA_PATH = DATA_DIR / "matches.csv"
OUTPUT_DIR = MODELS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLAYER_FEATURES_PATH = OUTPUT_DIR / "player_features.joblib"
H2H_FEATURES_PATH = OUTPUT_DIR / "h2h_features.joblib"


def build_feature_store():
    print("Loading historical data...")
    df = load_and_clean(csv_path=DATA_PATH)

    print("Running feature pipeline (once)...")
    df = compute_elo_ratings(df)
    df = add_player_and_surface_features(df)
    df = add_h2h_features(df)

    print("Extracting player features...")
    player_features = {}

    # Last known stats per player
    df_sorted = df.sort_values("match_date")

    players = set(df["player1_name"]) | set(df["player2_name"])

    for player in players:
        p1_rows = df_sorted[df_sorted["player1_name"] == player]
        p2_rows = df_sorted[df_sorted["player2_name"] == player]

        if p1_rows.empty and p2_rows.empty:
            continue

        last_rows = []
        if not p1_rows.empty:
            last_rows.append(p1_rows.iloc[-1])
        if not p2_rows.empty:
            last_rows.append(p2_rows.iloc[-1])

        # Most recent match
        last = max(last_rows, key=lambda r: r["match_date"])

        if last["player1_name"] == player:
            prefix = "player1"
        else:
            prefix = "player2"

        player_features[player] = {
            "elo": float(last[f"{prefix}_elo"]),
            "rank": float(last.get(f"{prefix}_rank", 500)),
            "rank_points": float(last.get(f"{prefix}_rank_points", 0)),
            "surface_winrate": {
                "hard": float(last.get(f"{prefix}_hard_winrate", 0.5)),
                "clay": float(last.get(f"{prefix}_clay_winrate", 0.5)),
                "grass": float(last.get(f"{prefix}_grass_winrate", 0.5)),
            },
        }

    print(f"Built player_features for {len(player_features)} players")

    # Head-to-head features
    print("Extracting head-to-head features...")
    h2h_features = defaultdict(lambda: {"wins_p1": 0, "wins_p2": 0, "matches": 0})

    for _, row in df.iterrows():
        p1 = row["player1_name"]
        p2 = row["player2_name"]

        key = (p1, p2)
        h2h_features[key]["matches"] += 1

        if row["target"] == 1:
            h2h_features[key]["wins_p1"] += 1
        else:
            h2h_features[key]["wins_p2"] += 1

    h2h_features = dict(h2h_features)
    print(f"Built H2H features for {len(h2h_features)} matchups")

    print("Saving feature stores...")
    joblib.dump(player_features, PLAYER_FEATURES_PATH)
    joblib.dump(h2h_features, H2H_FEATURES_PATH)

    print("Feature store build complete")
    print(f"→ {PLAYER_FEATURES_PATH}")
    print(f"→ {H2H_FEATURES_PATH}")


if __name__ == "__main__":
    build_feature_store()