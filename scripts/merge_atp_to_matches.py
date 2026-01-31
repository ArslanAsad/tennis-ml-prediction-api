"""
Merge all data/atp_matches_*.csv files into a single data/matches.csv.

Output schema:
match_date,
player1_name, player2_name,
surface, tournament_level,
player1_rank, player2_rank,
player1_rank_points, player2_rank_points,
target   (1 = player1 wins, 0 = player1 loses)

Fixes:
- Avoids label leakage
- Ensures both classes exist
- Randomizes player perspective
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = DATA_DIR / "matches.csv"

LEVEL_MAP = {
    "G": "Grand Slam",
    "M": "Masters",
    "A": "ATP 500",   
    "500": "ATP 500",
    "250": "ATP 250",
    "C": "Challenger",
    "F": "Futures",
}

VALID_SURFACES = {"hard", "clay", "grass"}


def map_level(raw):
    s = str(raw).strip().upper()
    return LEVEL_MAP.get(s, "ATP 250")


def main():
    files = sorted(DATA_DIR.glob("atp_matches_*.csv"), key=lambda p: p.name)
    if not files:
        print("No atp_matches_*.csv files found.")
        return

    dfs = []

    for p in files:
        df = pd.read_csv(p, low_memory=False)
        if df.empty:
            continue

        required = {
            "tourney_date": "match_date",
            "winner_name": "winner_name",
            "loser_name": "loser_name",
            "surface": "surface",
            "tourney_level": "tournament_level",
            "winner_rank": "winner_rank",
            "loser_rank": "loser_rank",
            "winner_rank_points": "winner_rank_points",
            "loser_rank_points": "loser_rank_points",
        }

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Skipping {p.name}, missing columns: {missing}")
            continue

        out = df[list(required.keys())].rename(columns=required)

        # Date
        out["match_date"] = pd.to_datetime(
            out["match_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )

        # Normalize text fields
        out["surface"] = out["surface"].astype(str).str.strip().str.lower()
        out["tournament_level"] = out["tournament_level"].map(map_level)

        # Drop invalid rows
        out = out.dropna(subset=[
            "match_date",
            "winner_name", "loser_name",
            "winner_rank", "loser_rank",
            "winner_rank_points", "loser_rank_points",
        ])

        out = out[out["surface"].isin(VALID_SURFACES)]

        # -----------------------------
        # ML-SAFE PLAYER RANDOMIZATION
        # -----------------------------
        out["player1_name"] = out["winner_name"]
        out["player2_name"] = out["loser_name"]
        out["player1_rank"] = out["winner_rank"]
        out["player2_rank"] = out["loser_rank"]
        out["player1_rank_points"] = out["winner_rank_points"]
        out["player2_rank_points"] = out["loser_rank_points"]

        # Target: did player1 win?
        out["target"] = 1

        # Randomly swap players (50%)
        swap_mask = np.random.rand(len(out)) < 0.5

        cols_to_swap = [
            ("player1_name", "player2_name"),
            ("player1_rank", "player2_rank"),
            ("player1_rank_points", "player2_rank_points"),
        ]

        for a, b in cols_to_swap:
            out.loc[swap_mask, [a, b]] = out.loc[swap_mask, [b, a]].values

        # Flip labels for swapped rows
        out.loc[swap_mask, "target"] = 0

        dfs.append(out)

    if not dfs:
        print("No data retained after processing.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("match_date").reset_index(drop=True)
    combined["match_date"] = combined["match_date"].dt.strftime("%Y-%m-%d")

    final_cols = [
        "match_date",
        "player1_name", "player2_name",
        "surface", "tournament_level",
        "player1_rank", "player2_rank",
        "player1_rank_points", "player2_rank_points",
        "target",
    ]

    combined = combined[final_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(combined):,} rows → {OUT_PATH}")
    print("Target distribution:")
    print(combined["target"].value_counts(normalize=True))
    print(f"Date range: {combined['match_date'].min()} → {combined['match_date'].max()}")


if __name__ == "__main__":
    main()