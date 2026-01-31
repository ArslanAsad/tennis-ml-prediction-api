"""
Data loader for historical tennis match data.

Loads from CSV, validates schema, removes incomplete records.
ML-safe version:
- No 'winner' column
- Uses binary 'target' label (1 = player1 wins, 0 = player1 loses)
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from config import REQUIRED_COLUMNS, VALID_SURFACES, DATA_DIR


# Column aliases (future-proofing)
COLUMN_ALIASES = {
    "match_date": ["match_date", "date", "Match date", "tourney_date"],
    "player1_name": ["player1_name", "player_1", "Player 1", "player1"],
    "player2_name": ["player2_name", "player_2", "Player 2", "player2"],
    "surface": ["surface", "Surface"],
    "tournament_level": ["tournament_level", "tourney_level", "level", "Tournament level"],
    "player1_rank": ["player1_rank", "rank_1", "Player 1 rank"],
    "player2_rank": ["player2_rank", "rank_2", "Player 2 rank"],
    "player1_rank_points": ["player1_rank_points", "points_1", "Player 1 ranking points"],
    "player2_rank_points": ["player2_rank_points", "points_2", "Player 2 ranking points"],
    "target": ["target", "label", "y"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map CSV columns to canonical names."""
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in mapping.values():
                mapping[alias] = canonical
                break

    # Accept exact canonical names
    for c in df.columns:
        if c in REQUIRED_COLUMNS and c not in mapping:
            mapping[c] = c

    if mapping:
        df = df.rename(columns={k: v for k, v in mapping.items() if k != v})

    return df


def load_matches(
    csv_path: Optional[Path] = None,
    dataframe: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load historical tennis match data from CSV or DataFrame.
    Returns DataFrame with canonical column names.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.is_absolute():
            path = DATA_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
    elif dataframe is not None:
        df = dataframe.copy()
    else:
        raise ValueError("Provide either csv_path or dataframe")

    df = _normalize_columns(df)

    # Schema validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate schema and remove incomplete records.
    Drops rows with missing required fields or invalid surface.
    """
    df = df[REQUIRED_COLUMNS].copy()

    # Drop rows with nulls in required columns
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Parse and validate dates
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"])

    # Normalize and validate surface
    df["surface"] = df["surface"].astype(str).str.strip().str.lower()
    df = df[df["surface"].isin(VALID_SURFACES)]

    # Normalize player names
    df["player1_name"] = df["player1_name"].astype(str).str.strip()
    df["player2_name"] = df["player2_name"].astype(str).str.strip()

    # Numeric columns
    numeric_cols = [
        "player1_rank",
        "player2_rank",
        "player1_rank_points",
        "player2_rank_points",
        "target",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    # Binary target
    df = df[df["target"].isin([0, 1])]

    # Sort for time-based splits / Elo later
    df = df.sort_values("match_date").reset_index(drop=True)

    return df


def load_and_clean(
    csv_path: Optional[Path] = None,
    dataframe: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load matches and return validated, cleaned DataFrame."""
    df = load_matches(csv_path=csv_path, dataframe=dataframe)
    return validate_and_clean(df)