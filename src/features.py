"""
Feature engineering for tennis match prediction.

Leakage-free version:
- Uses binary `target` instead of `winner`
- All features computed using ONLY pre-match information
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple

from config import RECENT_MATCHES_N, H2H_RECENT_N, ELO_K, ELO_INITIAL, VALID_SURFACES


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _update_elo(elo_a: float, elo_b: float, result_a: int) -> Tuple[float, float]:
    """Update Elo where result_a = 1 if A wins, 0 otherwise"""
    exp_a = _expected_score(elo_a, elo_b)
    exp_b = 1.0 - exp_a
    new_a = elo_a + ELO_K * (result_a - exp_a)
    new_b = elo_b + ELO_K * ((1 - result_a) - exp_b)
    return new_a, new_b


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute global and surface-specific Elo ratings."""
    df = df.sort_values("match_date").reset_index(drop=True)

    global_elo = defaultdict(lambda: ELO_INITIAL)
    surface_elo = defaultdict(lambda: ELO_INITIAL)

    p1_elo, p2_elo = [], []
    p1_elo_s, p2_elo_s = [], []

    for _, row in df.iterrows():
        p1, p2 = row["player1_name"], row["player2_name"]
        surf = row["surface"]
        result = row["target"]

        e1, e2 = global_elo[p1], global_elo[p2]
        s1, s2 = surface_elo[(p1, surf)], surface_elo[(p2, surf)]

        p1_elo.append(e1)
        p2_elo.append(e2)
        p1_elo_s.append(s1)
        p2_elo_s.append(s2)

        global_elo[p1], global_elo[p2] = _update_elo(e1, e2, result)
        surface_elo[(p1, surf)], surface_elo[(p2, surf)] = _update_elo(s1, s2, result)

    df = df.copy()
    df["player1_elo"] = p1_elo
    df["player2_elo"] = p2_elo
    df["player1_elo_surface"] = p1_elo_s
    df["player2_elo_surface"] = p2_elo_s
    return df


def add_player_and_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Career, season, recent form, surface stats."""
    df = df.sort_values("match_date").reset_index(drop=True)
    df["year"] = pd.to_datetime(df["match_date"]).dt.year

    wins = defaultdict(int)
    matches = defaultdict(int)
    season_wins = defaultdict(lambda: defaultdict(int))
    season_matches = defaultdict(lambda: defaultdict(int))
    recent = defaultdict(list)
    surf_wins = defaultdict(lambda: defaultdict(int))
    surf_matches = defaultdict(lambda: defaultdict(int))

    out = defaultdict(list)

    for _, r in df.iterrows():
        p1, p2 = r["player1_name"], r["player2_name"]
        y = r["target"]
        year, surf = r["year"], r["surface"]

        def pct(w, m): return w / m if m else 0.5

        out["player1_career_win_pct"].append(pct(wins[p1], matches[p1]))
        out["player2_career_win_pct"].append(pct(wins[p2], matches[p2]))

        out["player1_season_win_pct"].append(
            pct(season_wins[p1][year], season_matches[p1][year]) if season_matches[p1][year] else np.nan
        )
        out["player2_season_win_pct"].append(
            pct(season_wins[p2][year], season_matches[p2][year]) if season_matches[p2][year] else np.nan
        )

        out["player1_recent_form"].append(
            np.mean(recent[p1][-RECENT_MATCHES_N:]) if len(recent[p1]) >= RECENT_MATCHES_N else np.nan
        )
        out["player2_recent_form"].append(
            np.mean(recent[p2][-RECENT_MATCHES_N:]) if len(recent[p2]) >= RECENT_MATCHES_N else np.nan
        )

        out["player1_surface_win_rate"].append(
            pct(surf_wins[p1][surf], surf_matches[p1][surf]) if surf_matches[p1][surf] else np.nan
        )
        out["player2_surface_win_rate"].append(
            pct(surf_wins[p2][surf], surf_matches[p2][surf]) if surf_matches[p2][surf] else np.nan
        )

        out["player1_surface_matches"].append(surf_matches[p1][surf])
        out["player2_surface_matches"].append(surf_matches[p2][surf])

        # update stats AFTER match
        matches[p1] += 1
        matches[p2] += 1
        season_matches[p1][year] += 1
        season_matches[p2][year] += 1
        surf_matches[p1][surf] += 1
        surf_matches[p2][surf] += 1

        recent[p1].append(y)
        recent[p2].append(1 - y)

        if y == 1:
            wins[p1] += 1
            season_wins[p1][year] += 1
            surf_wins[p1][surf] += 1
        else:
            wins[p2] += 1
            season_wins[p2][year] += 1
            surf_wins[p2][surf] += 1

    for k, v in out.items():
        df[k] = v
    return df


def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head features."""
    df = df.sort_values("match_date").reset_index(drop=True)
    h2h = defaultdict(list)

    total, ratio, recent = [], [], []

    for _, r in df.iterrows():
        p1, p2, y = r["player1_name"], r["player2_name"], r["target"]
        key = tuple(sorted([p1, p2]))
        past = h2h[key]

        total.append(len(past))
        if past:
            wins_p1 = sum(1 for w in past if w == p1)
            ratio.append(wins_p1 / len(past))
            recent.append(
                sum(1 for w in past[-H2H_RECENT_N:] if w == p1) / min(len(past), H2H_RECENT_N)
                if len(past) >= H2H_RECENT_N else np.nan
            )
        else:
            ratio.append(0.5)
            recent.append(np.nan)

        h2h[key].append(p1 if y == 1 else p2)

    df["h2h_matches"] = total
    df["h2h_win_ratio_p1"] = ratio
    df["h2h_recent_p1"] = recent
    return df


def to_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to player1 − player2 difference features."""
    diff = pd.DataFrame(index=df.index)

    pairs = [
        ("player1_rank", "player2_rank", "rank_diff"),
        ("player1_rank_points", "player2_rank_points", "rank_points_diff"),
        ("player1_career_win_pct", "player2_career_win_pct", "career_win_pct_diff"),
        ("player1_season_win_pct", "player2_season_win_pct", "season_win_pct_diff"),
        ("player1_recent_form", "player2_recent_form", "recent_form_diff"),
        ("player1_surface_win_rate", "player2_surface_win_rate", "surface_win_rate_diff"),
        ("player1_surface_matches", "player2_surface_matches", "surface_matches_diff"),
        ("player1_elo", "player2_elo", "elo_diff"),
        ("player1_elo_surface", "player2_elo_surface", "elo_surface_diff"),
    ]

    for a, b, name in pairs:
        diff[name] = df[a] - df[b]

    diff["h2h_win_ratio_diff"] = (df["h2h_win_ratio_p1"] - 0.5) * 2
    diff["h2h_recent_diff"] = df["h2h_recent_p1"].fillna(0.5) - 0.5
    diff["h2h_matches"] = df["h2h_matches"]

    for s in VALID_SURFACES:
        diff[f"surface_{s}"] = (df["surface"] == s).astype(int)

    diff["target"] = df["target"]
    diff["match_date"] = df["match_date"]
    diff["player1_name"] = df["player1_name"]
    diff["player2_name"] = df["player2_name"]
    diff["surface"] = df["surface"]
    diff["tournament_level"] = df["tournament_level"]
    return diff


def build_feature_matrix(df: pd.DataFrame):
    df = compute_elo_ratings(df)
    df = add_player_and_surface_features(df)
    df = add_h2h_features(df)
    diff = to_difference_features(df)

    exclude = {"target", "match_date", "player1_name", "player2_name", "surface", "tournament_level"}
    feature_cols = [c for c in diff.columns if c not in exclude]

    X = diff[feature_cols].fillna(0)
    y = diff["target"]
    return X, y, feature_cols