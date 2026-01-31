"""
Prediction and explainability.
Accepts player1, player2, surface, tournament_level; outputs winner, probabilities, confidence, top features.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import joblib
import json

from config import MODELS_DIR, VALID_SURFACES

class TennisPredictor:
    """
    Loads a trained model and optional historical data; provides predict() with probabilities.
    Uses precomputed player + h2h feature stores.
    """

    def __init__(
        self,
        model_dir: Path = MODELS_DIR,
        player_features: Dict[str, dict] | None = None,
        h2h_features: Dict[Tuple[str, str], dict] | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "model.joblib")
        self.feature_names = joblib.load(self.model_dir / "feature_names.joblib")
        meta_path = self.model_dir / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.player_features = player_features or {}
        self.h2h_features = h2h_features or {}
        self._feature_index = {f: i for i, f in enumerate(self.feature_names)}
    

    def _build_feature_vector(
        self,
        p1: str,
        p2: str,
        surface: str,
        tournament_level: str,
    ) -> np.ndarray:
        if surface not in VALID_SURFACES:
            raise ValueError(f"surface must be one of {VALID_SURFACES}")
        if p1 not in self.player_features:
            raise ValueError(f"Unknown player: {p1}")
        if p2 not in self.player_features:
            raise ValueError(f"Unknown player: {p2}")

        f1 = self.player_features[p1]
        f2 = self.player_features[p2]
        h2h = self.h2h_features.get(
            (p1, p2),
            {"wins_p1": 0, "wins_p2": 0, "matches": 0},
        )
        x = np.zeros(len(self.feature_names), dtype=np.float32)
        def setf(name, value):
            idx = self._feature_index.get(name)
            if idx is not None:
                x[idx] = value

        # Core diffs
        setf("elo_diff", f1["elo"] - f2["elo"])
        setf("rank_diff", f1["rank"] - f2["rank"])
        setf("rank_points_diff", f1["rank_points"] - f2["rank_points"])

        # Surface winrate
        setf(
            "surface_winrate_diff",
            f1["surface_winrate"].get(surface, 0.5)
            - f2["surface_winrate"].get(surface, 0.5),
        )

        # Head-to-head
        if h2h["matches"] > 0:
            h2h_wr = h2h["wins_p1"] / h2h["matches"]
        else:
            h2h_wr = 0.5
        setf("h2h_winrate", h2h_wr)

        # Tournament level
        setf(f"level_{tournament_level}", 1.0)

        return x.reshape(1, -1)


    def predict(
        self,
        player1_name: str,
        player2_name: str,
        surface: str,
        tournament_level: str,
        explain: bool = False,
    ) -> dict:
        X = self._build_feature_vector(
            player1_name.strip(),
            player2_name.strip(),
            surface.strip().lower(),
            tournament_level,
        )
        prob_p1 = float(self.model.predict_proba(X)[0, 1])
        prob_p2 = 1.0 - prob_p1
        winner = player1_name if prob_p1 >= 0.5 else player2_name
        result = {
            "predicted_winner": winner,
            "player1_win_probability": round(prob_p1, 4),
            "player2_win_probability": round(prob_p2, 4),
            "model_confidence": round(max(prob_p1, prob_p2), 4),
        }
        if explain:
            result["top_contributing_features"] = self._feature_importance(top_k=10)
        else:
            result["top_contributing_features"] = []
        return result


    def _feature_importance(self, top_k: int = 10) -> list:
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_[0])
        else:
            return []

        pairs = list(zip(self.feature_names, imp))
        pairs.sort(key=lambda x: x[1], reverse=True)

        return [
            {"feature": f, "contribution": round(float(v), 4)}
            for f, v in pairs[:top_k]
        ]