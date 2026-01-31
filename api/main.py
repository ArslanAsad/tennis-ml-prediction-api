"""
REST API for tennis match prediction.
Endpoints: POST /predict, GET /players, GET /model-info. Loads model at startup.
"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

# Project root for imports
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MODELS_DIR, DATA_DIR, VALID_SURFACES
from src.predictor import TennisPredictor
from src.data_loader import load_and_clean


# Pydantic schemas
class PredictRequest(BaseModel):
    player1_name: str = Field(..., description="Player 1 name")
    player2_name: str = Field(..., description="Player 2 name")
    surface: str = Field(..., description="Surface: hard, clay, or grass")
    tournament_level: str = Field(default="ATP 250", description="Tournament level")
    explain: bool = Field(default=False, description="Whether to return top contributing features")

    class Config:
        json_schema_extra = {
            "example": {
                "player1_name": "Novak Djokovic",
                "player2_name": "Rafael Nadal",
                "surface": "clay",
                "tournament_level": "Grand Slam",
                "explain": True
            }
        }


class PredictResponse(BaseModel):
    predicted_winner: str
    player1_win_probability: float
    player2_win_probability: float
    model_confidence: float
    top_contributing_features: list[dict]


# App and startup
app = FastAPI(
    title="Tennis Match Prediction API",
    description="ML-based tennis match winner prediction with win probabilities and explainability",
    version="1.0.0",
)

predictor: TennisPredictor | None = None
players_cache: list[str] = []
metadata_cache: dict = {}


@app.on_event("startup")
def load_model_at_startup():
    """Load trained model and optional historical data at startup."""
    global predictor, players_cache, metadata_cache
    try:
        player_features = joblib.load(MODELS_DIR / "player_features.joblib")
        h2h_features = joblib.load(MODELS_DIR / "h2h_features.joblib")
        predictor = TennisPredictor(model_dir=MODELS_DIR, player_features=player_features, h2h_features=h2h_features)
        if predictor.metadata:
            metadata_cache = predictor.metadata
            players_cache = metadata_cache.get("players", [])
    except FileNotFoundError as e:
        predictor = None
        metadata_cache = {"error": str(e), "message": "Train a model first (run train script)."}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict match outcome: winner, win probabilities, confidence, top features."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and persist a model first.")
    surface = request.surface.strip().lower()
    if surface not in VALID_SURFACES:
        raise HTTPException(
            status_code=422,
            detail=f"surface must be one of: {list(VALID_SURFACES)}",
        )
    try:
        out = predictor.predict(
            player1_name=request.player1_name,
            player2_name=request.player2_name,
            surface=surface,
            tournament_level=request.tournament_level,
            explain=request.explain,
        )
        return PredictResponse(**out)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/players")
def list_players():
    """List available players (from historical data used for training)."""
    return {"players": players_cache, "count": len(players_cache)}


@app.get("/model-info")
def model_info():
    """Model metadata: name, feature names, metrics."""
    if metadata_cache and "error" in metadata_cache:
        return metadata_cache
    return {
        "model_name": metadata_cache.get("model_name"),
        "feature_names": metadata_cache.get("feature_names", []),
        "metrics": metadata_cache.get("metrics", {}),
        "n_train": metadata_cache.get("n_train"),
        "n_test": metadata_cache.get("n_test"),
        "test_size": metadata_cache.get("test_size"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}