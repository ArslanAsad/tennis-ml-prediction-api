"""Configuration for tennis match prediction system."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Data schema (minimum required columns)
REQUIRED_COLUMNS = [
    "match_date",
    "player1_name",
    "player2_name",
    "surface",
    "tournament_level",
    "player1_rank",
    "player2_rank",
    "player1_rank_points",
    "player2_rank_points",
    "target",
]

# Valid values
VALID_SURFACES = {"hard", "clay", "grass"}
VALID_TOURNAMENT_LEVELS = {"Grand Slam", "Masters", "ATP 500", "ATP 250", "Challenger", "Futures"}

# Feature engineering
RECENT_MATCHES_N = 10  # Last N matches for recent form
H2H_RECENT_N = 5  # Recent H2H outcomes
ELO_K = 32
ELO_INITIAL = 1500

# Training
TEST_SIZE = 0.2  # Time-based: last 20% of matches by date
RANDOM_STATE = 42
CV_FOLDS = 5

# Model names
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
DEFAULT_MODEL = "xgboost"
