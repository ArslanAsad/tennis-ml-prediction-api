# Tennis Match Prediction System

ML-based prediction of professional tennis match winners (ATP singles) using historical performance, rankings, surface, and contextual data. Provides win probabilities, model explainability (SHAP/feature importance), and a REST API.

**No post-match statistics are used** — only data known before the match (rankings, Elo, career/season win %, surface stats, H2H).

## Features

- **Data**: Load CSV with match date, both players, winner, surface, tournament level, rankings and points.
- **Features**: Player-based (ranking, points, career/season win %, recent form), surface-specific (win rate, matches), H2H (total matches, win ratio, recent H2H), global and surface Elo; all converted to player1 − player2 diff features.
- **Models**: Logistic Regression, Random Forest, XGBoost; time-based train/test split; hyperparameter tuning with cross-validation; persistence and evaluation (accuracy, ROC-AUC, log loss, calibration); comparison with “higher rank wins” and “higher Elo wins” baselines.
- **Prediction**: Input player1, player2, surface, tournament level → predicted winner, win probabilities, confidence, top contributing features.
- **API**: FastAPI with `POST /predict`, `GET /players`, `GET /model-info`; model loaded at startup.

## Data Source

This project uses historical ATP match data from [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp). The CSV files from that repository are merged and processed to create the dataset used for training the models.

## Setup

```bash
cd tennis-ml-predictor
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Data format

CSV with at least these columns (names can vary; see `src/data_loader.py` for aliases):

| Column              | Description                                  |
| ------------------- | -------------------------------------------- |
| match_date          | Date of match                                |
| player1_name        | First player                                 |
| player2_name        | Second player                                |
| surface             | hard / clay / grass                          |
| tournament_level    | e.g. Grand Slam                              |
| player1_rank        | Ranking (integer)                            |
| player2_rank        | Ranking (integer)                            |
| player1_rank_points | Ranking points                               |
| player2_rank_points | Ranking points                               |
| target              | Winner (1 = player1 wins, 0 = player1 loses) |

ATP-style columns like `winner_name`, `loser_name`, `winner_rank`, `loser_rank` are mapped automatically.

## Quick start

1. **Data**: Use `data/matches.csv`. Options:
   - **Real ATP data**: Place yearly `atp_matches_YYYY.csv` files in `data/`, then run:
     ```bash
     python scripts/merge_atp_to_matches.py
     ```
     Merges all `atp_matches_*.csv` into a single `data/matches.csv` (schema aligned, hard/clay/grass only, rows with missing ranks/points dropped).

2. **Train models**:

   ```bash
   python run_train.py
   ```

   Trains Logistic Regression, Random Forest, and XGBoost with time-based split and tuning, saves the best model and metadata to `models/`.

   ```bash
   python scripts/build_feature_store.py
   ```

   Builds feature store for inference, saves `player_features.joblib`, `h2h_features.joblib` to `models/`.

3. **Run API**:

   ```bash
   python run_api.py
   ```

   Server: http://127.0.0.1:8000
   - Docs: http://127.0.0.1:8000/docs
   - `POST /predict` — body: `player1_name`, `player2_name`, `surface`, `tournament_level`, `explain`
   - `GET /players` — list of players from training data
   - `GET /model-info` — model name, feature list, metrics

## Project layout

```
tennis-ml-predictor/
├── config.py              # Paths, schema, constants
├── requirements.txt
├── run_train.py           # Training entry point
├── run_api.py             # API entry point
├── data/
│   └── matches.csv        # Your or sample CSV
├── models/                # Trained model and metadata (after training)
├── src/
│   ├── data_loader.py     # Load and validate CSV
│   ├── features.py       # Elo, player/surface/H2H, diff features
│   ├── train.py          # Train, tune, evaluate, persist
│   └── predictor.py      # Predict + importance
├── api/
│   └── main.py           # FastAPI app
└── scripts/
    ├── build_feature_store.py
    ├── merge_atp_to_matches.py
```

## API examples

**Predict:**

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"player1_name\": \"Novak Djokovic\", \"player2_name\": \"Rafael Nadal\", \"surface\": \"clay\", \"tournament_level\": \"Grand Slam\", \"explain\": \"true\"}"
```

**Response:** `predicted_winner`, `player1_win_probability`, `player2_win_probability`, `model_confidence`, `top_contributing_features`.

**Players:** `GET http://127.0.0.1:8000/players`  
**Model info:** `GET http://127.0.0.1:8000/model-info`

## License

MIT.
