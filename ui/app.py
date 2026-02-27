"""
Streamlit UI for Tennis Match Prediction API.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests

# ── Page setup (must be first Streamlit call) ───────────────────────────────────
st.set_page_config(
    page_title="Tennis Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = st.sidebar.text_input("API Base URL", value="https://tennis-prediction-api.onrender.com").rstrip("/")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0f1e;
    color: #e8e8e8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1e2d40;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    letter-spacing: 0.08em;
    color: #ffffff;
    line-height: 1;
    margin: 0;
}
.hero h1 span { color: #34d399; }
.hero p {
    color: #6b7280;
    font-size: 1rem;
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
}

/* Court divider */
.court-line {
    height: 3px;
    background: linear-gradient(90deg, transparent, #34d399 30%, #34d399 70%, transparent);
    margin: 1.5rem auto 2rem;
    max-width: 500px;
    border-radius: 2px;
}

/* Card */
.card {
    background: #111827;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Probability bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.4rem 0;
}
.prob-label {
    width: 140px;
    font-size: 0.85rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #d1d5db;
}
.prob-bar-bg {
    flex: 1;
    background: #1e2d40;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.prob-pct {
    width: 44px;
    text-align: right;
    font-size: 0.85rem;
    font-weight: 700;
    color: #f9fafb;
}

/* Winner badge */
.winner-badge {
    display: inline-block;
    background: #064e3b;
    border: 1px solid #34d399;
    color: #34d399;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    padding: 0.35rem 1rem;
    border-radius: 6px;
}

/* Confidence pill */
.conf-pill {
    display: inline-block;
    background: #1e2d40;
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-left: 0.5rem;
}

/* Feature table */
.feat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #1e2d40;
    font-size: 0.83rem;
}
.feat-row:last-child { border-bottom: none; }
.feat-name { color: #9ca3af; }
.feat-val {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.pos { color: #34d399; }
.neg { color: #f87171; }

/* Status dot */
.status-ok { color: #34d399; }
.status-err { color: #f87171; }

/* Streamlit widget overrides */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select {
    background: #1a2535 !important;
    color: #e8e8e8 !important;
    border-color: #2d3f55 !important;
}
.stButton > button {
    background: #34d399 !important;
    color: #0a0f1e !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.15rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.06em;
    color: #f9fafb;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎾 MATCH <span>PREDICTOR</span></h1>
  <p>ATP Tennis · ML-Powered Win Probability</p>
</div>
<div class="court-line"></div>
""", unsafe_allow_html=True)

# ── Helper: call API ────────────────────────────────────────────────────────────
def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}")
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(path, payload):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

# ── Sidebar: status & players ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### API Status")
    health, err = api_get("/health")
    if health:
        loaded = health.get("model_loaded", False)
        st.markdown(
            f"<span class='status-ok'>● Connected</span>  &nbsp; Model: {'✅ Loaded' if loaded else '⚠️ Not loaded'}",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span class='status-err'>● Unreachable</span>", unsafe_allow_html=True)
        st.caption(err)

    st.markdown("---")
    st.markdown("### Model Info")
    info, _ = api_get("/model-info")
    if info and "model_name" in info:
        st.caption(f"**Model:** {info.get('model_name', 'N/A')}")
        metrics = info.get("metrics", {})
        if metrics:
            for k, v in metrics.items():
                st.caption(f"**{k}:** {v:.4f}" if isinstance(v, float) else f"**{k}:** {v}")
        st.caption(f"**Train samples:** {info.get('n_train', 'N/A')}")
        st.caption(f"**Test samples:** {info.get('n_test', 'N/A')}")
    elif info:
        st.caption(info.get("message", "No model metadata available."))

    st.markdown("---")
    st.markdown("### Available Players")
    players_data, _ = api_get("/players")
    players_list = players_data.get("players", []) if players_data else []
    if players_list:
        st.caption(f"{len(players_list)} players in database")
        search = st.text_input("Search players", placeholder="Type to filter…", label_visibility="collapsed")
        filtered = [p for p in players_list if search.lower() in p.lower()] if search else players_list[:50]
        for p in filtered[:60]:
            st.caption(p)
        if len(filtered) > 60:
            st.caption(f"… and {len(filtered) - 60} more")
    else:
        st.caption("No player list available.")

# ── Main: prediction form ───────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Player 1")
    p1 = st.text_input("Player 1 Name", placeholder="e.g. Novak Djokovic", key="p1", label_visibility="visible")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Player 2")
    p2 = st.text_input("Player 2 Name", placeholder="e.g. Rafael Nadal", key="p2", label_visibility="visible")
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns([1, 1, 1], gap="medium")

SURFACES = ["clay", "hard", "grass"]
LEVELS = ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "ATP Finals"]

with col3:
    surface = st.selectbox("🏟️ Surface", SURFACES, index=1)

with col4:
    level = st.selectbox("🏆 Tournament Level", LEVELS, index=3)

with col5:
    explain = st.checkbox("Show feature contributions", value=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("⚡  PREDICT WINNER")

# ── Result ──────────────────────────────────────────────────────────────────────
if predict_clicked:
    if not p1.strip() or not p2.strip():
        st.warning("Please enter both player names.")
    else:
        payload = {
            "player1_name": p1.strip(),
            "player2_name": p2.strip(),
            "surface": surface,
            "tournament_level": level,
            "explain": explain,
        }
        with st.spinner("Analysing match…"):
            result, err = api_post("/predict", payload)

        if err:
            st.error(f"Prediction failed: {err}")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            winner = result["predicted_winner"]
            p1_prob = result["player1_win_probability"]
            p2_prob = result["player2_win_probability"]
            confidence = result["model_confidence"]
            features = result.get("top_contributing_features", [])

            # Winner + confidence
            st.markdown(
                f"<div style='text-align:center; margin-bottom:1.5rem;'>"
                f"<div style='color:#6b7280; font-size:0.8rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem;'>Predicted Winner</div>"
                f"<span class='winner-badge'>🏆 {winner}</span>"
                f"<span class='conf-pill'>Confidence {confidence:.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Probability bars
            r1, r2 = st.columns(2, gap="large")

            def prob_bar(label, prob, color):
                pct = int(prob * 100)
                return f"""
                <div class="prob-row">
                  <div class="prob-label">{label}</div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
                  </div>
                  <div class="prob-pct">{pct}%</div>
                </div>"""

            with r1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    prob_bar(p1.strip()[:22], p1_prob, "#34d399") +
                    prob_bar(p2.strip()[:22], p2_prob, "#60a5fa"),
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Feature contributions
            with r2:
                if explain and features:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<h3 style='margin-top:0; font-size:1rem;'>TOP FEATURES</h3>", unsafe_allow_html=True)
                    rows_html = ""
                    for f in features:
                        name = str(f.get("feature", "")).replace("_", " ").title()
                        val = f.get("value", 0)
                        css = "pos" if val >= 0 else "neg"
                        sign = "+" if val >= 0 else ""
                        rows_html += f"""
                        <div class="feat-row">
                          <span class="feat-name">{name}</span>
                          <span class="feat-val {css}">{sign}{val:.4f}</span>
                        </div>"""
                    st.markdown(rows_html, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                elif explain:
                    st.info("No feature contribution data returned by the model.")
                else:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='color:#6b7280; font-size:0.85rem;'>Enable <b>Show feature contributions</b> above to see what drove this prediction.</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)