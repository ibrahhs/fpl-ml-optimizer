import io
import pandas as pd
import streamlit as st
from fpl_api import get_bootstrap_static, get_fixtures
from features import build_player_table
from model import train_predict
from optimizer import optimize_squad

st.set_page_config(page_title="FPL ML Optimizer", page_icon="⚽", layout="wide")

st.title("FPL ML Optimizer")
st.caption("Predict expected points with Gradient Boosting and optimize your squad with Integer Programming.")

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Budget (FPL tenths)", min_value=800, max_value=1500, value=1000, step=5)
    gw = st.number_input("Gameweek", min_value=1, max_value=38, value=1, step=1)
    starting_xi = st.checkbox("Optimize starting XI only", value=False)
    lock_ids_text = st.text_area("Lock element IDs (comma-separated)", help="Optional. Example: 1,2,3")
    run_btn = st.button("Run optimization")

@st.cache_data(show_spinner=False)
def load_data():
    bootstrap = get_bootstrap_static()
    fixtures = get_fixtures()
    return bootstrap, fixtures

def parse_locks(text):
    if not text.strip():
        return []
    try:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    except Exception:
        return []

if run_btn:
    with st.spinner("Fetching data and building model..."):
        bootstrap, fixtures = load_data()
        df, X, y, features = build_player_table(bootstrap, fixtures, int(gw))
        model, preds = train_predict(X, y)
        df["exp_pts"] = preds

    with st.spinner("Optimizing squad..."):
        lock_ids = parse_locks(lock_ids_text)
        chosen, total = optimize_squad(df[["id","web_name","team","pos","cost","exp_pts"]], int(budget), starting_xi=starting_xi, lock_ids=lock_ids)

    st.subheader("Top Picks")
    st.dataframe(chosen.sort_values("exp_pts", ascending=False).reset_index(drop=True))

    st.metric("Total expected points", round(total, 2))

    captain = chosen.sort_values("exp_pts", ascending=False).head(1)
    if not captain.empty:
        st.success(f"Captain pick: {captain['web_name'].iloc[0]}")

    # Downloads
    all_buf = io.BytesIO()
    chosen_buf = io.BytesIO()
    df.to_csv(all_buf, index=False)
    chosen.to_csv(chosen_buf, index=False)
    st.download_button("Download all players CSV", all_buf.getvalue(), file_name=f"players_gw{gw}.csv", mime="text/csv")
    st.download_button("Download chosen CSV", chosen_buf.getvalue(), file_name=f"chosen_gw{gw}.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: Lock your current team by element IDs to force include them.")
