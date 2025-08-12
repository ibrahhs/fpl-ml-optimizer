import io
import pandas as pd
import numpy as np
import streamlit as st
from fpl_api import get_bootstrap_static, get_fixtures
from features import build_player_table, player_directory, horizon_expected_points
from model import train_predict
from optimizer import optimize_squad

st.set_page_config(page_title="FPL 25/26 Optimizer v3", page_icon="⚽", layout="wide")
st.title("FPL 25/26 Optimizer v3")
st.caption("Live FPL data. Multi-GW planning, transfers vs current team, captain/vice, chip heuristics.")

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Budget (FPL tenths)", min_value=800, max_value=1500, value=1000, step=5)
    gw = st.number_input("Start GW", min_value=1, max_value=38, value=1, step=1)
    horizon = st.number_input("Horizon (GWs)", min_value=1, max_value=6, value=3, step=1)
    starting_xi = st.checkbox("Optimize starting XI only", value=False)
    lock_ids_text = st.text_area("Lock element IDs (comma-separated)", help="Optional. Example: 1,2,3")
    run_btn = st.button("Run optimization")

@st.cache_data(show_spinner=False)
def load_data():
    return get_bootstrap_static(), get_fixtures()

def parse_locks(text):
    if not text.strip():
        return []
    try:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    except Exception:
        return []

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Optimizer (single GW)", "Multi-GW Planner", "Player Directory", "Transfers vs Current Team"])

with tab1:
    if run_btn:
        with st.spinner("Fetching data and building model..."):
            bootstrap, fixtures = load_data()
            df, X, y, features = build_player_table(bootstrap, fixtures, int(gw))
            model, preds = train_predict(X, y)
            df["exp_pts"] = preds

        with st.spinner("Optimizing..."):
            lock_ids = parse_locks(lock_ids_text)
            chosen, total = optimize_squad(df[["id","web_name","team_name","pos","cost","exp_pts"]], int(budget), starting_xi=starting_xi, lock_ids=lock_ids)

        st.subheader("Top Picks")
        st.dataframe(chosen.sort_values("exp_pts", ascending=False)[["web_name","pos","team_name","cost","exp_pts"]].reset_index(drop=True), use_container_width=True)
        captain = chosen.sort_values("exp_pts", ascending=False).head(1)
        vice = chosen.sort_values("exp_pts", ascending=False).iloc[1:2]
        if not captain.empty:
            st.success(f"Captain: {captain['web_name'].iloc[0]}  |  Vice-captain: {vice['web_name'].iloc[0] if not vice.empty else '—'}")
        st.metric("Total expected points", round(total, 2))

with tab2:
    bootstrap, fixtures = load_data()
    st.subheader("Projected expected points over horizon")
    agg, cols = horizon_expected_points(bootstrap, fixtures, int(gw), int(horizon), train_predict)
    st.caption(f"Columns: {', '.join(cols)} (discounted slightly by 0.95^k).")
    show = agg.sort_values("ep_sum", ascending=False)[["web_name","team_name","pos","cost","ep_sum"] + cols].reset_index(drop=True)
    st.dataframe(show, use_container_width=True, height=400)
    # Optimize on horizon sum
    players = agg.rename(columns={"ep_sum":"exp_pts"})[["id","web_name","team_name","pos","cost","ep_sum","exp_pts"]].copy()
    chosen, total = optimize_squad(players, int(budget), starting_xi=False, lock_ids=[])
    st.subheader("Horizon-optimized squad (sum of EP)")
    st.dataframe(chosen.sort_values("exp_pts", ascending=False)[["web_name","pos","team_name","cost","exp_pts"]])
    st.metric("Total horizon EP", round(total, 2))

with tab3:
    bootstrap, fixtures = load_data()
    df, X, y, _ = build_player_table(bootstrap, fixtures, int(gw))
    directory = df[["id","web_name","pos","team_name","cost","status","news"]].sort_values(["team_name","pos","web_name"]).reset_index(drop=True)
    st.subheader("All player IDs")
    st.dataframe(directory, use_container_width=True)
    buf = io.BytesIO(); directory.to_csv(buf, index=False)
    st.download_button("Download player IDs CSV", buf.getvalue(), file_name="player_ids.csv", mime="text/csv")

with tab4:
    st.subheader("Suggest transfers vs your current team")
    st.caption("Paste your current 15 element IDs (comma-separated). Optionally set free transfers = 1 or 2. Budget is your total current value in tenths (estimate).")
    ids_text = st.text_area("Your current squad element IDs (15)", height=80, placeholder="e.g., 1,2,3,...")
    ft = st.number_input("Free transfers", min_value=1, max_value=2, value=1, step=1)
    hit_cost = st.number_input("Hit cost per extra transfer", min_value=0, max_value=8, value=4, step=4)
    run_transfers = st.button("Compute transfer suggestions")
    if run_transfers and ids_text.strip():
        bootstrap, fixtures = load_data()
        df, X, y, _ = build_player_table(bootstrap, fixtures, int(gw))
        model, preds = train_predict(X, y)
        df["exp_pts"] = preds
        # Ideal target squad for this GW
        best, total = optimize_squad(df[["id","web_name","team_name","pos","cost","exp_pts"]], int(budget), starting_xi=False, lock_ids=[])
        current_ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
        current = df[df["id"].isin(current_ids)][["id","web_name","team_name","pos","cost","exp_pts"]].copy()
        # If fewer than 15 provided, proceed with what we have
        target_set = set(best["id"].tolist())
        curr_set = set(current["id"].tolist())
        outs = list(curr_set - target_set)
        ins  = [pid for pid in best["id"].tolist() if pid not in curr_set]

        # Limit to FT transfers
        recommendations = []
        # Pair best "out" with best "in" by marginal EP gain
        gains = []
        for out_id in outs:
            for in_id in ins:
                gain = float(df.loc[df["id"]==in_id,"exp_pts"].values[0]) - float(df.loc[df["id"]==out_id,"exp_pts"].values[0])
                gains.append((gain, out_id, in_id))
        gains.sort(reverse=True)
        k = min(ft, len(gains))
        picks = gains[:k]
        for gain, out_id, in_id in picks:
            out_row = df[df["id"]==out_id].iloc[0]
            in_row  = df[df["id"]==in_id].iloc[0]
            recommendations.append({
                "out_id": int(out_id), "out_name": out_row["web_name"], "out_team": out_row["team_name"],
                "in_id": int(in_id), "in_name": in_row["web_name"], "in_team": in_row["team_name"],
                "gain_ep": round(float(gain),2)
            })
        rec_df = pd.DataFrame(recommendations)
        if rec_df.empty:
            st.info("Your team already matches or is very close to the optimizer's target.")
        else:
            st.dataframe(rec_df)

        # Net expected gain accounting for hits
        net_gain = sum(r["gain_ep"] for r in recommendations) - (0 if ft>=len(recommendations) else (len(recommendations)-ft)*hit_cost)
        st.metric("Net expected gain (approx)", round(net_gain,2))
