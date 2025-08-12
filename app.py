import io, json
import pandas as pd
import numpy as np
import streamlit as st
from fpl_api import get_bootstrap_static, get_fixtures
from features import build_player_table, player_directory, horizon_expected_points
from model import train_predict
from optimizer import optimize_squad

st.set_page_config(page_title="FPL 25/26 Optimizer v4", page_icon="⚽", layout="wide")
st.title("FPL 25/26 Optimizer v4")
st.caption("Live FPL data. Simple controls. Multi-GW planner, captain/vice, player IDs, and transfer suggestions.")

with st.sidebar:
    st.header("Quick Settings")
    budget = st.number_input("Budget (FPL tenths)", min_value=800, max_value=1500, value=1000, step=5)
    start_gw = st.number_input("Start GW", min_value=1, max_value=38, value=1, step=1)
    horizon = st.number_input("Horizon (GWs)", min_value=1, max_value=6, value=3, step=1)
    starting_xi = st.checkbox("Optimize starting XI only", value=False)
    lock_ids_text = st.text_area("Lock element IDs (comma-separated)", help="Optional.")
    run_btn = st.button("Run")

@st.cache_data(show_spinner=False)
def load_data():
    return get_bootstrap_static(), get_fixtures()

def parse_ids(text):
    if not text or not text.strip():
        return []
    out = []
    for x in text.replace("\n",",").split(","):
        x = x.strip()
        if not x: 
            continue
        try:
            out.append(int(x))
        except:
            pass
    return out

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Optimizer", "Multi-GW Planner", "Player IDs", "Transfers vs Team"])

with tab1:
    st.subheader("Single GW Optimizer")
    if run_btn:
        bootstrap, fixtures = load_data()
        df, X, y, features = build_player_table(bootstrap, fixtures, int(start_gw))
        model, preds = train_predict(X, y)
        df["exp_pts"] = preds
        lock_ids = parse_ids(lock_ids_text)
        chosen, total = optimize_squad(df[["id","web_name","team_name","pos","cost","exp_pts"]], int(budget), starting_xi=starting_xi, lock_ids=lock_ids)

        st.dataframe(chosen.sort_values("exp_pts", ascending=False)[["web_name","pos","team_name","cost","exp_pts"]].reset_index(drop=True), use_container_width=True)
        captain = chosen.sort_values("exp_pts", ascending=False).head(1)
        vice = chosen.sort_values("exp_pts", ascending=False).iloc[1:2]
        if not captain.empty:
            st.success(f"Captain: {captain['web_name'].iloc[0]}  |  Vice: {vice['web_name'].iloc[0] if not vice.empty else '—'}")
        st.metric("Total expected points", round(total,2))

        # Downloads
        all_buf = io.BytesIO(); chosen_buf = io.BytesIO()
        df.to_csv(all_buf, index=False); chosen.to_csv(chosen_buf, index=False)
        st.download_button("Download all players CSV", all_buf.getvalue(), file_name=f"players_gw{start_gw}.csv", mime="text/csv")
        st.download_button("Download chosen CSV", chosen_buf.getvalue(), file_name=f"chosen_gw{start_gw}.csv", mime="text/csv")

with tab2:
    st.subheader("Multi-GW Planner")
    if run_btn:
        bootstrap, fixtures = load_data()
        agg, cols = horizon_expected_points(bootstrap, fixtures, int(start_gw), int(horizon), train_predict)
        if agg.empty:
            st.warning("No data returned for the selected horizon.")
        else:
            show = agg.sort_values("ep_sum", ascending=False)[["web_name","team_name","pos","cost","ep_sum"] + cols].reset_index(drop=True)
            st.dataframe(show, use_container_width=True, height=400)

            # Optimize on horizon sum
            players = agg.rename(columns={"ep_sum": "exp_pts"})[["id","web_name","team_name","pos","cost","exp_pts"]].copy()
            chosen, total = optimize_squad(players, int(budget), starting_xi=False, lock_ids=[])
            st.caption("Optimized on the sum of projected points across the horizon.")
            st.dataframe(chosen.sort_values("exp_pts", ascending=False)[["web_name","pos","team_name","cost","exp_pts"]], use_container_width=True)
            st.metric("Total horizon EP", round(total,2))

with tab3:
    st.subheader("Player IDs")
    bootstrap, fixtures = load_data()
    df, X, y, _ = build_player_table(bootstrap, fixtures, int(start_gw))
    directory = df[["id","web_name","pos","team_name","cost","status","news"]].sort_values(["team_name","pos","web_name"]).reset_index(drop=True)
    st.dataframe(directory, use_container_width=True)
    buf = io.BytesIO(); directory.to_csv(buf, index=False)
    st.download_button("Download player IDs CSV", buf.getvalue(), file_name="player_ids.csv", mime="text/csv")

with tab4:
    st.subheader("Transfers vs your team (quick)")
    st.caption("Paste your 15 element IDs below. Set free transfers (1 or 2). The tool suggests best swaps by marginal EP gain and estimates net gain after hit cost.")
    ids_text = st.text_area("Your current 15 IDs", height=80, placeholder="e.g., 1,2,3,... (15 numbers)")
    ft = st.number_input("Free transfers", min_value=1, max_value=2, value=1, step=1)
    hit_cost = st.number_input("Hit cost per extra transfer", min_value=0, max_value=8, value=4, step=4)
    run_transfers = st.button("Suggest transfers")
    if run_transfers:
        current_ids = parse_ids(ids_text)
        if not current_ids:
            st.error("Please paste your element IDs.")
        else:
            bootstrap, fixtures = load_data()
            df, X, y, _ = build_player_table(bootstrap, fixtures, int(start_gw))
            model, preds = train_predict(X, y)
            df["exp_pts"] = preds

            # Ideal target squad for this GW
            best, total = optimize_squad(df[["id","web_name","team_name","pos","cost","exp_pts"]], int(budget), starting_xi=False, lock_ids=[])
            curr = df[df["id"].isin(current_ids)][["id","web_name","team_name","pos","cost","exp_pts"]].copy()

            target_set = set(best["id"])
            curr_set = set(curr["id"])

            outs = list(curr_set - target_set)
            ins = [pid for pid in best["id"].tolist() if pid not in curr_set]

            # Greedy top-k by marginal EP gain, same-position preference
            cand = []
            for out_id in outs:
                out_row = df[df["id"]==out_id].iloc[0]
                for in_id in ins:
                    in_row = df[df["id"]==in_id].iloc[0]
                    if in_row["pos"] != out_row["pos"]:
                        continue
                    gain = float(in_row["exp_pts"] - out_row["exp_pts"])
                    cand.append((gain, out_id, in_id))
            cand.sort(reverse=True)
            k = min(int(ft), len(cand))
            picks = cand[:k]
            rows = []
            for gain, out_id, in_id in picks:
                out_row = df[df["id"]==out_id].iloc[0]
                in_row  = df[df["id"]==in_id].iloc[0]
                rows.append({"out_id": int(out_id), "out": f"{out_row['web_name']} ({out_row['team_name']})",
                            "in_id": int(in_id), "in": f"{in_row['web_name']} ({in_row['team_name']})",
                            "gain_ep": round(float(gain),2)})
            rec_df = pd.DataFrame(rows)
            if rec_df.empty:
                st.info("Your team is already close to the optimized squad for this GW.")
            else:
                st.dataframe(rec_df, use_container_width=True)
                net_gain = sum(r["gain_ep"] for r in rows) - (0 if ft>=len(rows) else (len(rows)-ft)*hit_cost)
                st.metric("Net expected gain (approx)", round(net_gain,2))

st.markdown("---")
st.caption("Data: official public FPL endpoints. Budget is in tenths (1000 = £100.0).")
