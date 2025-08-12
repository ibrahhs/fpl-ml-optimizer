import pandas as pd
import numpy as np

POSITION_MAP = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

def build_player_table(bootstrap, fixtures, gw: int):
    elems = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id","name","short_name"]].rename(columns={"id":"team_id","name":"team_name"})
    elems = elems.merge(teams, left_on="team", right_on="team_id", how="left")

    fx = pd.DataFrame(fixtures)
    fx_gw = fx[fx["event"] == gw].copy()

    opp_strength = {}
    for _, row in fx_gw.iterrows():
        h = row["team_h"]; a = row["team_a"]
        opp_strength[(h, "H")] = row.get("team_a_difficulty", 3)
        opp_strength[(a, "A")] = row.get("team_h_difficulty", 3)

    df = elems[[
        "id","web_name","first_name","second_name",
        "now_cost","element_type","team","team_name","short_name",
        "selected_by_percent","form","points_per_game","minutes","ict_index","influence","creativity","threat",
        "goals_scored","assists","clean_sheets","total_points","chance_of_playing_next_round",
        "news","status"
    ]].copy()

    df["pos"] = df["element_type"].map(POSITION_MAP)
    df["cost"] = df["now_cost"].astype(int)
    df["sel_pct"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df["ppg"]  = pd.to_numeric(df["points_per_game"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")
    df["influence"] = pd.to_numeric(df["influence"], errors="coerce")
    df["creativity"] = pd.to_numeric(df["creativity"], errors="coerce")
    df["threat"] = pd.to_numeric(df["threat"], errors="coerce")
    df["goals_scored"] = pd.to_numeric(df["goals_scored"], errors="coerce")
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce")
    df["clean_sheets"] = pd.to_numeric(df["clean_sheets"], errors="coerce")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df["x_play"] = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(95)/100.0

    def opp_diff(row):
        # average GW difficulty for the player's team if exists
        vals = [v for (t,_), v in opp_strength.items() if t==row["team"]]
        return float(np.mean(vals)) if vals else 3.0
    df["opp_difficulty"] = df.apply(opp_diff, axis=1)

    df["value_ppg"] = df["ppg"] / df["cost"].replace(0, np.nan)
    df["form_scaled"] = df["form"] * df["x_play"]
    df["attacking"] = df["goals_scored"]*4 + df["assists"]*3 + df["threat"]/10
    df["creative"] = df["creativity"]/10 + df["assists"]*2
    df["defensive"] = df["clean_sheets"]*4 + df["influence"]/10
    df["fixture_adj"] = (4 - df["opp_difficulty"])

    df["y_proxy"] = (0.45*df["ppg"] + 0.55*df["form"]) * (0.7 + 0.3*df["fixture_adj"]/3.0) * df["x_play"]

    features = ["cost","sel_pct","form","ppg","minutes","ict_index","attacking","creative","defensive","fixture_adj","value_ppg","form_scaled"]
    X = df[features].fillna(0.0)
    y = df["y_proxy"].fillna(df["ppg"].fillna(0.0))
    return df, X, y, features

def player_directory(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["id","web_name","first_name","second_name","pos","team_name","short_name","cost","sel_pct","status","news"]
    out = df[cols].sort_values(["team_name","pos","web_name"]).reset_index(drop=True)
    return out

def horizon_expected_points(bootstrap, fixtures, start_gw: int, horizon: int, model_fn):
    """Project expected points for each player over multiple GWs using the same modeling pipeline per GW,
    then return a DataFrame with aggregated EP over horizon (simple sum with 0.9 decay)."""
    agg = None
    for k in range(horizon):
        gw = start_gw + k
        df, X, y, _ = build_player_table(bootstrap, fixtures, gw)
        model, preds = model_fn(X, y)
        df_k = df[["id","web_name","team_name","pos","cost"]].copy()
        df_k[f"gw{gw}_ep"] = preds * (0.95**k)  # slight decay
        if agg is None:
            agg = df_k
        else:
            agg = agg.merge(df_k, on=["id","web_name","team_name","pos","cost"], how="outer")
    ep_cols = [c for c in agg.columns if c.startswith("gw")]
    agg["ep_sum"] = agg[ep_cols].fillna(0).sum(axis=1)
    return agg, ep_cols
