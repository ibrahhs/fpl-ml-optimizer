import pandas as pd
import numpy as np

POSITION_MAP = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

def build_player_table(bootstrap, fixtures, gw: int):
    elems = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id","name","strength_overall_home","strength_overall_away"]]
    teams = teams.rename(columns={"id":"team_id"})
    elems = elems.merge(teams, left_on="team", right_on="team_id", how="left")

    fx = pd.DataFrame(fixtures)
    fx = fx[fx["event"] == gw]
    opp_strength = {}
    for _, row in fx.iterrows():
        h = row["team_h"]
        a = row["team_a"]
        opp_strength[(h, "H")] = row.get("team_a_difficulty", 3)
        opp_strength[(a, "A")] = row.get("team_h_difficulty", 3)

    df = elems[[
        "id","web_name","now_cost","element_type","team","selected_by_percent",
        "form","points_per_game","minutes","ict_index","influence","creativity","threat",
        "goals_scored","assists","clean_sheets","total_points","chance_of_playing_next_round",
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
        keyH = (row["team"], "H")
        keyA = (row["team"], "A")
        return float(opp_strength.get(keyH, opp_strength.get(keyA, 3)))
    df["opp_difficulty"] = df.apply(opp_diff, axis=1)

    df["value_ppg"] = df["ppg"] / df["cost"].replace(0, np.nan)
    df["form_scaled"] = df["form"] * df["x_play"]
    df["attacking"] = df["goals_scored"]*4 + df["assists"]*3 + df["threat"]/10
    df["creative"] = df["creativity"]/10 + df["assists"]*2
    df["defensive"] = df["clean_sheets"]*4 + df["influence"]/10
    df["fixture_adj"] = (4 - df["opp_difficulty"])

    df["y_proxy"] = (0.5*df["ppg"] + 0.5*df["form"]) * (0.7 + 0.3*df["fixture_adj"]/3.0) * df["x_play"]

    features = ["cost","sel_pct","form","ppg","minutes","ict_index","attacking","creative","defensive","fixture_adj","value_ppg","form_scaled"]
    X = df[features].fillna(0.0)
    y = df["y_proxy"].fillna(df["ppg"].fillna(0.0))
    return df, X, y, features
