import pandas as pd
from typing import List, Tuple
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, lpSum, PULP_CBC_CMD

SQUAD_SIZE = 15
POSITION_LIMITS = {"GK":2, "DEF":5, "MID":5, "FWD":3}
TEAM_LIMIT = 3

def optimize_squad(players: pd.DataFrame, budget: int, starting_xi: bool=False, lock_ids: List[int]=None) -> Tuple[pd.DataFrame, float]:
    df = players.copy()
    lock_ids = set(lock_ids or [])
    prob = LpProblem("FPL_Optimize", LpMaximize)
    x = {pid: LpVariable(f"x_{pid}", 0, 1, LpBinary) for pid in df["id"]}
    prob += lpSum(x[pid] * df.loc[df["id"]==pid, "exp_pts"].values[0] for pid in df["id"])
    prob += lpSum(x[pid] * int(df.loc[df["id"]==pid, "cost"].values[0]) for pid in df["id"]) <= budget
    if starting_xi:
        prob += lpSum(x.values()) == 11
    else:
        prob += lpSum(x.values()) == SQUAD_SIZE
    for pos, limit in POSITION_LIMITS.items():
        prob += lpSum(x[pid] for pid in df[df["pos"]==pos]["id"]) <= (limit if not starting_xi else 11)
    for t in df["team_name"].unique():
        prob += lpSum(x[pid] for pid in df[df["team_name"]==t]["id"]) <= TEAM_LIMIT
    if starting_xi:
        prob += lpSum(x[pid] for pid in df[df["pos"]=="GK"]["id"]) == 1
        prob += lpSum(x[pid] for pid in df[df["pos"]=="DEF"]["id"]) >= 3
        prob += lpSum(x[pid] for pid in df[df["pos"]=="MID"]["id"]) >= 2
        prob += lpSum(x[pid] for pid in df[df["pos"]=="FWD"]["id"]) >= 1
    for pid in lock_ids:
        if pid in x:
            prob += x[pid] == 1
    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    chosen = df[df["id"].map(lambda pid: x[pid].value() >= 0.99)].copy()
    total_points = float((chosen["exp_pts"]).sum())
    return chosen, total_points
