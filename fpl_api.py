import requests
BASE = "https://fantasy.premierleague.com/api"

def get_bootstrap_static():
    r = requests.get(f"{BASE}/bootstrap-static/", timeout=30)
    r.raise_for_status()
    return r.json()

def get_fixtures():
    r = requests.get(f"{BASE}/fixtures/", timeout=30)
    r.raise_for_status()
    return r.json()
