#!/usr/bin/env python3
"""
Pulls the latest USL data from American Soccer Analysis and rebuilds the
scouting dashboard (index.html). Run by GitHub Actions on a schedule, or by
hand:  python refresh.py
"""
import os, json, datetime, sys
import numpy as np
import pandas as pd
from itscalledsoccer.client import AmericanSoccerAnalysis

# ----- settings you might ever change -----
LEAGUES = ["uslc", "usl1"]                       # USL Championship + USL League One
SEASON  = os.environ.get("ASA_SEASON") or str(datetime.date.today().year)
LG_NAME = {"uslc": "USL Championship", "usl1": "USL League One"}
ACTIONS = ["Dribbling", "Fouling", "Interrupting", "Passing", "Receiving", "Shooting"]
# ------------------------------------------

notes = []
def note(m):
    print("•", m); notes.append(m)

print(f"Pulling American Soccer Analysis data — season {SEASON}, leagues {LEAGUES}")
client = AmericanSoccerAnalysis()

def get(method, leagues=None, **kw):
    try:
        df = getattr(client, method)(leagues=(leagues or LEAGUES), **kw)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception as e:
        note(f"{method} failed: {e}")
        return pd.DataFrame()

xg = get("get_player_xgoals",      season_name=SEASON, split_by_teams=True)
xp = get("get_player_xpass",       season_name=SEASON, split_by_teams=True)
ga = get("get_player_goals_added", season_name=SEASON, split_by_teams=True)
players = get("get_players")
teams   = get("get_teams")

if xg.empty:
    note("No xGoals rows returned — the season may not have started, or ASA hasn't published yet.")
    print("Nothing to build. Exiting without changing index.html.")
    sys.exit(0)

def col(df, name):
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df))

# ---- team metadata + which league each team is in ----
team_lg, team_full, team_abbr, team_lg_short = {}, {}, {}, {}
CONF_MAP = {}   # team abbr -> conference name (if ASA exposes one)
LG_SHORT_CODE = {"uslc": "USLC", "usl1": "USL1"}
for lg in LEAGUES:
    try:
        t = client.get_teams(leagues=lg)
        conf_col = next((c for c in t.columns if "conference" in str(c).lower()), None)
        if conf_col:
            note(f"get_teams({lg}) conference column: {conf_col}")
        for _, r in t.iterrows():
            team_lg[r["team_id"]]   = LG_NAME[lg]
            team_lg_short[r["team_id"]] = LG_SHORT_CODE.get(lg, lg.upper())
            team_full[r["team_id"]] = r.get("team_name")
            team_abbr[r["team_id"]] = r.get("team_abbreviation")
            if conf_col and pd.notna(r.get(conf_col)) and r.get("team_abbreviation"):
                CONF_MAP[r.get("team_abbreviation")] = str(r.get(conf_col))
    except Exception as e:
        note(f"get_teams({lg}) failed: {e}")
note(f"Conference info for {len(CONF_MAP)} teams." if CONF_MAP else "No conference column in ASA teams feed; table will be combined.")

# ---- player metadata ----
pmeta = {}
for _, r in players.iterrows():
    pid = r.get("player_id")
    ht = r.get("height")
    if (ht is None or (isinstance(ht, float) and np.isnan(ht))) and "height_ft" in players.columns:
        f, i = r.get("height_ft"), r.get("height_in")
        ht = f"{int(f)}' {int(i)}\"" if pd.notna(f) else None
    pmeta[pid] = {
        "player_name": r.get("player_name"),
        "birth_date": r.get("birth_date"),
        "nationality": r.get("nationality"),
        "height": ht,
        "weight_lb": r.get("weight_lb") if "weight_lb" in players.columns else r.get("weight"),
        "secondary_general_position": r.get("secondary_general_position"),
    }

# ---- base = xGoals; merge in xPass columns ----
df = xg.copy()
xp_cols = ["attempted_passes", "pass_completion_percentage", "xpass_completion_percentage",
           "passes_completed_over_expected", "passes_completed_over_expected_p100",
           "avg_distance_yds", "avg_vertical_distance_yds", "share_team_touches", "count_games"]
if not xp.empty:
    keep = ["player_id", "team_id"] + [c for c in xp_cols if c in xp.columns]
    df = df.merge(xp[keep], on=["player_id", "team_id"], how="left")

# ---- flatten goals-added (one nested list per player) ----
ga_map = {}
if not ga.empty and "data" in ga.columns:
    for _, r in ga.iterrows():
        key = (r["player_id"], r.get("team_id"))
        rec = {}
        for a in r["data"] if isinstance(r["data"], list) else []:
            k = a.get("action_type", "").lower()
            rec[f"ga_actions_{k}"]  = a.get("num_actions_for")
            rec[f"ga_raw_{k}"]      = a.get("goals_added_raw")
            rec[f"ga_aboveavg_{k}"] = a.get("goals_added_above_avg")
        ga_map[key] = rec
else:
    note("goals-added returned no 'data' column — g+ stats will be blank this run.")

def gaval(pid, tid, field):
    return ga_map.get((pid, tid), {}).get(field, np.nan)

def num(x):
    return float(x) if x is not None and pd.notna(x) else np.nan

def safe_div(a, b):
    a, b = num(a), num(b)
    return a / b if (pd.notna(a) and pd.notna(b) and b != 0) else np.nan

def per90(v, mins):
    v, mins = num(v), num(mins)
    return v / mins * 90 if (pd.notna(v) and pd.notna(mins) and mins > 0) else np.nan

rows = []
yr = int(SEASON)
for _, r in df.iterrows():
    pid, tid = r["player_id"], r.get("team_id")
    pm = pmeta.get(pid, {})
    mins = num(r.get("minutes_played"))
    # goals-added action values + totals
    gak = {}
    for a in ACTIONS:
        k = a.lower()
        gak[f"ga_actions_{k}"]  = gaval(pid, tid, f"ga_actions_{k}")
        gak[f"ga_raw_{k}"]      = gaval(pid, tid, f"ga_raw_{k}")
        gak[f"ga_aboveavg_{k}"] = gaval(pid, tid, f"ga_aboveavg_{k}")
    raws = [gak[f"ga_raw_{a.lower()}"] for a in ACTIONS]
    aavs = [gak[f"ga_aboveavg_{a.lower()}"] for a in ACTIONS]
    ga_raw_total = np.nansum(raws) if any(pd.notna(x) for x in raws) else np.nan
    ga_aa_total  = np.nansum(aavs) if any(pd.notna(x) for x in aavs) else np.nan

    bd = pm.get("birth_date")
    age = np.nan
    if isinstance(bd, str) and len(bd) >= 4 and bd[:4].isdigit():
        age = yr - int(bd[:4])

    goals = num(r.get("goals")); shots = num(r.get("shots"))
    ast = num(r.get("primary_assists"))

    rows.append({
        "player_name": pm.get("player_name"),
        "team_abbreviation": team_abbr.get(tid),
        "team_name": team_full.get(tid),
        "league_name": team_lg.get(tid),
        "general_position": r.get("general_position"),
        "secondary_general_position": pm.get("secondary_general_position"),
        "age_at_season": age,
        "nationality": pm.get("nationality"),
        "minutes_played": mins,
        "count_games": num(r.get("count_games")),
        "height": pm.get("height"),
        "weight_lb": num(pm.get("weight_lb")),
        # shooting
        "shots": shots, "shots_on_target": num(r.get("shots_on_target")),
        "goals": goals, "xgoals": num(r.get("xgoals")),
        "goals_minus_xgoals": num(r.get("goals_minus_xgoals")),
        "xplace": num(r.get("xplace")),
        "goals_per_shot": safe_div(goals, shots),
        "xgoals_per_shot": safe_div(r.get("xgoals"), shots),
        "shots_p90": per90(shots, mins), "shots_on_target_p90": per90(r.get("shots_on_target"), mins),
        "goals_p90": per90(goals, mins), "xgoals_p90": per90(r.get("xgoals"), mins),
        # creation
        "key_passes": num(r.get("key_passes")), "primary_assists": ast,
        "xassists": num(r.get("xassists")),
        "primary_assists_minus_xassists": num(r.get("primary_assists_minus_xassists")),
        "key_passes_p90": per90(r.get("key_passes"), mins),
        "primary_assists_p90": per90(ast, mins), "xassists_p90": per90(r.get("xassists"), mins),
        # production
        "goals_plus_primary_assists": num(r.get("goals_plus_primary_assists")),
        "xgoals_plus_xassists": num(r.get("xgoals_plus_xassists")),
        "goals_plus_assists_p90": per90(r.get("goals_plus_primary_assists"), mins),
        "xgoals_plus_xassists_p90": per90(r.get("xgoals_plus_xassists"), mins),
        # value
        "points_added": num(r.get("points_added")), "xpoints_added": num(r.get("xpoints_added")),
        # passing
        "attempted_passes": num(r.get("attempted_passes")),
        "pass_completion_percentage": num(r.get("pass_completion_percentage")),
        "xpass_completion_percentage": num(r.get("xpass_completion_percentage")),
        "passes_completed_over_expected": num(r.get("passes_completed_over_expected")),
        "passes_completed_over_expected_p100": num(r.get("passes_completed_over_expected_p100")),
        "avg_distance_yds": num(r.get("avg_distance_yds")),
        "avg_vertical_distance_yds": num(r.get("avg_vertical_distance_yds")),
        "share_team_touches": num(r.get("share_team_touches")),
        # goals added
        "ga_raw_total": ga_raw_total, "ga_aboveavg_total": ga_aa_total,
        "goals_added_raw_p90": per90(ga_raw_total, mins),
        **{k: num(v) for k, v in gak.items()},
    })

full = pd.DataFrame(rows)
note(f"Assembled {len(full)} player rows.")

# ---------- defensive counting stats from API-Football (graceful: optional) ----------
import re, unicodedata, os, time, json as _json
import urllib.request, urllib.error

def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()

DEF_KEYS = {"tkl", "intc", "blk", "tkl90", "intc90", "blk90", "tip90"}
DEF_BY_KEY = {}   # (norm_name, lg_short) -> {tkl,intc,blk,min}
DEF_OK = False

API_KEY = os.environ.get("APIFOOTBALL_KEY", "").strip()
API_BASE = "https://v3.football.api-sports.io"

def _api_get(path):
    req = urllib.request.Request(API_BASE + path, headers={"x-apisports-key": API_KEY})
    with urllib.request.urlopen(req, timeout=45) as r:
        return _json.loads(r.read().decode())

if not API_KEY:
    note("APIFOOTBALL_KEY secret not set — defensive stats skipped (everything else still builds).")
else:
    try:
        lj = _api_get("/leagues?country=USA&type=League")
        targets = {}  # league_id -> (lg_short, season_year)
        for item in lj.get("response", []):
            nm = (item.get("league", {}).get("name") or "").lower()
            seasons = item.get("seasons", []) or []
            cur = next((s for s in seasons if s.get("current")), seasons[-1] if seasons else None)
            if not cur:
                continue
            yr = cur.get("year")
            if "championship" in nm:
                targets[item["league"]["id"]] = ("USLC", yr)
            elif "league one" in nm or "league 1" in nm:
                targets[item["league"]["id"]] = ("USL1", yr)
        note(f"API-Football USL leagues: {sorted(targets.values())}")
        for lid, (lgshort, yr) in targets.items():
            page, total, pulled = 1, 1, 0
            while page <= total and page <= 45:
                try:
                    pj = _api_get(f"/players?league={lid}&season={yr}&page={page}")
                except urllib.error.HTTPError as he:
                    note(f"API-Football {lgshort} page {page} HTTP {he.code}; stopping league.")
                    break
                total = (pj.get("paging", {}) or {}).get("total", 1) or 1
                for pl in pj.get("response", []):
                    nm = (pl.get("player", {}) or {}).get("name")
                    stats = pl.get("statistics") or [{}]
                    st = stats[0] if stats else {}
                    tk = st.get("tackles", {}) or {}
                    gm = st.get("games", {}) or {}
                    DEF_BY_KEY[(_norm(nm), lgshort)] = {
                        "tkl": num(tk.get("total")), "intc": num(tk.get("interceptions")),
                        "blk": num(tk.get("blocks")), "min": num(gm.get("minutes"))}
                    pulled += 1
                page += 1
                time.sleep(7)  # respect free-tier rate limit (10 req/min)
            note(f"API-Football {lgshort}: {pulled} players (league {lid}, season {yr}).")
            DEF_OK = DEF_OK or pulled > 0
    except Exception as e:
        note(f"API-Football defensive fetch failed: {e}")

def def_value(key, drec):
    if not drec:
        return None
    mins = drec.get("min") or 0
    if key == "tkl":  return vi(drec.get("tkl"))
    if key == "intc": return vi(drec.get("intc"))
    if key == "blk":  return vi(drec.get("blk"))
    if mins and mins > 0:
        t = drec.get("tkl") or 0; i = drec.get("intc") or 0; b = drec.get("blk") or 0
        if key == "tkl90":  return vr(t / mins * 90)
        if key == "intc90": return vr(i / mins * 90)
        if key == "blk90":  return vr(b / mins * 90)
        if key == "tip90":  return vr((t + i) / mins * 90)
    return None


# ---------- build dashboard rows/cols (same layout as the original) ----------
INT, DEC, ONE, PCT, AGE = "int", "dec2", "dec2", "pct", "int"
cfg = [
 ("name","Player","id","id","str",None,False),("tm","Tm","id","id","str",None,False),
 ("team","Team","id","id","str",None,False),("lg","Lg","id","id","str",None,False),
 ("pos","Pos","id","id","str",None,False),("pos2","2nd","id","id","str",None,False),
 ("age","Age","id","id","int",None,False),("nat","Nat","id","id","str",None,False),
 ("min","Min","id","id","int",None,False),("games","Gms","id","id","int",None,False),
 ("ht","Ht","id","id","str",None,False),("wt","Wt","id","id","int",None,False),
 ("shots","Shots","Shooting","metric","int",True,False),("sot","SoT","Shooting","metric","int",True,False),
 ("goals","Goals","Shooting","metric","int",True,False),("xg","xG","Shooting","metric","dec2",True,False),
 ("gxg","G-xG","Shooting","metric","dec2",True,True),("gps","G/Sh","Shooting","metric","dec2",True,False),
 ("xgps","xG/Sh","Shooting","metric","dec2",True,False),("shp90","Shots/90","Shooting","metric","dec2",True,True),
 ("sotp90","SoT/90","Shooting","metric","dec2",True,False),("gp90","Goals/90","Shooting","metric","dec2",True,True),
 ("xgp90","xG/90","Shooting","metric","dec2",True,True),
 ("kp","KeyP","Creation","metric","int",True,False),("ast","Ast","Creation","metric","int",True,False),
 ("xa","xA","Creation","metric","dec2",True,False),("axa","A-xA","Creation","metric","dec2",True,False),
 ("kpp90","KeyP/90","Creation","metric","dec2",True,True),("astp90","Ast/90","Creation","metric","dec2",True,True),
 ("xap90","xA/90","Creation","metric","dec2",True,True),
 ("ga2","G+A","Production","metric","int",True,False),("xgxa","xG+xA","Production","metric","dec2",True,False),
 ("gap90","G+A/90","Production","metric","dec2",True,True),("xgxap90","xG+xA/90","Production","metric","dec2",True,False),
 ("pa","Pts Add","Value","metric","dec2",True,False),("xpa","xPts Add","Value","metric","dec2",True,False),
 ("patt","Pass Att","Passing","metric","int",True,False),("ppct","Pass%","Passing","metric","pct",True,True),
 ("xppct","xPass%","Passing","metric","pct",True,False),("poe","Pass +/-Exp","Passing","metric","dec2",True,False),
 ("poe100","Pass+/-/100","Passing","metric","dec2",True,True),("adist","Avg Dist","Passing","metric","dist",None,False),
 ("vdist","Vert Dist","Passing","metric","dist",None,False),("tsh","Touch%","Passing","metric","pct",None,True),
 ("graw","g+ Raw","Goals Added","metric","dec2",True,False),("gaa","g+ vsAvg","Goals Added","metric","dec2",True,True),
 ("gp90r","g+ /90","Goals Added","metric","dec2",True,True),
 ("gpd","g+ Drib","Goals Added","metric","dec2",True,False),("gpp","g+ Pass","Goals Added","metric","dec2",True,False),
 ("gpr","g+ Recv","Goals Added","metric","dec2",True,False),("gpsh","g+ Shoot","Goals Added","metric","dec2",True,False),
 ("gpi","g+ Intrpt","Goals Added","metric","dec2",True,False),("gpf","g+ Foul","Goals Added","metric","dec2",True,False),
 ("gpd_a","g+ Drib vsAvg","Goals Added","metric","dec2",True,False),("gpp_a","g+ Pass vsAvg","Goals Added","metric","dec2",True,True),
 ("gpr_a","g+ Recv vsAvg","Goals Added","metric","dec2",True,True),("gpsh_a","g+ Shoot vsAvg","Goals Added","metric","dec2",True,False),
 ("gpi_a","g+ Intrpt vsAvg","Goals Added","metric","dec2",True,True),("gpf_a","g+ Foul vsAvg","Goals Added","metric","dec2",True,False),
 # ---- defensive counting stats (API-Football) — keep in sync with defense.py DEF_CFG ----
 ("tkl","Tackles","Defending","metric","int",True,False),("intc","Interceptions","Defending","metric","int",True,False),
 ("blk","Blocks","Defending","metric","int",True,False),
 ("tkl90","Tackles/90","Defending","metric","dec2",True,True),("intc90","Interceptions/90","Defending","metric","dec2",True,True),
 ("blk90","Blocks/90","Defending","metric","dec2",True,True),("tip90","Tkl+Int/90","Defending","metric","dec2",True,True),
]
src = {
 "name":"player_name","tm":"team_abbreviation","team":"team_name","pos":"general_position",
 "pos2":"secondary_general_position","age":"age_at_season","nat":"nationality","min":"minutes_played",
 "games":"count_games","ht":"height","wt":"weight_lb",
 "shots":"shots","sot":"shots_on_target","goals":"goals","xg":"xgoals","gxg":"goals_minus_xgoals",
 "gps":"goals_per_shot","xgps":"xgoals_per_shot","shp90":"shots_p90","sotp90":"shots_on_target_p90",
 "gp90":"goals_p90","xgp90":"xgoals_p90","kp":"key_passes","ast":"primary_assists","xa":"xassists",
 "axa":"primary_assists_minus_xassists","kpp90":"key_passes_p90","astp90":"primary_assists_p90","xap90":"xassists_p90",
 "ga2":"goals_plus_primary_assists","xgxa":"xgoals_plus_xassists","gap90":"goals_plus_assists_p90","xgxap90":"xgoals_plus_xassists_p90",
 "pa":"points_added","xpa":"xpoints_added","patt":"attempted_passes","ppct":"pass_completion_percentage",
 "xppct":"xpass_completion_percentage","poe":"passes_completed_over_expected","poe100":"passes_completed_over_expected_p100",
 "adist":"avg_distance_yds","vdist":"avg_vertical_distance_yds","tsh":"share_team_touches",
 "graw":"ga_raw_total","gaa":"ga_aboveavg_total","gp90r":"goals_added_raw_p90",
 "gpd":"ga_raw_dribbling","gpp":"ga_raw_passing","gpr":"ga_raw_receiving","gpsh":"ga_raw_shooting","gpi":"ga_raw_interrupting","gpf":"ga_raw_fouling",
 "gpd_a":"ga_aboveavg_dribbling","gpp_a":"ga_aboveavg_passing","gpr_a":"ga_aboveavg_receiving","gpsh_a":"ga_aboveavg_shooting","gpi_a":"ga_aboveavg_interrupting","gpf_a":"ga_aboveavg_fouling",
}
LG_SHORT = {"USL Championship": "USLC", "USL League One": "USL1"}

def vi(x):  return None if pd.isna(x) else int(round(float(x)))
def vr(x):  return None if pd.isna(x) else round(float(x), 3)
def vs(x):  return None if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x)
fmt_fn = {"int": vi, "dec2": vr, "pct": vr, "dist": vr, "str": vs}

data_rows = []
def_matched = 0
for _, r in full.iterrows():
    lg_short = LG_SHORT.get(r["league_name"], r["league_name"])
    drec = DEF_BY_KEY.get((_norm(r["player_name"]), lg_short))
    if drec:
        def_matched += 1
    rec = []
    for key, label, grp, role, fmt, hib, piz in cfg:
        if key == "lg":
            rec.append(lg_short); continue
        if key in DEF_KEYS:
            rec.append(def_value(key, drec)); continue
        rec.append(fmt_fn[fmt](r[src[key]]))
    data_rows.append(rec)
note(f"Defensive stats matched to {def_matched}/{len(full)} players (Sofascore ok={DEF_OK}).")

cols_js = [{"key": k, "label": l, "group": g, "role": ro, "fmt": f, "hib": h, "pizza": p}
           for (k, l, g, ro, f, h, p) in cfg]

# ---------- MLS NEXT PRO (separate feeder-league pool — own page, not merged into USL) ----------
# Self-contained on purpose: if anything here fails, the main USL dashboard still builds fine.
MLSNP_ROWS = []
try:
    mnp_xg = get("get_player_xgoals", leagues=["mlsnp"], season_name=SEASON, split_by_teams=True)
    if mnp_xg.empty:
        note("MLS Next Pro: no xGoals rows returned (season may not have data yet).")
    else:
        mnp_xp = get("get_player_xpass", leagues=["mlsnp"], season_name=SEASON, split_by_teams=True)
        mnp_ga = get("get_player_goals_added", leagues=["mlsnp"], season_name=SEASON, split_by_teams=True)
        mnp_players = get("get_players", leagues=["mlsnp"])
        mnp_teams   = get("get_teams", leagues=["mlsnp"])

        mnp_team_full, mnp_team_abbr = {}, {}
        for _, r in mnp_teams.iterrows():
            mnp_team_full[r["team_id"]] = r.get("team_name")
            mnp_team_abbr[r["team_id"]] = r.get("team_abbreviation")

        mnp_pmeta = {}
        for _, r in mnp_players.iterrows():
            mnp_pmeta[r.get("player_id")] = {
                "player_name": r.get("player_name"),
                "birth_date": r.get("birth_date"),
                "secondary_general_position": r.get("secondary_general_position"),
            }

        mnp_df = mnp_xg.copy()
        if not mnp_xp.empty:
            keep = ["player_id", "team_id"] + [c for c in xp_cols if c in mnp_xp.columns]
            mnp_df = mnp_df.merge(mnp_xp[keep], on=["player_id", "team_id"], how="left")

        mnp_ga_map = {}
        if not mnp_ga.empty and "data" in mnp_ga.columns:
            for _, r in mnp_ga.iterrows():
                key = (r["player_id"], r.get("team_id"))
                rec = {}
                for a in r["data"] if isinstance(r["data"], list) else []:
                    k = a.get("action_type", "").lower()
                    rec[f"ga_raw_{k}"]      = a.get("goals_added_raw")
                    rec[f"ga_aboveavg_{k}"] = a.get("goals_added_above_avg")
                mnp_ga_map[key] = rec

        def mnp_gaval(pid, tid, field):
            return mnp_ga_map.get((pid, tid), {}).get(field, np.nan)

        for _, r in mnp_df.iterrows():
            pid, tid = r["player_id"], r.get("team_id")
            pm = mnp_pmeta.get(pid, {})
            mins = num(r.get("minutes_played"))
            gak = {}
            for a in ACTIONS:
                k = a.lower()
                gak[f"ga_raw_{k}"]      = mnp_gaval(pid, tid, f"ga_raw_{k}")
                gak[f"ga_aboveavg_{k}"] = mnp_gaval(pid, tid, f"ga_aboveavg_{k}")
            raws = [gak[f"ga_raw_{a.lower()}"] for a in ACTIONS]
            aavs = [gak[f"ga_aboveavg_{a.lower()}"] for a in ACTIONS]
            ga_raw_total = np.nansum(raws) if any(pd.notna(x) for x in raws) else np.nan
            ga_aa_total  = np.nansum(aavs) if any(pd.notna(x) for x in aavs) else np.nan

            bd = pm.get("birth_date")
            age = np.nan
            if isinstance(bd, str) and len(bd) >= 4 and bd[:4].isdigit():
                age = yr - int(bd[:4])

            goals = num(r.get("goals")); shots = num(r.get("shots"))
            ast = num(r.get("primary_assists"))

            mnp_src_row = {
                "player_name": pm.get("player_name"),
                "team_abbreviation": mnp_team_abbr.get(tid), "team_name": mnp_team_full.get(tid),
                "general_position": r.get("general_position"), "secondary_general_position": pm.get("secondary_general_position"),
                "age_at_season": age, "nationality": None,
                "minutes_played": mins, "count_games": num(r.get("count_games")), "height": None, "weight_lb": None,
                "shots": shots, "shots_on_target": num(r.get("shots_on_target")),
                "goals": goals, "xgoals": num(r.get("xgoals")), "goals_minus_xgoals": num(r.get("goals_minus_xgoals")),
                "goals_per_shot": safe_div(goals, shots), "xgoals_per_shot": safe_div(r.get("xgoals"), shots),
                "shots_p90": per90(shots, mins), "shots_on_target_p90": per90(r.get("shots_on_target"), mins),
                "goals_p90": per90(goals, mins), "xgoals_p90": per90(r.get("xgoals"), mins),
                "key_passes": num(r.get("key_passes")), "primary_assists": ast, "xassists": num(r.get("xassists")),
                "primary_assists_minus_xassists": num(r.get("primary_assists_minus_xassists")),
                "key_passes_p90": per90(r.get("key_passes"), mins),
                "primary_assists_p90": per90(ast, mins), "xassists_p90": per90(r.get("xassists"), mins),
                "goals_plus_primary_assists": num(r.get("goals_plus_primary_assists")),
                "xgoals_plus_xassists": num(r.get("xgoals_plus_xassists")),
                "goals_plus_assists_p90": per90(r.get("goals_plus_primary_assists"), mins),
                "xgoals_plus_xassists_p90": per90(r.get("xgoals_plus_xassists"), mins),
                "points_added": num(r.get("points_added")), "xpoints_added": num(r.get("xpoints_added")),
                "attempted_passes": num(r.get("attempted_passes")),
                "pass_completion_percentage": num(r.get("pass_completion_percentage")),
                "xpass_completion_percentage": num(r.get("xpass_completion_percentage")),
                "passes_completed_over_expected": num(r.get("passes_completed_over_expected")),
                "passes_completed_over_expected_p100": num(r.get("passes_completed_over_expected_p100")),
                "avg_distance_yds": num(r.get("avg_distance_yds")), "avg_vertical_distance_yds": num(r.get("avg_vertical_distance_yds")),
                "share_team_touches": num(r.get("share_team_touches")),
                "ga_raw_total": ga_raw_total, "ga_aboveavg_total": ga_aa_total,
                "goals_added_raw_p90": per90(ga_raw_total, mins),
                **{k: num(v) for k, v in gak.items()},
            }
            rec = []
            for key, label, grp, role, fmt, hib, piz in cfg:
                if key == "lg": rec.append("MLSNP"); continue
                if key in DEF_KEYS: rec.append(None); continue  # no defensive counting stats for MLSNP yet
                rec.append(fmt_fn[fmt](mnp_src_row.get(src[key])))
            MLSNP_ROWS.append(rec)
        note(f"MLS Next Pro: {len(MLSNP_ROWS)} player rows pulled (season {SEASON}).")
except Exception as e:
    note(f"MLS Next Pro pull failed (main USL dashboard is unaffected): {e}")

# ---- results & fixtures (games) ----
GAMES = []
# league per club from THIS season's rosters (authoritative; ASA team lists include historical members)
ABBR_LG = {}
for _, r in full.iterrows():
    ab = r.get("team_abbreviation")
    if ab and ab not in ABBR_LG:
        ABBR_LG[ab] = LG_SHORT.get(r["league_name"], r["league_name"])

games_raw = get("get_games", season_name=SEASON)
if not games_raw.empty:
    note("games columns: " + ", ".join(map(str, games_raw.columns)))
    def firstcol(cands):
        for c in cands:
            if c in games_raw.columns:
                return c
        return None
    c_home = firstcol(["home_team_id", "home_team", "homeTeamId"])
    c_away = firstcol(["away_team_id", "away_team", "awayTeamId"])
    c_date = firstcol(["date_time_utc", "datetime_utc", "date_time", "game_date", "date"])
    c_hs   = firstcol(["home_score", "home_goals", "home_score_total", "score_home"])
    c_as   = firstcol(["away_score", "away_goals", "away_score_total", "score_away"])
    note(f"games fields -> home:{c_home} away:{c_away} date:{c_date} score:{c_hs}/{c_as}")
    for _, g in games_raw.iterrows():
        hid = g.get(c_home) if c_home else None
        aid = g.get(c_away) if c_away else None
        h, a = team_abbr.get(hid), team_abbr.get(aid)
        d = str(g.get(c_date) or "")[:10] if c_date else ""
        if not (h and a and d):
            continue
        hs = g.get(c_hs) if c_hs else None
        as_ = g.get(c_as) if c_as else None
        final = (hs is not None and pd.notna(hs) and as_ is not None and pd.notna(as_))
        GAMES.append({"h": h, "a": a, "d": d,
                      "lg": ABBR_LG.get(h) or team_lg_short.get(hid),
                      "hs": int(hs) if final else None, "as": int(as_) if final else None})
    GAMES.sort(key=lambda r: r["d"])
    fin = [r for r in GAMES if r["hs"] is not None]           # full season — standings need every result
    up  = [r for r in GAMES if r["hs"] is None][:40]
    GAMES = fin + up
    note(f"Loaded {len(fin)} results (full season) + {len(up)} upcoming fixtures.")
else:
    note("get_games returned nothing — results section will be empty this run.")

template = open("template.html", encoding="utf-8").read()
out = (template
       .replace("__COLS__", json.dumps(cols_js, separators=(",", ":")))
       .replace("__DATA__", json.dumps(data_rows, separators=(",", ":")))
       .replace("__GAMES__", json.dumps(GAMES, separators=(",", ":")))
       .replace("__CONF__", json.dumps(CONF_MAP, separators=(",", ":")))
       .replace("__MLSNPDATA__", json.dumps(MLSNP_ROWS, separators=(",", ":")) if MLSNP_ROWS else "null")
       .replace("__GAMESSAMPLE__", "false")
       .replace("__DEFSAMPLE__", "false"))
open("index.html", "w", encoding="utf-8").write(out)

stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
open("build_report.txt", "w").write(
    f"Last rebuild: {stamp}\nSeason: {SEASON}\nPlayers: {len(full)}\nMLS Next Pro players: {len(MLSNP_ROWS)}\n" +
    ("\nNotes:\n- " + "\n- ".join(notes) if notes else "\nNo warnings."))
print(f"\nBuilt index.html — {len(full)} players, season {SEASON}, {stamp}")
