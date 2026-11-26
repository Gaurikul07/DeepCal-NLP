"""Fixed, parameterised SQL tools. The AI may ask for these by name but can never write SQL itself."""
import datetime
from app.core import db, calc_plan, has_body_profile
import json

DEFAULT_TARGET = 2000
NUTRIENTS = {"calories": "calories", "protein": "protein", "carbs": "carbs", "fat": "fat", "fiber": "fiber", "sugar": "sugar", "sodium": "sodium"}


def _targets(user: dict) -> dict:
    prof = json.loads(user.get("profile") or "{}")
    if has_body_profile(prof):
        p = calc_plan(prof)
        return {"calories": p["target"], "protein": p["protein_g"], "carbs": p["carbs_g"], "fat": p["fat_g"], "fiber": 30, "known": True}
    return {"calories": DEFAULT_TARGET, "protein": None, "carbs": None, "fat": None, "fiber": 30, "known": False}


def get_today_totals(user: dict, day: str | None = None) -> dict:
    day = day or str(datetime.date.today())
    with db() as c:
        r = c.execute("SELECT COALESCE(SUM(calories),0) calories, COALESCE(SUM(protein),0) protein, COALESCE(SUM(carbs),0) carbs, "
                      "COALESCE(SUM(fat),0) fat, COALESCE(SUM(fiber),0) fiber, COALESCE(SUM(sugar),0) sugar, COALESCE(SUM(sodium),0) sodium, "
                      "COUNT(*) n FROM meals WHERE uid=? AND day=?", (user["id"], day)).fetchone()
        foods = [f"{m['food']} ({m['meal']})" for m in c.execute("SELECT meal, food FROM meals WHERE uid=? AND day=? ORDER BY id", (user["id"], day))]
    d = {k: round(r[k], 1) for k in r.keys() if k != "n"}; d["items_logged"] = r["n"]; d["foods"] = foods
    return d


def get_remaining_macros(user: dict) -> dict:
    t, tg = get_today_totals(user), _targets(user)
    rem = {k: (round(tg[k] - t[k], 1) if tg.get(k) else None) for k in ("calories", "protein", "carbs", "fat")}
    return {"remaining": rem, "targets": tg, "eaten": {k: t[k] for k in ("calories", "protein", "carbs", "fat")}}


def get_week_totals(user: dict) -> dict:
    today = datetime.date.today()
    days = [str(today - datetime.timedelta(days=i)) for i in range(6, -1, -1)]
    with db() as c:
        rows = {r["day"]: dict(r) for r in c.execute(
            "SELECT day, SUM(calories) calories, SUM(protein) protein, SUM(carbs) carbs, SUM(fat) fat, SUM(fiber) fiber, SUM(sugar) sugar "
            "FROM meals WHERE uid=? AND day>=? GROUP BY day", (user["id"], days[0]))}
    per_day = [{"day": d, **{k: round(rows.get(d, {}).get(k) or 0) for k in ("calories", "protein", "carbs", "fat", "fiber", "sugar")}} for d in days]
    logged = [x for x in per_day if x["calories"] > 0]
    avg = {k: round(sum(x[k] for x in logged) / len(logged)) if logged else 0 for k in ("calories", "protein", "carbs", "fat", "fiber", "sugar")}
    return {"per_day": per_day, "days_logged": len(logged), "average_per_logged_day": avg, "targets": _targets(user)}


TOOLS = {"get_today_totals": get_today_totals, "get_remaining_macros": get_remaining_macros, "get_week_totals": get_week_totals}
