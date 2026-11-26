"""Voice assistant router: speech text -> intent -> action -> reply (+ confirmation cards so the user can fix mistakes).

Hybrid pipeline:  your TF-IDF classifier first; Gemini only when its confidence is low.
                  your regex/unit extractor + Indian foods table first; Gemini only for foods it can't find."""
import re, json, base64, datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.core import db, rate_limit, current_user, CONF_THRESHOLD, calc_plan, has_body_profile, gemini_json
from app.nlp import intents as intent_model
from app.nlp.extractor import extract_items, extract_meal_time, extract_nutrient, extract_plan_target, _normalise
from app.nlp.foodmatch import get_index
from app.nlp import gemini_parser as gp
from app.nlp import preferences as prefs_mod
from app import tools

router = APIRouter(prefix="/api")
NUT_LABEL = {"calories": "kcal", "protein": "g protein", "carbs": "g carbs", "fat": "g fat", "fiber": "g fiber", "sugar": "g sugar", "sodium": "mg sodium"}


# ---------------------------------------------------------------- helpers
def _profile(user) -> dict:
    try: return json.loads(user.get("profile") or "{}")
    except ValueError: return {}

def _prefs(user) -> dict:
    return _profile(user).get("preferences", {})

def _save_prefs(uid: int, prefs: dict):
    with db() as c:
        row = c.execute("SELECT profile FROM users WHERE id=?", (uid,)).fetchone()
        prof = json.loads(row["profile"] or "{}"); prof["preferences"] = prefs
        c.execute("UPDATE users SET profile=? WHERE id=?", (json.dumps(prof), uid))

def _log(uid, text, intent, source, conf, parsed) -> int:
    with db() as c:
        cur = c.execute("INSERT INTO voice_logs(uid,transcript,intent,intent_source,confidence,parsed_json) VALUES(?,?,?,?,?,?)",
                        (uid, text, intent, source, round(conf or 0, 3), json.dumps(parsed, default=str)))
        return cur.lastrowid

def _gemini_ok(fn, *a, **k):
    """Run a Gemini helper; return (result, None) or (None, friendly_error). Keeps the voice flow alive when Gemini is down/rate-limited."""
    try: return fn(*a, **k), None
    except HTTPException as e: return None, str(e.detail)
    except gp.ParseError: return None, "I couldn't understand the AI answer. Please try again."

def _week_fallback(data: dict) -> str:
    a, n = data["average_per_logged_day"], data["days_logged"]
    if not n: return "You haven't logged any meals in the last 7 days yet. Log a few meals and I'll summarise your week."
    t = data["targets"]
    msg = f"You logged food on {n} of the last 7 days, averaging {a['calories']} kcal and {a['protein']} g protein per logged day."
    if t.get("protein") and a["protein"] < 0.8 * t["protein"]: msg += f" Your protein is below your {t['protein']} g target, so add dal, paneer, eggs or curd."
    return msg

def _plan_slot_match(name: str, slot: str) -> bool:
    n, s = name.lower(), slot.lower()
    return n == s or s in n or n in s or ("snack" in s and "snack" in n)

def _find_plan_meal(plan: dict, day: str, slot: str):
    for d in plan.get("days", []):
        if d.get("day", "").lower() == day.lower():
            for m in d.get("meals", []):
                if _plan_slot_match(m.get("name", ""), slot): return d, m
    return None, None


# ---------------------------------------------------------------- intent classification (hybrid)
def classify(text: str):
    """-> (intent, confidence, source, parsed_by_gemini|None).  source: 'classifier' | 'gemini' | 'classifier-lowconf'"""
    intent, conf = intent_model.predict(text)
    if intent and conf >= CONF_THRESHOLD: return intent, conf, "classifier", None
    parsed, err = _gemini_ok(gp.parse_with_gemini, text)
    if parsed: return parsed.intent, conf, "gemini", parsed
    if intent: return intent, conf, "classifier-lowconf", None      # Gemini unavailable: best guess beats failing
    raise HTTPException(503, err or "I couldn't understand that. Please try again.")


# ---------------------------------------------------------------- the main endpoint
class VoiceIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)

@router.post("/voice")
def voice(body: VoiceIn, u=Depends(rate_limit)):
    text = " ".join(body.text.split())
    intent, conf, source, parsed = classify(text)
    handler = HANDLERS.get(intent, h_chat)
    result = handler(text, u, parsed)
    result.setdefault("card", None)
    result.update(intent=intent, confidence=round(conf, 2), source=source, transcript=text)
    result["log_id"] = _log(u["id"], text, intent, source, conf, result.pop("_parsed", {"card": result["card"]}))
    return result


# ---------------------------------------------------------------- intent handlers
def h_log_meal(text, u, parsed):
    idx = get_index()
    meal, explicit = extract_meal_time(text)
    if not explicit and parsed and parsed.meal: meal, explicit = parsed.meal, True
    raw, how = extract_items(text), "rules"
    if not raw:
        if parsed is None:
            parsed, err = _gemini_ok(gp.parse_with_gemini, text)
            if parsed is None: return {"reply": "Sorry, I couldn't tell which food you ate. " + (err or "Could you say it again?")}
        raw, how = [i.model_dump() for i in parsed.items], "gemini"
    if not raw: return {"reply": "I didn't catch any food in that. Try: 'I had two rotis and dal for lunch'."}

    resolved, unknown = [], []
    for it in raw:
        r = idx.resolve(it)
        (resolved if r else unknown).append(r or it)
    note = ""
    if unknown:
        guessed, err = _gemini_ok(gp.nutrition_with_gemini, unknown)
        if guessed: resolved += guessed
        else: note = f" I couldn't find {', '.join(x['food'] for x in unknown)} in my food list" + (f" ({err})" if err else "") + "."
    if not resolved: return {"reply": "I couldn't find those foods." + note}

    warnings = prefs_mod.check_items(resolved, _prefs(u))
    kcal = round(sum(i["calories"] for i in resolved))
    names = ", ".join(f"{i['quantity']:g} {i['unit'] or ''} {i['name']}".replace("  ", " ") for i in resolved)
    reply = f"I heard {names} for {meal}: about {kcal} calories. Please confirm or edit." + note
    shaky = [i for i in resolved if i.get("uncertain")]
    if shaky: reply += " Not sure about " + ", ".join(f"'{i['heard_as']}' (I used {i['name']})" for i in shaky) + ", please check."
    if warnings: reply += " " + warnings[0]
    return {"reply": reply, "card": {"type": "meal", "meal": meal, "meal_guessed": not explicit, "items": resolved, "warnings": warnings, "total_kcal": kcal},
            "_parsed": {"extraction": how, "raw_items": raw, "resolved": resolved, "meal": meal}}


def h_today_summary(text, u, parsed):
    t, nut = tools.get_today_totals(u), extract_nutrient(text)
    if not t["items_logged"]: return {"reply": "You haven't logged anything today yet."}
    if nut:
        return {"reply": f"So far today you've had {round(t[nut])} {NUT_LABEL[nut]}.", "card": {"type": "totals", "totals": t}}
    return {"reply": f"Today you've had {round(t['calories'])} kcal, {round(t['protein'])} g protein, {round(t['carbs'])} g carbs and {round(t['fat'])} g fat "
                     f"across {t['items_logged']} items.", "card": {"type": "totals", "totals": t}}


def h_remaining(text, u, parsed):
    r, nut = tools.get_remaining_macros(u), extract_nutrient(text)
    nut = nut if nut in ("protein", "carbs", "fat") else "calories"
    left, known = r["remaining"][nut], r["targets"]["known"]
    assume = "" if known else " I'm assuming a 2000 kcal target; fill in BMI & Diet for an accurate one."
    if left is None: return {"reply": f"I need your profile to know your {nut} target. Open BMI & Diet and generate your plan first."}
    what = "kcal" if nut == "calories" else f"g of {nut}"
    if left >= 0: reply = f"You have {round(left)} {what} left today."
    else: reply = f"You're {round(-left)} {what} over your target today."
    return {"reply": reply + assume, "card": {"type": "remaining", "data": r}}


def h_suggest(text, u, parsed):
    meal, _ = extract_meal_time(text)
    prof = _profile(u); prefs = _prefs(u)
    res, err = _gemini_ok(gp.suggest_with_gemini, meal, tools.get_remaining_macros(u)["remaining"], prefs_mod.describe(prefs), prof.get("diet", "any"), text)
    if not res: return {"reply": "I couldn't fetch suggestions right now. " + (err or "")}
    return {"reply": res.get("reply") or "Here are a few ideas.", "card": {"type": "suggestions", "items": res.get("suggestions", [])[:4]}}


def h_edit_plan(text, u, parsed):
    day, slot = extract_plan_target(text)
    with db() as c: row = c.execute("SELECT plan FROM plans WHERE uid=?", (u["id"],)).fetchone()
    if not row: return {"reply": "You don't have a diet plan yet. Generate one in BMI & Diet first."}
    if not (day and slot): return {"reply": "Which day and meal should I change? For example: 'swap Wednesday's dinner for something vegetarian'."}
    plan = json.loads(row["plan"]); d, m = _find_plan_meal(plan, day, slot)
    if not m: return {"reply": f"I couldn't find {day} {slot.lower()} in your plan."}
    res, err = _gemini_ok(gp.plan_edit_with_gemini, day, m["name"], m, text, prefs_mod.describe(_prefs(u)), m.get("kcal"))
    if not res or not res.get("menu"): return {"reply": "I couldn't write the new meal right now. " + (err or "")}
    new = {"menu": str(res["menu"]), "kcal": float(res.get("kcal") or m.get("kcal") or 0)}
    return {"reply": f"{res.get('reply') or 'Here is the change.'} Confirm to update your plan.",
            "card": {"type": "plan_edit", "day": day, "slot": m["name"], "before": {"menu": m.get("menu"), "kcal": m.get("kcal")}, "after": new}}


def h_set_pref(text, u, parsed):
    found = prefs_mod.extract_preferences(text)
    if not (found["avoid"] or found["diet"]):
        raw, _ = _gemini_ok(gemini_json, 'Extract food restrictions from: ' + json.dumps(text) + '. Return ONLY JSON: {"avoid":[allowed tags: onion,garlic,peanut,nuts,dairy,gluten,egg,fish,nonveg,soy],"allergies":[subset of avoid],"diet":"vegetarian|vegan|jain|eggetarian|null"}')
        if isinstance(raw, dict):
            ok = set(prefs_mod.AVOID_KEYWORDS.values())
            found = {"avoid": [a for a in raw.get("avoid", []) if a in ok], "allergies": [a for a in raw.get("allergies", []) if a in ok],
                     "diet": raw.get("diet") if raw.get("diet") in prefs_mod.DIET_BLOCKS else None}
    if not (found["avoid"] or found["diet"]): return {"reply": "I couldn't find a food restriction in that. Try: 'I'm allergic to peanuts' or 'I don't eat onion'."}
    merged = prefs_mod.merge_preferences(_prefs(u), found); _save_prefs(u["id"], merged)
    bits = []
    if found["diet"]: bits.append(f"{found['diet']} diet")
    if found["allergies"]: bits.append("allergic to " + ", ".join(found["allergies"]))
    rest = [a for a in found["avoid"] if a not in found["allergies"]]
    if rest: bits.append("avoiding " + ", ".join(rest))
    return {"reply": "Saved: " + "; ".join(bits) + ". I'll use this in your plans and warn you when you log those foods.", "card": {"type": "preferences", "preferences": merged}}


def h_weekly(text, u, parsed):
    data = tools.get_week_totals(u)
    if not data["days_logged"]: return {"reply": _week_fallback(data)}
    reply, _ = _gemini_ok(gp.explain_with_gemini, text, data, prefs_mod.describe(_prefs(u)), "weekly")
    return {"reply": reply or _week_fallback(data), "card": {"type": "week", "data": data}}


COACH_CUES = re.compile(r"\b(my|me|mera|meri|mujhe|i|am|week|hafte|today|aaj|left|low|why)\b")
def h_chat(text, u, parsed):
    prefs = prefs_mod.describe(_prefs(u))
    if COACH_CUES.search(text.lower()):                         # coach chat with fixed tools
        chosen, _ = _gemini_ok(gp.pick_tools, text, list(tools.TOOLS))
        if chosen:
            data = {n: tools.TOOLS[n](u) for n in chosen}        # only whitelisted functions run
            reply, err = _gemini_ok(gp.explain_with_gemini, text, data, prefs, "coach")
            if reply: return {"reply": reply, "card": {"type": "coach", "tools": chosen}}
    reply, err = _gemini_ok(gp.chat_with_gemini, text, prefs)
    return {"reply": reply or ("Sorry, I can't reach the AI right now. " + (err or ""))}


HANDLERS = {"log_meal": h_log_meal, "today_summary": h_today_summary, "remaining_calories": h_remaining, "suggest_meal": h_suggest,
            "edit_plan": h_edit_plan, "set_preference": h_set_pref, "weekly_insight": h_weekly, "chat": h_chat}


# ---------------------------------------------------------------- audio -> text (high-accuracy mode, works in the mobile app)
ALLOWED_AUDIO = {"audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/aac", "audio/flac", "audio/webm", "audio/mp4"}

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...), u=Depends(rate_limit)):
    data = await file.read()
    if len(data) > 5_000_000: raise HTTPException(413, "Audio is too long. Keep it under about 30 seconds.")
    if not data: raise HTTPException(400, "No audio received.")
    mime = (file.content_type or "audio/wav").split(";")[0].strip().lower()
    if mime not in ALLOWED_AUDIO: raise HTTPException(415, f"Unsupported audio type {mime}")
    try: text = gp.transcribe_audio(base64.b64encode(data).decode(), mime)
    except gp.ParseError: raise HTTPException(422, "I couldn't hear anything. Please try again.")
    return {"text": text}


# ---------------------------------------------------------------- confirmations (the correction loop)
class MealItem(BaseModel):
    name: str; grams: float = 0; calories: float = 0; protein: float = 0; fat: float = 0
    carbs: float = 0; fiber: float = 0; sugar: float = 0; sodium_mg: float = 0

class ConfirmIn(BaseModel):
    log_id: Optional[int] = None
    meal: str
    items: list[MealItem]

@router.post("/meals/confirm")
def confirm_meal(b: ConfirmIn, u=Depends(current_user)):
    if not b.items: raise HTTPException(400, "Nothing to save")
    day, meal = str(datetime.date.today()), b.meal.strip().capitalize()
    with db() as c:
        for i in b.items:
            c.execute("INSERT INTO meals(uid,day,meal,food,grams,calories,protein,fat,carbs,fiber,sugar,sodium) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (u["id"], day, meal, i.name, i.grams, i.calories, i.protein, i.fat, i.carbs, i.fiber, i.sugar, i.sodium_mg))
        if b.log_id:                                              # what the user finally saved = ground truth for your NLP evaluation
            row = c.execute("SELECT parsed_json FROM voice_logs WHERE id=? AND uid=?", (b.log_id, u["id"])).fetchone()
            if row:
                orig = (json.loads(row["parsed_json"] or "{}").get("resolved") or [])
                edited = [(o.get("name"), round(o.get("grams", 0))) for o in orig] != [(i.name, round(i.grams)) for i in b.items]
                c.execute("UPDATE voice_logs SET corrected_json=? WHERE id=?",
                          (json.dumps({"meal": meal, "edited": edited, "items": [i.model_dump() for i in b.items]}), b.log_id))
    return {"ok": True, "saved": len(b.items)}


class PlanConfirm(BaseModel):
    log_id: Optional[int] = None
    day: str; slot: str; menu: str; kcal: float = 0

@router.post("/plan/confirm")
def confirm_plan(b: PlanConfirm, u=Depends(current_user)):
    with db() as c:
        row = c.execute("SELECT plan FROM plans WHERE uid=?", (u["id"],)).fetchone()
        if not row: raise HTTPException(404, "No plan found")
        plan = json.loads(row["plan"]); d, m = _find_plan_meal(plan, b.day, b.slot)
        if not m: raise HTTPException(404, "That meal is not in your plan")
        m["menu"], m["kcal"] = b.menu, b.kcal
        d["total_kcal"] = round(sum(float(x.get("kcal") or 0) for x in d["meals"]))
        c.execute("UPDATE plans SET plan=? WHERE uid=?", (json.dumps(plan), u["id"]))
        if b.log_id: c.execute("UPDATE voice_logs SET corrected_json=? WHERE id=? AND uid=?", (json.dumps(b.model_dump()), b.log_id, u["id"]))
    return plan


# ---------------------------------------------------------------- photo + voice correction ("that's actually two rotis")
CORRECTION_NOISE = {"actually", "sorry", "wait", "correction", "no", "not", "nahi", "nahin", "galat", "wrong", "its", "it", "was", "were", "that", "thats", "it's", "wo", "woh", "yeh", "ye", "tha"}

class CorrectIn(BaseModel):
    items: list[dict]
    text: str = Field(min_length=1, max_length=300)

def _shape_scan_item(r: dict) -> dict:
    r = dict(r); r.setdefault("category", "home"); r["confidence"] = r.get("confidence", 100)
    return r

@router.post("/scan/correct")
def correct_scan(b: CorrectIn, u=Depends(rate_limit)):
    idx, items = get_index(), [dict(i) for i in b.items]
    cleaned = " ".join(w for w in _normalise(b.text).split() if w not in CORRECTION_NOISE)
    changed = False
    for new in extract_items(cleaned):
        resolved = idx.resolve(new)
        if not resolved: continue
        for n, old in enumerate(items):
            of, _ = idx.match(old.get("name", ""))
            if of and of["name"] == resolved["name"]:
                items[n] = _shape_scan_item(resolved); changed = True; break
    if changed: return {"items": items, "source": "rules"}
    fixed, err = _gemini_ok(gp.correct_items_with_gemini, items, b.text)
    if not fixed: raise HTTPException(422, err or "I couldn't apply that correction. Try: 'it was two rotis'.")
    return {"items": [_shape_scan_item(i) for i in fixed], "source": "gemini"}


# ---------------------------------------------------------------- preferences, insights, stats
@router.get("/preferences")
def get_prefs(u=Depends(current_user)): return _prefs(u)

@router.delete("/preferences/{tag}")
def del_pref(tag: str, u=Depends(current_user)):
    merged = prefs_mod.remove_preference(_prefs(u), tag); _save_prefs(u["id"], merged); return merged

@router.get("/insights/weekly")
def weekly(u=Depends(rate_limit)):
    data = tools.get_week_totals(u)
    reply, _ = _gemini_ok(gp.explain_with_gemini, "Give me my weekly insight", data, prefs_mod.describe(_prefs(u)), "weekly") if data["days_logged"] else (None, None)
    return {"reply": reply or _week_fallback(data), "data": data}

@router.get("/voice/stats")
def voice_stats(u=Depends(current_user)):
    with db() as c:
        rows = c.execute("SELECT intent_source s, COUNT(*) n, SUM(corrected_json IS NOT NULL) confirmed FROM voice_logs WHERE uid=? GROUP BY intent_source", (u["id"],)).fetchall()
        edited = c.execute("SELECT COUNT(*) FROM voice_logs WHERE uid=? AND corrected_json LIKE '%\"edited\": true%'", (u["id"],)).fetchone()[0]
    return {"by_source": [dict(r) for r in rows], "meals_edited_by_user": edited}
