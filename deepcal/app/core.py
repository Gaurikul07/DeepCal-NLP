"""Shared pieces: config, database, auth, Gemini client, calorie maths, rate limiting.
Kept separate from main.py so routers (voice.py) can import them without circular imports."""
import os, json, time, hmac, hashlib, sqlite3, collections
from pathlib import Path
import requests
from dotenv import load_dotenv
from fastapi import Header, HTTPException, Depends

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")
KEY = os.getenv("GEMINI_API_KEY", "")          # never hard-code the key; it lives in .env
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
SECRET = os.getenv("SECRET_KEY", "dev-secret-change-me").encode()
DB_PATH = Path(os.getenv("DB_PATH", BASE / "deepcal.db"))
CONF_THRESHOLD = float(os.getenv("INTENT_CONF_THRESHOLD", "0.7"))   # classifier vs Gemini switch
RATE_LIMIT_PER_MIN = int(os.getenv("VOICE_RATE_LIMIT", "20"))

# ---------------- DB ----------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw TEXT, salt TEXT, profile TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS meals(id INTEGER PRIMARY KEY, uid INTEGER, day TEXT, meal TEXT, food TEXT, grams REAL,
  calories REAL, protein REAL, fat REAL, carbs REAL, fiber REAL, sugar REAL, sodium REAL);
CREATE TABLE IF NOT EXISTS plans(uid INTEGER PRIMARY KEY, plan TEXT);
CREATE TABLE IF NOT EXISTS voice_logs(id INTEGER PRIMARY KEY, uid INTEGER, transcript TEXT, intent TEXT, intent_source TEXT,
  confidence REAL, parsed_json TEXT, corrected_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS foods(name TEXT PRIMARY KEY, aliases TEXT DEFAULT '', kcal REAL, protein REAL, fat REAL, carbs REAL,
  fiber REAL, sugar REAL, sodium_mg REAL, serving_g REAL DEFAULT 100, tags TEXT DEFAULT '', source TEXT DEFAULT 'seed');
CREATE TABLE IF NOT EXISTS units(unit TEXT, food TEXT, grams REAL, PRIMARY KEY(unit, food));
"""

def db():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c

def init_db():
    """Create tables and (re)seed the Indian foods + units tables from data/*.csv.
    INSERT OR IGNORE means rows you edit/add in SQLite are never overwritten."""
    import csv
    with db() as c:
        for r in csv.DictReader(open(BASE / "data" / "foods.csv", encoding="utf-8")):
            c.execute("INSERT OR IGNORE INTO foods(name,aliases,kcal,protein,fat,carbs,fiber,sugar,sodium_mg,serving_g,tags) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (r["name"], r["aliases"], r["kcal"], r["protein"], r["fat"], r["carbs"], r["fiber"], r["sugar"], r["sodium_mg"], r["serving_g"], r["tags"]))
        for r in csv.DictReader(open(BASE / "data" / "units.csv", encoding="utf-8")):
            c.execute("INSERT OR IGNORE INTO units(unit,food,grams) VALUES(?,?,?)", (r["unit"], r["food"], r["grams"]))

# ---------------- Auth (PBKDF2 + signed token) ----------------
def hpw(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex()
def make_token(uid):
    body = f"{uid}.{int(time.time()) + 7*86400}"
    return body + "." + hmac.new(SECRET, body.encode(), "sha256").hexdigest()
def current_user(authorization: str = Header(None)):
    try:
        uid, exp, sig = (authorization or "").replace("Bearer ", "").split(".")
        ok = hmac.compare_digest(sig, hmac.new(SECRET, f"{uid}.{exp}".encode(), "sha256").hexdigest())
        if not ok or int(exp) < time.time(): raise ValueError
    except Exception:
        raise HTTPException(401, "Please login again")
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE id=?", (int(uid),)).fetchone()
    if not u: raise HTTPException(401, "User not found")
    return dict(u)

# ---------------- Rate limiting (in-memory sliding window, per user) ----------------
_hits = collections.defaultdict(collections.deque)
def rate_limit(u=Depends(current_user)):
    q, now = _hits[u["id"]], time.time()
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, f"Too many voice requests. Max {RATE_LIMIT_PER_MIN} per minute, please wait a moment.")
    q.append(now)
    return u

# ---------------- Gemini ----------------
CANDIDATE_MODELS = [MODEL, "gemini-3-flash-preview", "gemini-3.8-flash", "gemini-flash-latest"]

def gemini(parts, schema_hint=""):
    """Call Gemini and return parsed JSON. Resilient across fallback models. Raises HTTPException with a friendly message on failure."""
    if not KEY: raise HTTPException(400, "GEMINI_API_KEY missing. Add it to the .env file and restart the server.")
    body = {"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    last_err = None
    seen = set()
    models_to_try = [m for m in CANDIDATE_MODELS if m and not (m in seen or seen.add(m))]
    for m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": KEY}, json=body, timeout=45)
            if r.status_code == 200:
                try:
                    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(txt.strip().removeprefix("```json").removesuffix("```"))
                except Exception:
                    raise HTTPException(502, "Could not read the AI response, please try again")
            if r.status_code == 429:
                last_err = HTTPException(429, "Gemini rate limit reached. Wait a few seconds.")
                continue
            last_err = HTTPException(502, f"Gemini error {r.status_code}: {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            last_err = HTTPException(502, f"Gemini connection failed: {e}")
    raise last_err or HTTPException(502, "Could not reach the AI service.")

def gemini_json(prompt: str):
    return gemini([{"text": prompt}])

# ---------------- Calorie maths ----------------
def calc_plan(p):
    w, h, a = float(p["weight"]), float(p["height"]), int(p["age"])
    bmi = w / ((h/100)**2)
    cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    bmr = 10*w + 6.25*h - 5*a + (5 if p.get("sex", "male") == "male" else -161)
    tdee = bmr * float(p.get("activity", 1.4))
    target = tdee + {"lose": -500, "gain": 400}.get(p.get("goal", "maintain"), 0)
    target = max(target, 1200)
    prot = w * (1.8 if p.get("goal") != "maintain" else 1.4)
    return dict(bmi=round(bmi, 1), category=cat, bmr=round(bmr), tdee=round(tdee), target=round(target), protein_g=round(prot),
                fat_g=round(target*.27/9), carbs_g=round((target - prot*4 - target*.27)/4), water_l=round(w*0.035, 1),
                ideal_weight=f"{round(18.5*(h/100)**2)}-{round(24.9*(h/100)**2)} kg")

def has_body_profile(prof: dict) -> bool:
    return bool(prof) and all(prof.get(k) for k in ("age", "weight", "height"))
