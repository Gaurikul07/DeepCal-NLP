import json, hmac, base64, sqlite3, secrets, datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core import (BASE, KEY, db, init_db, hpw, make_token, current_user, gemini, calc_plan, has_body_profile)
from app.nlp import preferences as prefs_mod
from app.voice import router as voice_router

app = FastAPI(title="DeepCal AI")
app.include_router(voice_router)

@app.on_event("startup")
def _startup():
    init_db()          # creates tables (incl. voice_logs, foods, units) and seeds the Indian food data

class Reg(BaseModel): name: str; email: str; password: str
class Login(BaseModel): email: str; password: str

@app.post("/api/register")
def register(r: Reg):
    if len(r.password) < 6: raise HTTPException(400, "Password must be at least 6 characters")
    salt = secrets.token_hex(8)
    try:
        with db() as c:
            cur = c.execute("INSERT INTO users(name,email,pw,salt) VALUES(?,?,?,?)", (r.name.strip(), r.email.lower().strip(), hpw(r.password, salt), salt))
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")
    return {"token": make_token(cur.lastrowid), "name": r.name}

@app.post("/api/login")
def login(r: Login):
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE email=?", (r.email.lower().strip(),)).fetchone()
    if not u or not hmac.compare_digest(u["pw"], hpw(r.password, u["salt"])):
        raise HTTPException(401, "Wrong email or password")
    return {"token": make_token(u["id"]), "name": u["name"]}

@app.get("/api/me")
def me(u=Depends(current_user)):
    return {"name": u["name"], "email": u["email"], "profile": json.loads(u["profile"]), "gemini": bool(KEY)}

# ---------------- Gemini food-photo analysis ----------------
FOOD_PROMPT = """You are an expert nutritionist and food-recognition system. Look at the image and identify EVERY food or drink item visible
(home-cooked, restaurant, street food, Indian regional dishes like dosa/idli/samosa/biryani/chapati/dal/paneer, fast food, fruits, vegetables,
sweets, beverages, and packaged/branded products). For packaged food, read the brand/label; if a nutrition label is visible use its values.
Estimate each item's weight in grams from plate/bowl/hand/object size. Nutrition values must be for the estimated weight (not per 100 g).
If no food is visible return {"items": [], "note": "No food detected"}.
Return ONLY JSON: {"items":[{"name":str,"category":"home|restaurant|packaged|fruit|beverage|other","grams":number,"calories":number,
"protein":number,"fat":number,"carbs":number,"fiber":number,"sugar":number,"sodium_mg":number,"glycemic_index":number,"confidence":0-100}],
"health_score":number(1-10),"health_note":str,"suggestions":[str,str,str]}"""

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), u=Depends(current_user)):
    data = await file.read()
    mime = file.content_type if (file.content_type or "").startswith("image/") else "image/jpeg"
    return gemini([{"text": FOOD_PROMPT}, {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}])

# ---------------- Meals / dashboard ----------------
class Item(BaseModel):
    name: str; grams: float = 0; calories: float = 0; protein: float = 0; fat: float = 0
    carbs: float = 0; fiber: float = 0; sugar: float = 0; sodium_mg: float = 0
class MealIn(BaseModel): meal: str; items: list[Item]

@app.post("/api/meals")
def add_meals(m: MealIn, u=Depends(current_user)):
    day = str(datetime.date.today())
    with db() as c:
        for i in m.items:
            c.execute("INSERT INTO meals(uid,day,meal,food,grams,calories,protein,fat,carbs,fiber,sugar,sodium) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (u["id"], day, m.meal, i.name, i.grams, i.calories, i.protein, i.fat, i.carbs, i.fiber, i.sugar, i.sodium_mg))
    return {"ok": True}

@app.delete("/api/meals/{mid}")
def del_meal(mid: int, u=Depends(current_user)):
    with db() as c: c.execute("DELETE FROM meals WHERE id=? AND uid=?", (mid, u["id"]))
    return {"ok": True}

@app.get("/api/summary")
def summary(u=Depends(current_user)):
    today = str(datetime.date.today())
    with db() as c:
        meals = [dict(r) for r in c.execute("SELECT * FROM meals WHERE uid=? AND day=? ORDER BY id", (u["id"], today))]
        wk = {r["day"]: r["k"] for r in c.execute("SELECT day, SUM(calories) k FROM meals WHERE uid=? GROUP BY day ORDER BY day DESC LIMIT 7", (u["id"],))}
    tot = {k: round(sum(m[k] for m in meals), 1) for k in ("calories", "protein", "fat", "carbs", "fiber", "sugar", "sodium")}
    prof = json.loads(u["profile"]); plan = calc_plan(prof) if has_body_profile(prof) else None
    days = [str(datetime.date.today() - datetime.timedelta(days=i)) for i in range(6, -1, -1)]
    return dict(meals=meals, total=tot, plan=plan, week=[{"day": d[5:], "kcal": round(wk.get(d, 0))} for d in days])

# ---------------- BMI + Diet plan ----------------
class Profile(BaseModel):
    age: int; weight: float; height: float; sex: str = "male"; activity: float = 1.4; goal: str = "maintain"
    diet: str = "vegetarian"; cuisine: str = "Indian"; conditions: str = ""

@app.post("/api/profile")
def save_profile(p: Profile, u=Depends(current_user)):
    prof = json.loads(u["profile"] or "{}"); prof.update(p.model_dump())        # keep voice-saved preferences
    with db() as c: c.execute("UPDATE users SET profile=? WHERE id=?", (json.dumps(prof), u["id"]))
    return calc_plan(p.model_dump())

@app.post("/api/diet-plan")
def diet_plan(p: Profile, u=Depends(current_user)):
    save_profile(p, u); u = {**u, "profile": json.dumps({**json.loads(u["profile"] or "{}"), **p.model_dump()})}
    plan = calc_plan(p.model_dump()); pref_text = prefs_mod.describe(json.loads(u["profile"]).get("preferences"))
    prompt = f"""Create a practical 7-day {p.diet} {p.cuisine} diet plan for: age {p.age}, {p.sex}, {p.weight} kg, {p.height} cm, BMI {plan['bmi']} ({plan['category']}),
goal: {p.goal}, activity factor {p.activity}, health conditions/allergies: {p.conditions or 'none'}. Food preferences told by voice: {pref_text} (never include avoided foods).
Daily target: {plan['target']} kcal, protein {plan['protein_g']} g, fat {plan['fat_g']} g, carbs {plan['carbs_g']} g. Use affordable everyday foods with portion sizes.
Return ONLY JSON: {{"summary":str,"days":[{{"day":"Monday","meals":[{{"name":"Breakfast|Mid-morning|Lunch|Evening snack|Dinner","menu":str,"kcal":number}}],"total_kcal":number}}],
"eat_more":[str],"avoid":[str],"habits":[str],"disclaimer":"General guidance, consult a doctor/dietitian for medical conditions."}}"""
    result = gemini([{"text": prompt}]); result["numbers"] = plan
    with db() as c: c.execute("INSERT OR REPLACE INTO plans(uid,plan) VALUES(?,?)", (u["id"], json.dumps(result)))
    return result

@app.get("/api/diet-plan")
def last_plan(u=Depends(current_user)):
    with db() as c: r = c.execute("SELECT plan FROM plans WHERE uid=?", (u["id"],)).fetchone()
    return json.loads(r["plan"]) if r else None

@app.get("/sw.js")
def sw(): return FileResponse(BASE / "static" / "sw.js", media_type="application/javascript", headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})
@app.get("/manifest.json")
def manifest(): return FileResponse(BASE / "static" / "manifest.json", media_type="application/manifest+json")

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
@app.get("/")
def index(): return FileResponse(BASE / "static" / "index.html")
