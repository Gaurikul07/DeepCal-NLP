"""Indian food lookup: alias matching (rapidfuzz, with a difflib fallback), katori/roti/glass -> grams, nutrition scaling.
Database first, Gemini only as a fallback (see voice.py). That is faster, cheaper and gives the same numbers every time."""
import csv, difflib
from pathlib import Path

try:
    from rapidfuzz import process, fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:          # keeps the project usable (and testable) without the optional dependency
    HAVE_RAPIDFUZZ = False

MATCH_THRESHOLD = 82         # 0-100. Below this we treat the food as "unknown" and ask Gemini.
NUT = ("kcal", "protein", "fat", "carbs", "fiber", "sugar", "sodium_mg")


class FoodIndex:
    def __init__(self, foods: list[dict], units: list[dict]):
        self.foods = {f["name"]: f for f in foods}
        self.choices = {}                                    # searchable string -> canonical food name
        for f in foods:
            self.choices[f["name"].lower()] = f["name"]
            for a in (f.get("aliases") or "").split(";"):
                if a.strip(): self.choices.setdefault(a.strip().lower(), f["name"])
        self.units = {(u["unit"].lower(), u["food"].lower()): float(u["grams"]) for u in units}

    # ---- loaders ----
    @classmethod
    def from_csv(cls, data_dir):
        d = Path(data_dir)
        foods = list(csv.DictReader(open(d / "foods.csv", encoding="utf-8")))
        units = list(csv.DictReader(open(d / "units.csv", encoding="utf-8")))
        return cls(foods, units)

    @classmethod
    def from_db(cls, conn):
        foods = [dict(r) for r in conn.execute("SELECT * FROM foods")]
        units = [dict(r) for r in conn.execute("SELECT * FROM units")]
        return cls(foods, units)

    # ---- matching ----
    def match(self, query: str):
        """Return (food_dict | None, score 0-100)."""
        q = " ".join(query.lower().split())
        if not q: return None, 0
        for cand in (q, q[:-2] if q.endswith("es") else None, q[:-1] if q.endswith("s") else None):
            if cand and cand in self.choices: return self.foods[self.choices[cand]], 100
        if HAVE_RAPIDFUZZ:
            hit = process.extractOne(q, list(self.choices), scorer=fuzz.WRatio)
            best, score = (hit[0], hit[1]) if hit else (None, 0)
        else:
            best, score = None, 0
            words = set(q.split())
            for c in self.choices:
                s = difflib.SequenceMatcher(None, q, c).ratio() * 100
                if len(c) >= 3 and c in words: s = max(s, 90)          # whole-word containment, e.g. "boiled rice" ~ "rice"
                if s > score: best, score = c, s
        if best and score >= MATCH_THRESHOLD: return self.foods[self.choices[best]], round(score)
        return None, round(score)

    # ---- units ----
    def grams_for(self, food: dict, quantity: float, unit: str | None) -> float:
        serving = float(food.get("serving_g") or 100)
        if unit in (None, "piece"): return round(quantity * serving, 1)
        name = food["name"].lower()
        per = self.units.get((unit, name)) or self.units.get((unit, "*"))
        return round(quantity * (per if per else serving), 1)

    # ---- nutrition ----
    @staticmethod
    def nutrition(food: dict, grams: float) -> dict:
        k = grams / 100.0
        return {"calories": round(float(food["kcal"]) * k, 1), "protein": round(float(food["protein"]) * k, 1),
                "fat": round(float(food["fat"]) * k, 1), "carbs": round(float(food["carbs"]) * k, 1),
                "fiber": round(float(food["fiber"]) * k, 1), "sugar": round(float(food["sugar"]) * k, 1),
                "sodium_mg": round(float(food["sodium_mg"]) * k, 1)}

    def resolve(self, item: dict):
        """item = {food, quantity, unit, grams}. Returns a full meal-item dict, or None if the food is not in the database."""
        food, score = self.match(item["food"])
        if not food: return None
        grams = item.get("grams") or self.grams_for(food, item.get("quantity") or 1, item.get("unit"))
        out = {"name": food["name"], "heard_as": item["food"], "grams": grams, "quantity": item.get("quantity") or 1,
               "unit": item.get("unit"), "source": "database", "match_score": score,
               "uncertain": score < 95,        # fuzzy match: the UI highlights it so the user double-checks (e.g. "peanut chaat" -> peanuts)
               "tags": [t for t in (food.get("tags") or "").split(";") if t]}
        out.update(self.nutrition(food, grams))
        return out


_INDEX = None
def get_index(force_reload: bool = False) -> FoodIndex:
    """Lazy singleton backed by the SQLite tables (so edits you make in the DB are picked up on reload)."""
    global _INDEX
    if _INDEX is None or force_reload:
        from app.core import db
        with db() as c: _INDEX = FoodIndex.from_db(c)
    return _INDEX
