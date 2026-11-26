"""Rule-based extraction of meal time, food items, quantities and units from English / Hindi / Hinglish text.
No external services: this is the "your own NLP" part. Gemini is only used when these rules find nothing."""
import re, datetime

NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "half": 0.5, "quarter": 0.25, "couple": 2,
    # Hindi in Roman script (spellings vary, so list the common ones)
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6, "saat": 7,
    "aath": 8, "nau": 9, "das": 10, "aadha": 0.5, "adha": 0.5, "aadhi": 0.5, "dedh": 1.5, "dhai": 2.5, "dhaai": 2.5,
    "sava": 1.25, "paune": 0.75,
}
ARTICLES = {"a", "an", "ek"}

UNIT_SYNONYMS = {
    "katori": "katori", "katoris": "katori", "katora": "katori", "bowl": "bowl", "bowls": "bowl", "cup": "cup", "cups": "cup",
    "glass": "glass", "glasses": "glass", "gilas": "glass", "glaas": "glass", "plate": "plate", "plates": "plate",
    "piece": "piece", "pieces": "piece", "pcs": "piece", "pc": "piece", "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp", "spoon": "spoon", "spoons": "spoon", "chamach": "spoon", "chammach": "spoon",
    "slice": "slice", "slices": "slice", "handful": "handful", "mutthi": "handful", "scoop": "scoop", "scoops": "scoop",
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g", "gramme": "g", "kg": "kg", "kilo": "kg", "ml": "ml",
    "litre": "litre", "liter": "litre", "litres": "litre", "liters": "litre", "bottle": "bottle", "can": "can",
}
MASS_UNITS = {"g", "kg", "ml", "litre"}   # when quantity is given in these, grams are known directly

MEAL_WORDS = {
    "breakfast": "breakfast", "nashta": "breakfast", "naashta": "breakfast", "nasta": "breakfast", "subah": "breakfast",
    "morning": "breakfast", "brunch": "breakfast",
    "lunch": "lunch", "dopahar": "lunch", "afternoon": "lunch", "lunchtime": "lunch",
    "dinner": "dinner", "raat": "dinner", "night": "dinner", "supper": "dinner", "dinnertime": "dinner",
    "snack": "snack", "snacks": "snack", "shaam": "snack", "evening": "snack", "tiffin": "snack",
}

STOPWORDS = set("""i me my mine we had have has having ate eat eaten eating took take taking drank drink drinking just log add
please also today aaj yesterday kal for in at on the of some little bit thoda thodi sa si with plus
maine mene maine mujhe humne khaya khayi khaye khai kha khaaya khayaa liya li lia liye piya pi piyi pee kiya kiye kar karo
ka ki ke ko me mein se ne tha thi the hai hain hu hoon bhi aur and or then after before about around approximately approx
meal khana khane food entry record note to my was were is are be been it that this those these so like around""".split())

SPLIT_RE = re.compile(r"\s*(?:,|;|\+|&|\band\b|\baur\b|\bplus\b|\bwith\b|\btatha\b|\bke saath\b|\bsaath\b)\s*")


def _normalise(text: str) -> str:
    t = text.lower().replace("½", " 0.5 ").replace("¼", " 0.25 ").replace("¾", " 0.75 ")
    t = re.sub(r"(\d+)\s*/\s*(\d+)", lambda m: f" {int(m.group(1)) / max(int(m.group(2)), 1):g} ", t)   # 1/2 -> 0.5
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)                                                       # 250g -> 250 g
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    t = re.sub(r"\b(one|two|three|ek|do|teen|1|2|3)\s+and\s+a\s+half\b",
               lambda m: f" {(NUM_WORDS.get(m.group(1)) or float(m.group(1))) + 0.5:g} ", t)       # "one and a half" -> 1.5
    t = re.sub(r"[^a-z0-9.,;+&/ ]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _num(tok):
    if tok in NUM_WORDS: return NUM_WORDS[tok]
    try: return float(tok)
    except ValueError: return None


def extract_meal_time(text: str, now: datetime.datetime | None = None):
    """Return (meal, explicit). explicit=False means we guessed from the clock."""
    for w in _normalise(text).split():
        if w in MEAL_WORDS: return MEAL_WORDS[w], True
    h = (now or datetime.datetime.now()).hour
    return ("breakfast" if h < 11 else "lunch" if h < 16 else "snack" if h < 19 else "dinner"), False


def extract_items(text: str):
    """'maine subah do paratha aur chai li' -> [{food:'paratha', quantity:2, unit:None}, {food:'chai', quantity:1, unit:None}]"""
    items = []
    for chunk in SPLIT_RE.split(_normalise(text)):
        toks = [t for t in chunk.split() if t not in MEAL_WORDS]
        if not toks: continue
        qty, unit, used = None, None, set()
        # unit first, so "a bowl of dal" and "bowl dal" work; then the number just before it
        for i, t in enumerate(toks):
            if t in UNIT_SYNONYMS and (i == 0 or toks[i-1] != "of"):
                unit = UNIT_SYNONYMS[t]; used.add(i)
                j = i - 1; nums = []
                while j >= 0 and _num(toks[j]) is not None: nums.append((j, toks[j])); j -= 1
                if not nums and i + 1 < len(toks) and _num(toks[i+1]) is not None and toks[i+1] not in ARTICLES:
                    nums = [(i + 1, toks[i+1])]                       # "rice 200 g" style handled below; "bowl 2" rare
                if nums:
                    vals = [(k, _num(w)) for k, w in nums]
                    real = [v for k, v in vals if toks[k] not in ARTICLES] or [v for _, v in vals]
                    qty = real[0]; used.update(k for k, _ in nums)
                break
        if unit is None or qty is None:
            for i, t in enumerate(toks):
                if i in used: continue
                v = _num(t)
                if v is not None and not (t in ARTICLES and len(toks) == 1):
                    if qty is None: qty = v
                    used.add(i)
                    if t not in ARTICLES and t != "half": break
                    # allow "half a" / "a half": keep scanning for a stronger number
                    continue
        # "rice 200 g": number then mass unit appearing after the food
        if unit is None:
            for i in range(len(toks) - 1):
                if _num(toks[i]) is not None and toks[i+1] in MASS_UNITS | {"g"} and toks[i+1] in UNIT_SYNONYMS:
                    qty, unit = _num(toks[i]), UNIT_SYNONYMS[toks[i+1]]; used.update({i, i+1})
        food_toks = [t for i, t in enumerate(toks) if i not in used and t not in STOPWORDS and t != "of" and _num(t) is None]
        # numbers used as qty are already in `used`; stray number words that were food-like (none) are dropped
        food = " ".join(food_toks).strip()
        if not food: continue
        q = qty if qty is not None else 1.0
        items.append({"food": food, "quantity": q, "unit": unit,
                      "grams": (q * {"g": 1, "kg": 1000, "ml": 1, "litre": 1000}[unit]) if unit in MASS_UNITS else None})
    return items


# ---------- nutrient keyword for "how much protein have I had" ----------
NUTRIENT_WORDS = {
    "calories": ["calorie", "calories", "kcal", "cal", "calorie's"], "protein": ["protein", "proteins"],
    "carbs": ["carb", "carbs", "carbohydrate", "carbohydrates"], "fat": ["fat", "fats"], "fiber": ["fiber", "fibre"],
    "sugar": ["sugar", "cheeni", "meetha", "sweet"], "sodium": ["sodium", "salt", "namak"],
}
def extract_nutrient(text: str):
    words = set(_normalise(text).split())
    for k, ws in NUTRIENT_WORDS.items():
        if words & set(ws): return k
    return None


# ---------- day + meal slot for "swap Wednesday's dinner for something vegetarian" ----------
DAY_WORDS = {
    "monday": "Monday", "mon": "Monday", "somvar": "Monday", "somvaar": "Monday", "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
    "mangalvar": "Tuesday", "mangalvaar": "Tuesday", "wednesday": "Wednesday", "wed": "Wednesday", "budhvar": "Wednesday", "budhvaar": "Wednesday",
    "thursday": "Thursday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "guruvar": "Thursday", "guruvaar": "Thursday", "veervar": "Thursday",
    "friday": "Friday", "fri": "Friday", "shukravar": "Friday", "shukravaar": "Friday", "saturday": "Saturday", "sat": "Saturday",
    "shanivar": "Saturday", "shanivaar": "Saturday", "sunday": "Sunday", "sun": "Sunday", "ravivar": "Sunday", "ravivaar": "Sunday", "itwar": "Sunday",
}
SLOT_WORDS = {"breakfast": "Breakfast", "nashta": "Breakfast", "naashta": "Breakfast", "lunch": "Lunch", "dopahar": "Lunch",
              "dinner": "Dinner", "raat": "Dinner", "snack": "Evening snack", "snacks": "Evening snack", "shaam": "Evening snack",
              "midmorning": "Mid-morning"}
def extract_plan_target(text: str):
    t = _normalise(text).replace("mid morning", "midmorning").replace("mid-morning", "midmorning")
    t = re.sub(r"'s\b", "", t)
    words = t.split()
    day = next((DAY_WORDS[w] for w in words if w in DAY_WORDS), None)
    if day is None and ("today" in words or "aaj" in words): day = datetime.date.today().strftime("%A")
    if day is None and ("tomorrow" in words or "kal" in words): day = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%A")
    slot = next((SLOT_WORDS[w] for w in words if w in SLOT_WORDS), None)
    return day, slot
