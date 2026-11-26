"""Turns "I'm allergic to peanuts and don't eat onion" into structured preferences, and checks foods against them."""
import re
from .extractor import _normalise

# keyword -> canonical tag (same tags as the `tags` column in foods.csv)
AVOID_KEYWORDS = {
    "onion": "onion", "onions": "onion", "pyaz": "onion", "pyaaj": "onion", "garlic": "garlic", "lehsun": "garlic",
    "peanut": "peanut", "peanuts": "peanut", "mungfali": "peanut", "moongfali": "peanut", "groundnut": "peanut", "groundnuts": "peanut",
    "nuts": "nuts", "nut": "nuts", "almond": "nuts", "almonds": "nuts", "cashew": "nuts", "cashews": "nuts",
    "lactose": "dairy", "dairy": "dairy", "milk": "dairy", "doodh": "dairy", "dahi": "dairy", "curd": "dairy", "paneer": "dairy",
    "gluten": "gluten", "wheat": "gluten", "maida": "gluten", "egg": "egg", "eggs": "egg", "anda": "egg", "ande": "egg",
    "fish": "fish", "seafood": "fish", "meat": "nonveg", "chicken": "nonveg", "mutton": "nonveg", "nonveg": "nonveg",
    "soy": "soy", "soya": "soy",
}
DIET_WORDS = {"vegetarian": "vegetarian", "veg": "vegetarian", "shakahari": "vegetarian", "vegan": "vegan", "jain": "jain",
              "eggetarian": "eggetarian"}
DIET_BLOCKS = {"vegan": {"dairy", "egg", "nonveg", "fish"}, "vegetarian": {"nonveg", "fish", "egg"},
               "eggetarian": {"nonveg", "fish"}, "jain": {"nonveg", "fish", "egg", "onion"}}
AVOID_CUE = re.compile(r"\b(dont|do not|doesnt|cant|cannot|avoid|no|hate|dislike|allergic|allergy|intolerant|intolerance|"
                       r"nahi|nahin|na khata|na khati|mat|without|not eat)\b")
ALLERGY_CUE = re.compile(r"\b(allergic|allergy|intolerant|intolerance)\b")
CLAUSE_SPLIT = re.compile(r",|\bbut\b|\blekin\b|\bpar\b|\.|;")


def extract_preferences(text: str) -> dict:
    """-> {"avoid": [tags], "allergies": [tags], "diet": str|None}. Empty lists when nothing is found."""
    t = text.lower().replace("don't", "dont").replace("doesn't", "doesnt").replace("can't", "cant").replace("isn't", "isnt")
    avoid, allergies, diet = [], [], None
    for clause in CLAUSE_SPLIT.split(t):
        n = _normalise(clause)
        words = n.split()
        for w in words:
            if w in DIET_WORDS and re.search(r"\b(i am|im|i'm|main|hoon|hu|pure|only)\b", n + " " + clause): diet = DIET_WORDS[w]
        state = None                      # None = no cue yet, "avoid" or "allergy"; carried across "and"/"aur" inside a clause
        for part in re.split(r"\band\b|\baur\b", n):
            if ALLERGY_CUE.search(part): state = "allergy"
            elif AVOID_CUE.search(part): state = "avoid"
            if state is None: continue
            for w in part.split():
                tag = AVOID_KEYWORDS.get(w)
                if not tag: continue
                if tag not in avoid: avoid.append(tag)
                if state == "allergy" and tag not in allergies: allergies.append(tag)
    return {"avoid": avoid, "allergies": allergies, "diet": diet}


def merge_preferences(current: dict | None, new: dict) -> dict:
    cur = dict(current or {})
    for k in ("avoid", "allergies"):
        cur[k] = sorted(set(cur.get(k, [])) | set(new.get(k, [])))
    if new.get("diet"): cur["diet"] = new["diet"]
    return cur


def remove_preference(current: dict, tag: str) -> dict:
    cur = dict(current or {})
    for k in ("avoid", "allergies"): cur[k] = [x for x in cur.get(k, []) if x != tag]
    if cur.get("diet") == tag: cur["diet"] = None
    return cur


def check_items(items: list[dict], prefs: dict | None) -> list[str]:
    """Warnings for foods that clash with what the user told us. Tags come from the foods table, so unknown foods are not checked."""
    if not prefs: return []
    blocked = set(prefs.get("avoid", [])) | DIET_BLOCKS.get(prefs.get("diet") or "", set())
    allergies = set(prefs.get("allergies", []))
    out = []
    for it in items:
        hit = [t for t in it.get("tags", []) if t in blocked]
        if not hit: continue
        sev = "Allergy warning" if set(hit) & allergies else "Heads up"
        out.append(f"{sev}: {it['name']} contains {', '.join(hit)}, which you asked me to avoid.")
    return out


def describe(prefs: dict | None) -> str:
    """Short text used inside Gemini prompts (meal suggestions, plan edits, weekly plan generation)."""
    if not prefs: return "none"
    bits = []
    if prefs.get("diet"): bits.append(f"diet: {prefs['diet']}")
    if prefs.get("allergies"): bits.append("ALLERGIC to: " + ", ".join(prefs["allergies"]))
    rest = [a for a in prefs.get("avoid", []) if a not in prefs.get("allergies", [])]
    if rest: bits.append("avoids: " + ", ".join(rest))
    return "; ".join(bits) or "none"
