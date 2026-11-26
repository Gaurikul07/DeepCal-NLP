"""Everything that talks to Gemini for the voice assistant. Gemini returns JSON, Pydantic validates it,
and our own Python code decides what to do. Gemini never writes SQL and never touches the database."""
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

Intent = Literal["log_meal", "today_summary", "remaining_calories", "suggest_meal", "edit_plan", "set_preference", "weekly_insight", "chat"]
Meal = Literal["breakfast", "lunch", "dinner", "snack"]


class ParseError(Exception):
    """Gemini answered, but not with something we can trust."""


def _gj(prompt):
    from app.core import gemini_json          # imported lazily so unit tests can run without FastAPI installed
    return gemini_json(prompt)


class ParsedItem(BaseModel):
    food: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    grams: Optional[float] = None

    @field_validator("food")
    @classmethod
    def _food(cls, v):
        v = (v or "").strip().lower()
        if not v: raise ValueError("empty food")
        return v

    @field_validator("quantity", "grams", mode="before")
    @classmethod
    def _num(cls, v):
        try: return float(v) if v not in (None, "") else None
        except (TypeError, ValueError): return None


class Parsed(BaseModel):
    intent: Intent = "chat"
    meal: Optional[Meal] = None
    items: list[ParsedItem] = Field(default_factory=list)

    @field_validator("meal", mode="before")
    @classmethod
    def _meal(cls, v):
        v = str(v).strip().lower() if v else None
        return v if v in ("breakfast", "lunch", "dinner", "snack") else None


class NutriItem(BaseModel):
    name: str
    grams: float = 0
    calories: float = 0
    protein: float = 0
    fat: float = 0
    carbs: float = 0
    fiber: float = 0
    sugar: float = 0
    sodium_mg: float = 0


PARSE_PROMPT = """You are DeepCal's nutrition assistant. The user's sentence may be English, Hindi (Roman script) or mixed Hinglish,
for example "maine subah do paratha aur chai li" or "ek katori dal". It comes from speech recognition, so food names may be misspelled.
Convert it into JSON only, with no extra text:
{"intent": "log_meal | today_summary | remaining_calories | suggest_meal | edit_plan | set_preference | weekly_insight | chat",
 "meal": "breakfast | lunch | dinner | snack | null",
 "items": [{"food": "", "quantity": null, "grams": null, "unit": "katori|bowl|cup|glass|plate|piece|tbsp|tsp|slice|handful|scoop|g|ml|null"}]}
Rules: items only for log_meal (food in English, singular, e.g. "roti" not "rotis"); quantity is a number ("do" = 2, "aadha" = 0.5);
use the first sentence's meaning, not your own opinion. User: """


def parse_with_gemini(text: str) -> Parsed:
    raw = _gj(PARSE_PROMPT + json.dumps(text))
    try:
        if isinstance(raw, list): raw = {"intent": "log_meal", "items": raw}
        return Parsed.model_validate(raw)
    except ValidationError as e:
        raise ParseError(str(e)) from e


def nutrition_with_gemini(items: list[dict]) -> list[dict]:
    """Fallback for foods that are not in our Indian foods table. Returns validated meal-item dicts tagged source='gemini'."""
    prompt = ("You are a nutritionist. For each food below estimate the weight in grams (use quantity+unit if given; a katori is about 150 g, "
              "a roti about 40 g, a glass about 250 ml) and the nutrition for THAT weight, using Indian/home-style preparation by default.\n"
              'Return ONLY JSON: {"items":[{"name":str,"grams":number,"calories":number,"protein":number,"fat":number,"carbs":number,'
              '"fiber":number,"sugar":number,"sodium_mg":number}]} in the same order.\nFoods: ' + json.dumps(items))
    raw = _gj(prompt)
    out = []
    for src, r in zip(items, (raw or {}).get("items", [])):
        try: n = NutriItem.model_validate(r)
        except ValidationError: continue
        d = n.model_dump(); d.update(heard_as=src["food"], quantity=src.get("quantity") or 1, unit=src.get("unit"), source="gemini", match_score=0, tags=[])
        out.append(d)
    return out


def suggest_with_gemini(meal: str | None, remaining: dict, prefs_text: str, diet: str, text: str) -> dict:
    prompt = (f"Suggest 3 practical Indian meal ideas for {meal or 'the next meal'}. User request: {json.dumps(text)}.\n"
              f"Remaining today: {json.dumps(remaining)}. Preferences/allergies: {prefs_text}. Diet type: {diet}. Never include foods the user avoids.\n"
              'Return ONLY JSON: {"reply": "one friendly spoken sentence summarising the options (max 45 words)", '
              '"suggestions":[{"name":str,"why":str,"kcal":number}]}. General guidance only, not medical advice.')
    return _gj(prompt)


def plan_edit_with_gemini(day: str, slot: str, current: dict | None, request: str, prefs_text: str, target_kcal: float | None) -> dict:
    prompt = (f"A user's 7-day diet plan has this meal: {json.dumps(current)} ({day}, {slot}). They said: {json.dumps(request)}.\n"
              f"Preferences/allergies: {prefs_text}. Meal target about {target_kcal or 'unchanged'} kcal. Write the replacement meal.\n"
              'Return ONLY JSON: {"menu": "dishes with portion sizes", "kcal": number, "reply": "one short spoken sentence"}')
    return _gj(prompt)


def correct_items_with_gemini(items: list[dict], correction: str) -> list[dict]:
    prompt = ("A food-photo scan detected these items (nutrition is for the grams shown): " + json.dumps(items) +
              f"\nThe user corrected it by voice (English/Hinglish): {json.dumps(correction)}.\n"
              "Apply ONLY what they said (change quantity, rename, add or remove an item) and recompute nutrition for new weights.\n"
              'Return ONLY JSON: {"items":[{"name":str,"category":str,"grams":number,"calories":number,"protein":number,"fat":number,'
              '"carbs":number,"fiber":number,"sugar":number,"sodium_mg":number,"confidence":number}]}')
    return (_gj(prompt) or {}).get("items", [])


def transcribe_audio(audio_b64: str, mime: str) -> str:
    from app.core import gemini
    raw = gemini([{"text": 'Transcribe this short voice note exactly. It may be English, Hindi or Hinglish; write Hindi words in Roman script '
                           '(e.g. "maine do roti khayi"). Food names matter most. Return ONLY JSON: {"transcript": str}'},
                  {"inline_data": {"mime_type": mime, "data": audio_b64}}])
    t = (raw or {}).get("transcript", "")
    if not isinstance(t, str) or not t.strip(): raise ParseError("empty transcript")
    return t.strip()


def explain_with_gemini(question: str, data: dict, prefs_text: str, kind: str = "coach") -> str:
    """The model only EXPLAINS numbers that our fixed SQL tools produced. It cannot query anything itself."""
    task = ("Write a short weekly insight (max 90 words): one trend, one win, one suggestion." if kind == "weekly"
            else "Answer the user's question using ONLY the data given. If the data cannot answer it, say so.")
    prompt = (f"You are DeepCal's friendly nutrition coach for an Indian user. {task} Plain sentences, no markdown, suitable for reading aloud. "
              "General guidance only; do not diagnose or give medical advice.\n"
              f"Question: {json.dumps(question)}\nData (from the user's own logs): {json.dumps(data)}\nPreferences: {prefs_text}\n"
              'Return ONLY JSON: {"reply": str}')
    return str((_gj(prompt) or {}).get("reply", "")).strip()


def chat_with_gemini(text: str, prefs_text: str) -> str:
    prompt = ("You are DeepCal's nutrition assistant for an Indian user. Answer in at most 60 words, plain sentences (it will be read aloud). "
              "General guidance only, not medical advice; for medical conditions suggest a doctor or dietitian.\n"
              f"Preferences: {prefs_text}\nUser: {json.dumps(text)}\nReturn ONLY JSON: " + '{"reply": str}')
    return str((_gj(prompt) or {}).get("reply", "")).strip()


def pick_tools(question: str, tool_names: list[str]) -> list[str]:
    """Coach chat with fixed tools: Gemini only chooses from our whitelist; the server runs the SQL functions."""
    raw = _gj(f"Pick which of these data tools are needed to answer the user's question: {tool_names}. "
              f'Question: {json.dumps(question)}. Return ONLY JSON: {{"tools": [names]}}. Return an empty list if none are needed.')
    chosen = (raw or {}).get("tools", []) if isinstance(raw, dict) else []
    return [t for t in chosen if t in tool_names]
