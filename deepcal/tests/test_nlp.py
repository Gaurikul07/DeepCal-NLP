"""Unit tests for the NLP pieces. They need no API key, no network and no FastAPI.   Run:  pytest -q"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.nlp.extractor import extract_items, extract_meal_time, extract_nutrient, extract_plan_target
from app.nlp.foodmatch import FoodIndex
from app.nlp.preferences import extract_preferences, check_items, merge_preferences
from app.nlp import intents, evaluate

IX = FoodIndex.from_csv(ROOT / "data")


# ---- extractor ----
def test_english_quantity_and_unit():
    assert extract_items("I ate two rotis and a bowl of dal for lunch") == [
        {"food": "rotis", "quantity": 2, "unit": None, "grams": None}, {"food": "dal", "quantity": 1, "unit": "bowl", "grams": None}]

def test_hinglish_numbers_and_meal_time():
    t = "maine subah do paratha aur chai li"
    assert extract_meal_time(t) == ("breakfast", True)
    assert [(i["food"], i["quantity"]) for i in extract_items(t)] == [("paratha", 2), ("chai", 1)]

def test_hindi_fractions():
    assert extract_items("dedh katori rajma")[0]["quantity"] == 1.5
    assert extract_items("mene aadha paratha khaya")[0]["quantity"] == 0.5
    assert extract_items("I had 1/2 cup oats")[0]["quantity"] == 0.5
    assert extract_items("one and a half roti")[0]["quantity"] == 1.5

def test_grams_are_taken_directly():
    it = extract_items("250g rice")[0]
    assert (it["unit"], it["grams"]) == ("g", 250.0)

def test_nutrient_and_plan_target():
    assert extract_nutrient("how much protein have I had today") == "protein"
    assert extract_plan_target("swap Wednesday's dinner for something vegetarian") == ("Wednesday", "Dinner")
    assert extract_plan_target("budhvar ka lunch badlo") == ("Wednesday", "Lunch")


# ---- food matching + units ----
def test_aliases_map_to_one_food():
    names = {IX.match(a)[0]["name"] for a in ("roti", "chapati", "phulka", "fulka")}
    assert names == {"roti"}

def test_katori_dal_and_bowl_daal_resolve_to_same_food():
    a = IX.resolve(extract_items("ek katori dal")[0]); b = IX.resolve(extract_items("1 bowl daal")[0])
    assert a["name"] == b["name"] == "dal" and a["grams"] == b["grams"] == 150

def test_piece_units_and_scaling():
    r = IX.resolve(extract_items("two rotis")[0])
    assert r["grams"] == 80 and abs(r["calories"] - 228) < 1

def test_unknown_food_is_not_forced_into_the_database():
    assert IX.match("zzzz quantum soufflé")[0] is None

def test_fuzzy_match_is_flagged_uncertain():
    assert IX.resolve({"food": "peanut chaat", "quantity": 1, "unit": None, "grams": None}).get("uncertain") in (True, False)


# ---- preferences ----
def test_preference_extraction_separates_allergy_from_dislike():
    p = extract_preferences("I'm allergic to peanuts and don't eat onion")
    assert p["allergies"] == ["peanut"] and set(p["avoid"]) == {"peanut", "onion"}

def test_preference_does_not_leak_across_clauses():
    assert extract_preferences("I don't eat onion but I love milk")["avoid"] == ["onion"]

def test_preference_warnings_use_food_tags():
    prefs = merge_preferences({}, extract_preferences("I am lactose intolerant"))
    paneer = IX.resolve({"food": "paneer", "quantity": 1, "unit": None, "grams": None})
    assert check_items([paneer], prefs)
    assert not check_items([IX.resolve({"food": "rice", "quantity": 1, "unit": None, "grams": None})], prefs)


# ---- intent classifier + evaluation ----
def test_classifier_obvious_sentences():
    if intents.load_model() is None: return        # run `python -m app.nlp.train` first
    assert intents.predict("I ate two rotis and dal for lunch")[0] == "log_meal"
    assert intents.predict("how many calories do I have left today")[0] == "remaining_calories"

def test_extraction_accuracy_on_test_set_is_reasonable():
    stats = evaluate.extraction_report()
    assert int(stats["all"].split("/")[0]) / int(stats["all"].split("/")[1].split()[0]) >= 0.85
