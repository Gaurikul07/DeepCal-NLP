"""Measure your own NLP against the test sets.   python -m app.nlp.evaluate
 1. Extraction accuracy of the rule-based extractor on data/extraction_test.csv (offline, no API key needed).
 2. Intent accuracy: classifier-only vs hybrid on data/intents.csv holdout (see models/metrics.json), and, if GEMINI_API_KEY is set,
    Gemini-only and extraction-by-Gemini on the same sentences for a side-by-side table (costs a few API calls)."""
import csv, json, sys
from pathlib import Path
from app.nlp.extractor import extract_items
from app.nlp.foodmatch import FoodIndex

ROOT = Path(__file__).resolve().parents[2]

def parse_expected(s):
    out = set()
    for part in s.split(";"):
        n, q, u = part.rsplit(":", 2)
        out.add((n, float(q), u or None))
    return out

def rules_extract(ix, text):
    got = set()
    for it in extract_items(text):
        r = ix.resolve(it)
        got.add((r["name"] if r else it["food"], float(it["quantity"]), it["unit"]))
    return got

def extraction_report(extractor=None):
    ix = FoodIndex.from_csv(ROOT / "data")
    rows = list(csv.DictReader(open(ROOT / "data" / "extraction_test.csv", encoding="utf-8")))
    stats = {}
    for r in rows:
        got = (extractor or (lambda t: rules_extract(ix, t)))(r["text"])
        ok = got == parse_expected(r["expected"])
        for k in ("all", r["lang"]):
            a = stats.setdefault(k, [0, 0]); a[0] += ok; a[1] += 1
        if not ok: print("  MISS:", r["text"], "| expected", sorted(parse_expected(r["expected"]), key=str), "| got", sorted(got, key=str))
    return {k: f"{a[0]}/{a[1]} = {a[0]/a[1]:.0%}" for k, a in stats.items()}

if __name__ == "__main__":
    print("Rule-based extraction (exact match on food, quantity, unit):"); print(extraction_report())
    m = ROOT / "models" / "metrics.json"
    if m.exists():
        d = json.load(open(m)); print(f"Intent classifier holdout accuracy {d['holdout_accuracy']:.1%}; handles {d['model_coverage']:.0%} at conf>={d['threshold']} with {d['accuracy_when_confident']:.1%} accuracy")
    try:
        from app.core import KEY
        if KEY and "--gemini" in sys.argv:
            from app.nlp.gemini_parser import parse_with_gemini
            def via_gemini(t):
                ix = FoodIndex.from_csv(ROOT / "data")
                return {((ix.resolve(i.model_dump()) or {"name": i.food})["name"], float(i.quantity or 1), i.unit) for i in parse_with_gemini(t).items}
            print("Gemini extraction:"); print(extraction_report(via_gemini))
    except Exception as e:
        print("Gemini comparison skipped:", e)
