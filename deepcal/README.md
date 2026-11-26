# DeepCal AI – food scanner, diet planner, tracker + **voice & NLP assistant**

## Setup
```bash
cd deepcal
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env
python -m app.nlp.train          # re-trains the intent classifier for YOUR scikit-learn version (takes seconds)
```
Open `.env`, paste your key from https://aistudio.google.com/apikey and set a long random `SECRET_KEY`.

## Run
```bash
uvicorn app.main:app --reload --port 8080
```
Open http://localhost:8080 → Sign up → **Voice** tab. (Mic needs `localhost` or https; browser speech recognition works in Chrome/Edge.)

## What's new
| Feature | Where |
|---|---|
| Mic + spoken replies (Web Speech API), English / Hindi / Hinglish | `static/index.html` (Voice tab) |
| `POST /api/voice`: text → intent → action → reply | `app/voice.py` |
| Review card: edit grams / meal / remove items, then **Confirm** (`POST /api/meals/confirm`) | Voice tab |
| `voice_logs` table: transcript, parsed JSON, what the user finally saved (`corrected_json`) | `app/core.py` |
| Intents: `log_meal`, `today_summary`, `remaining_calories`, `suggest_meal`, `edit_plan`, `set_preference`, `weekly_insight`, `chat` | `app/voice.py` |
| Indian foods table (104 foods) + units table (katori, glass, roti, scoop…) + alias matching (rapidfuzz) | `data/foods.csv`, `data/units.csv`, `app/nlp/foodmatch.py` |
| Hinglish numbers & words (ek, do, dedh, aadha, subah, maine…) | `app/nlp/extractor.py` |
| **Your own** intent classifier (TF-IDF char+word n-grams + Logistic Regression), Gemini only as fallback | `app/nlp/intents.py`, `train.py` |
| Rule-based food extractor (food, quantity, unit, meal time) | `app/nlp/extractor.py` |
| Voice plan editing (“swap Wednesday’s dinner for something vegetarian”) with confirm | `h_edit_plan`, `/api/plan/confirm` |
| Preference extraction (“allergic to peanuts, don’t eat onion”) → saved in profile JSON, used in plans + warnings | `app/nlp/preferences.py` |
| Photo + voice correction (“it was two rotis”) | Scan tab → 🎤 button, `/api/scan/correct` |
| Weekly insight (SQL tools → Gemini explains; can be read aloud) | Dashboard card, `/api/insights/weekly` |
| Coach chat with fixed tools (Gemini picks from a whitelist; it never writes SQL) | `app/tools.py` |
| Record-audio mode (MediaRecorder → WAV → Gemini transcription) for better Hinglish and for the mobile app | `/api/transcribe` |
| Rate limit on voice endpoints (default 20/min/user) | `app/core.py` |
| PWA (installable): manifest, service worker, icons | `static/` |
| Tests + evaluation scripts | `tests/`, `app/nlp/evaluate.py` |

## How a voice request flows
```
speech → text → classifier (confidence ≥ 0.7?) ── yes → intent
                                   └─ no → Gemini parses intent (JSON, validated by Pydantic)
log_meal: regex extractor → foods table (rapidfuzz) → Gemini only for foods not in the table → review card → Confirm → meals table
others:   fixed SQL functions (app/tools.py) → reply text → speech
```

## Numbers you can quote (and their limits)
Run `python -m app.nlp.train` and `python -m app.nlp.evaluate`; results are saved to `models/metrics.json`.
With the shipped data (267 labelled sentences, 8 intents, 80/20 split):
- Intent classifier holdout accuracy: **90.7%** (5-fold CV: 88.4%). At confidence ≥ 0.7 it answers ~48% of test sentences alone, at 100% accuracy on those; the rest go to Gemini.
- Rule extractor: 52/52 on `data/extraction_test.csv` (32 Hinglish, 20 English).

**Be honest in interviews:** both datasets are small and were written by the same person who wrote the rules, so these numbers are optimistic. To make them real: use the app for a couple of weeks, then export `voice_logs` (transcript + `corrected_json`), label them, add them to `data/intents.csv` / `extraction_test.csv`, retrain, and report accuracy on sentences the model never saw. `python -m app.nlp.evaluate --gemini` adds a Gemini-only extraction comparison (needs your key).

## Important notes
- **Food data is approximate.** `data/foods.csv` holds typical per-100 g values (cooked/prepared) compiled from general reference ranges, not copied from a licensed dataset. For accuracy, load values from the NIN *Indian Food Composition Tables (IFCT 2017)* or an open Indian dataset (check its licence) into the `foods` table. Rows you add/edit in SQLite are never overwritten (`INSERT OR IGNORE`).
- **Portion sizes are editable** in the `units` table. A katori is 150 g and a bowl of dal is also set to 150 g so both phrases give the same result; a bowl of rice is 200 g.
- Fuzzy matches below 95% are flagged “check: heard …” on the review card (e.g. “peanut chaat” → peanuts), so wrong matches are caught before saving.
- Allergy warnings use the `tags` column of `foods.csv` (best effort: mixed dishes may contain ingredients not tagged). Never rely on them for a serious allergy.
- Nutrition replies are general guidance, not medical advice.
- Deployment: SQLite files are lost on hosts with ephemeral disks. Set `DB_PATH` to a persistent volume or move to PostgreSQL; keep the Gemini key on the server only, serve over HTTPS.

## Tests
```bash
pytest -q          # tests/test_nlp.py (offline) + tests/test_api.py (needs requirements installed)
```

## Turning it into an app (Step 14)
1. **PWA** is already in: open the deployed HTTPS site on Android Chrome → *Install app*.
2. **Deploy**: `Procfile` (Render/Railway) or `Dockerfile` (Fly.io) are included. Use a persistent disk for `DB_PATH`.
3. **Capacitor** (optional, later): `npm i @capacitor/core @capacitor/cli @capacitor/android`, `npx cap init`, set `server.url` to your deployed site (or copy `static/` to `www`), `npx cap add android`. Inside Android WebViews the browser speech API is unreliable, so switch the Voice tab to **Record + Gemini** (already implemented).
