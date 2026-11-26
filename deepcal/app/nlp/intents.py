"""Your own intent classifier (TF-IDF char n-grams + Logistic Regression). Gemini is only the fallback when confidence is low."""
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "intent_model.joblib"
INTENTS = ["log_meal", "today_summary", "remaining_calories", "suggest_meal", "edit_plan", "set_preference", "weekly_insight", "chat"]

_model = None
_tried = False
def load_model():
    global _model
    global _tried
    if _model is None and not _tried and MODEL_PATH.exists():
        _tried = True
        try: _model = joblib.load(MODEL_PATH)
        except Exception as e:      # e.g. scikit-learn version changed: retrain with `python -m app.nlp.train`
            print("Intent model could not be loaded, falling back to Gemini:", e)
    return _model

def predict(text: str):
    """-> (intent, confidence 0-1) or (None, 0.0) if no trained model exists yet (run `python -m app.nlp.train`)."""
    m = load_model()
    if m is None: return None, 0.0
    probs = m.predict_proba([text.lower()])[0]
    i = int(probs.argmax())
    return m.classes_[i], float(probs[i])
