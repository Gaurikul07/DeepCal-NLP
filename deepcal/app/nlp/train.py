"""Train + evaluate the intent classifier.   Usage (from project root):   python -m app.nlp.train
Writes models/intent_model.joblib and models/metrics.json (accuracy, per-intent F1, confusion matrix, confidence coverage)."""
import csv, json, sys
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parents[2]

def build():
    # char_wb n-grams cope with Hinglish, where one word has many spellings (daal/dal, khaya/khaaya);
    # word 1-2 grams add phrase cues ("how many", "left today") so more sentences clear the confidence threshold.
    feats = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                       TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True))
    return make_pipeline(feats, LogisticRegression(max_iter=3000, C=30))

def main(threshold=0.7):
    rows = list(csv.DictReader(open(ROOT / "data" / "intents.csv", encoding="utf-8")))
    X, y = [r["text"].lower() for r in rows], [r["intent"] for r in rows]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build().fit(Xtr, ytr)
    pred = model.predict(Xte); probs = model.predict_proba(Xte).max(axis=1)
    acc = accuracy_score(yte, pred)
    labels = sorted(set(y))
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    cv = cross_val_score(build(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean()
    conf = probs >= threshold                                     # hybrid: model answers when confident, Gemini otherwise
    cov = float(conf.mean())
    acc_conf = float(sum(p == t for p, t, c in zip(pred, yte, conf) if c) / max(conf.sum(), 1))
    print(classification_report(yte, pred, zero_division=0))
    print("labels:", labels); print(confusion_matrix(yte, pred, labels=labels))
    print(f"holdout accuracy {acc:.3f} | 5-fold CV {cv:.3f} | model handles {cov:.0%} of test sentences at conf>={threshold} with accuracy {acc_conf:.3f}")
    (ROOT / "models").mkdir(exist_ok=True)
    json.dump({"n_train": len(Xtr), "n_test": len(Xte), "holdout_accuracy": round(acc, 4), "cv5_accuracy": round(float(cv), 4),
               "threshold": threshold, "model_coverage": round(cov, 4), "accuracy_when_confident": round(acc_conf, 4),
               "per_intent": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score")} for k, v in rep.items() if k in labels},
               "labels": labels, "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist()},
              open(ROOT / "models" / "metrics.json", "w"), indent=2)
    joblib.dump(build().fit(X, y), ROOT / "models" / "intent_model.joblib")     # final model uses all labelled data
    print("saved models/intent_model.joblib and models/metrics.json")

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.7)
