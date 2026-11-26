"""API-level test (needs `pip install -r requirements.txt`). Gemini is never called: the rule path + classifier handle these sentences."""
import os, sys, tempfile
from pathlib import Path
os.environ["DB_PATH"] = tempfile.mktemp(suffix=".db")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app


def test_voice_log_confirm_and_summary():
    with TestClient(app) as c:
        tok = c.post("/api/register", json={"name": "T", "email": "t@example.com", "password": "secret1"}).json()["token"]
        h = {"Authorization": "Bearer " + tok}
        r = c.post("/api/voice", json={"text": "I ate two rotis and a bowl of dal for lunch"}, headers=h).json()
        assert r["intent"] == "log_meal" and r["card"]["type"] == "meal" and r["card"]["items"]
        assert c.get("/api/summary", headers=h).json()["meals"] == []               # nothing saved before confirmation
        ok = c.post("/api/meals/confirm", json={"log_id": r["log_id"], "meal": r["card"]["meal"], "items": r["card"]["items"]}, headers=h)
        assert ok.json()["ok"]
        assert len(c.get("/api/summary", headers=h).json()["meals"]) == len(r["card"]["items"])
        s = c.post("/api/voice", json={"text": "how much protein have I had today"}, headers=h).json()
        assert "protein" in s["reply"]


def test_preferences_survive_profile_save():
    with TestClient(app) as c:
        tok = c.post("/api/register", json={"name": "P", "email": "p@example.com", "password": "secret1"}).json()["token"]
        h = {"Authorization": "Bearer " + tok}
        c.post("/api/voice", json={"text": "I'm allergic to peanuts and don't eat onion"}, headers=h)
        c.post("/api/profile", json={"age": 22, "weight": 60, "height": 165}, headers=h)
        assert "peanut" in c.get("/api/preferences", headers=h).json()["allergies"]


def test_rate_limit():
    with TestClient(app) as c:
        tok = c.post("/api/register", json={"name": "R", "email": "r@example.com", "password": "secret1"}).json()["token"]
        h = {"Authorization": "Bearer " + tok}
        codes = [c.post("/api/voice", json={"text": "log one banana"}, headers=h).status_code for _ in range(25)]
        assert 429 in codes
