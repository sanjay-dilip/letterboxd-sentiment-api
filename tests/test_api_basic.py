from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_analyze_text_basic():
    payload = {"text": "I loved this movie so much, it was beautiful."}
    resp = client.post("/analyze-text", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sentiment" in data
    assert "probability" in data
    assert "all_probs" in data

def test_analyze_text_empty():
    payload = {"text": "   "}
    resp = client.post("/analyze-text", json=payload)
    assert resp.status_code == 400

def test_movie_summary_not_found():
    payload = {"movie": "This Movie Does Not Exist 12345"}
    resp = client.post("/movie-summary", json=payload)
    assert resp.status_code in (200, 404)  # depending on how you want to handle