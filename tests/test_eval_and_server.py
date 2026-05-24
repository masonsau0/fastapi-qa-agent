"""Tests for eval metrics and the FastAPI server (mock backend).

The metric tests are the most important ones — if scoring is broken, the
benchmark numbers in the README are wrong.
"""

from __future__ import annotations

import os

import pytest

from src.eval.run_eval import file_citation_accuracy, keyword_recall

# ---------- keyword_recall ----------


def test_keyword_recall_all_present():
    assert (
        keyword_recall("Carmack Amendment, 9 months, 49 CFR", ["Carmack", "9 months", "49 CFR"])
        == 1.0
    )


def test_keyword_recall_partial():
    score = keyword_recall("Carmack Amendment only", ["Carmack", "9 months", "49 CFR"])
    assert score == pytest.approx(1 / 3)


def test_keyword_recall_case_insensitive():
    assert keyword_recall("FASTAPI uses pydantic", ["fastapi", "Pydantic"]) == 1.0


def test_keyword_recall_empty_keywords_is_zero():
    assert keyword_recall("anything", []) == 0.0


# ---------- file_citation_accuracy ----------


def test_file_citation_all_expected():
    answer = "See fastapi/routing.py:120-180 and fastapi/applications.py for context."
    expected = ["fastapi/routing.py", "fastapi/applications.py"]
    assert file_citation_accuracy(answer, expected) == 1.0


def test_file_citation_some_wrong():
    # answer cites two files; one is in expected, one isn't.
    answer = "See fastapi/routing.py:1-10 and some/unrelated/file.py for context."
    expected = ["fastapi/routing.py"]
    assert file_citation_accuracy(answer, expected) == 0.5


def test_file_citation_no_citations():
    # No citations at all → 0.0 (we want to penalize uncited answers).
    assert file_citation_accuracy("Just a vague answer.", ["fastapi/routing.py"]) == 0.0


def test_file_citation_strips_line_numbers():
    answer = "See fastapi/routing.py:1-20 and fastapi/routing.py:50-60."
    expected = ["fastapi/routing.py"]
    # Both citations point to the same file → 100% accurate.
    assert file_citation_accuracy(answer, expected) == 1.0


# ---------- Server ----------


@pytest.fixture(scope="module")
def client():
    os.environ["USE_MOCK"] = "true"
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_mock(client):
    r = client.post("/ask", json={"question": "How does routing work?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "tool_trace" in body
    assert "MOCK" in body["answer"]


def test_ask_rejects_short_question(client):
    r = client.post("/ask", json={"question": "a"})
    assert r.status_code == 422  # pydantic min_length


def test_ask_rejects_huge_question(client):
    r = client.post("/ask", json={"question": "x" * 5000})
    assert r.status_code == 422
