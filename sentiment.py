"""Analiza sentymentu używając pre-trained distilbert-sst2."""
import sys
from transformers import pipeline

_model = None


def get_pipeline():
    """Inicjalizuje i zwraca pipeline sentymentu (singleton)."""
    global _model
    if _model is None:
        _model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            revision="main",
        )
    return _model


def analyze(texts: list[str]) -> list[dict]:
    """
    Analizuje listę tekstów.
    Zwraca listę słowników {label: 'POSITIVE'/'NEGATIVE', score: float}.
    Puste teksty są pomijane i zastępowane neutralnym wynikiem.
    """
    if not texts:
        return []

    pipe = get_pipeline()

    results = []
    for text in texts:
        if text and text.strip():
            try:
                result = pipe(text.strip(), truncation=True, max_length=512)
                results.append(result[0])
            except Exception as e:
                print(f"Warning: skipping item due to {type(e).__name__}: {e}", file=sys.stderr)
                results.append({"label": "NEUTRAL", "score": 0.5})
        else:
            results.append({"label": "NEUTRAL", "score": 0.5})

    return results
