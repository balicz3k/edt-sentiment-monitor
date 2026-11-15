"""Dwa modele sentymentu z tym samym interfejsem: predict(texts) -> (etykiety, pewności).

Etykieta 1 = pozytywny, 0 = negatywny. Puste teksty trzeba odfiltrować
wcześniej (robi to app.py) - nie zgaduję dla nich etykiety.
"""
from pathlib import Path

import joblib
import numpy as np

from text_cleaning import clean_many

MODEL_PATH = Path(__file__).resolve().parent / "models" / "tfidf_logreg.joblib"
TRANSFORMER_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LABEL_NAMES = {1: "pozytywny", 0: "negatywny"}


class OwnModel:
    """TF-IDF + regresja logistyczna wytrenowana na recenzjach Steam (train_model.py)."""
    name = "Własny: TF-IDF + LogReg"

    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Brak pliku {path.name}. Uruchom najpierw: python train_model.py")
        saved = joblib.load(path)
        self.pipeline = saved["pipeline"]
        self.normalization = saved["normalization"]

    def clean(self, texts: list[str]) -> list[str]:
        return clean_many(texts, normalize=self.normalization)

    def predict_clean(self, cleaned: list[str]):
        proba = self.pipeline.predict_proba(cleaned)[:, 1]
        labels = (proba >= 0.5).astype(int)
        confidence = np.where(labels == 1, proba, 1 - proba)
        return labels, confidence

    def predict(self, texts: list[str]):
        return self.predict_clean(self.clean(texts))


class TransformerModel:
    """Gotowy DistilBERT dostrojony na SST-2 (zdania z recenzji filmów).
    Dostaje surowy tekst - ma własny tokenizator i nie potrzebuje czyszczenia."""
    name = "Transformer: DistilBERT SST-2"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import pipeline
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipe = pipeline("sentiment-analysis", model=TRANSFORMER_NAME, device=device)

    def predict(self, texts: list[str], batch_size: int = 32):
        out = self.pipe(texts, truncation=True, max_length=512, batch_size=batch_size)
        labels = np.array([1 if o["label"] == "POSITIVE" else 0 for o in out])
        confidence = np.array([o["score"] for o in out])
        return labels, confidence
