"""Trening własnego modelu: TF-IDF (1-2-gramy) + regresja logistyczna.

Wymaga danych z scripts/download_data.py. Wynik:
  models/tfidf_logreg.joblib     - model (wektoryzator + klasyfikator)
  results/metrics.json           - metryki na zbiorze testowym
  results/confusion_matrix.png   - macierz pomyłek
  results/top_features.csv       - słowa o największych wagach
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from text_cleaning import clean_many

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "tfidf_logreg.joblib"
RESULTS_DIR = ROOT / "results"
SEED = 42

# Wybrane w notebooks/01_model.ipynb na podstawie F1 na zbiorze walidacyjnym
# (porównanie: bez normalizacji / stemming / lematyzacja). Różnice są małe,
# stemming wypadł minimalnie najlepiej i jest kilka razy szybszy od spaCy.
NORMALIZATION = "stem"


def load_clean(split: str, normalize: str | None = NORMALIZATION) -> pd.DataFrame:
    """Wczytuje podział i dokleja kolumnę 'clean'. Wynik czyszczenia trzymam w
    data/, bo lematyzacja wszystkich recenzji trwa ok. pół minuty."""
    cache = DATA_DIR / f"clean_{split}_{normalize}.csv"
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    if cache.exists():
        df["clean"] = pd.read_csv(cache, keep_default_na=False)["clean"]
    else:
        df["clean"] = clean_many(df["text"].tolist(), normalize=normalize)
        df[["clean"]].to_csv(cache, index=False)
    return df


def make_vectorizer() -> TfidfVectorizer:
    # tekst jest już oczyszczony i stokenizowany - dzielę po spacjach
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2), min_df=2, max_df=0.9,
                           sublinear_tf=True, max_features=100_000)


def evaluate(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def save_confusion_matrix(y_true, y_pred, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["negatywna", "pozytywna"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("predykcja")
    ax.set_ylabel("prawdziwa etykieta")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    train = load_clean("train")
    test = load_clean("test")
    print(f"train: {len(train)}, test: {len(test)}, normalizacja: {NORMALIZATION}")

    # dobór C walidacją krzyżową (5 foldów) tylko na zbiorze treningowym
    pipe = Pipeline([("tfidf", make_vectorizer()),
                     ("clf", LogisticRegression(max_iter=2000))])
    grid = GridSearchCV(pipe, {"clf__C": [0.1, 0.3, 1, 3, 10]}, scoring="f1",
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    t0 = time.time()
    grid.fit(train["clean"], train["label"])
    train_time = time.time() - t0
    model = grid.best_estimator_
    cv_table = {str(c): round(float(s), 4) for c, s in
                zip(grid.cv_results_["param_clf__C"], grid.cv_results_["mean_test_score"])}
    print("F1 w CV dla C:", cv_table, "-> najlepsze C =", grid.best_params_["clf__C"])

    t0 = time.time()
    proba = model.predict_proba(test["clean"])[:, 1]
    predict_time = time.time() - t0
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "normalization": NORMALIZATION,
        "n_train": len(train), "n_test": len(test),
        "best_C": grid.best_params_["clf__C"], "cv_f1_by_C": cv_table,
        "train_time_s": round(train_time, 1),
        "predict_ms_per_text": round(1000 * predict_time / len(test), 3),
        "tfidf_logreg": evaluate(test["label"], pred, proba),
    }

    # dwa proste modele dla porównania (te same cechy TF-IDF, parametry domyślne)
    for name, clf in (("tfidf_naive_bayes", MultinomialNB()),
                      ("tfidf_linear_svc", LinearSVC())):
        other = Pipeline([("tfidf", make_vectorizer()), ("clf", clf)])
        other.fit(train["clean"], train["label"])
        metrics[name] = evaluate(test["label"], other.predict(test["clean"]))

    for name in ("tfidf_logreg", "tfidf_naive_bayes", "tfidf_linear_svc"):
        m = metrics[name]
        print(f"{name:18s} acc={m['accuracy']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f}")

    save_confusion_matrix(test["label"], pred, "TF-IDF + LogReg (test)",
                          RESULTS_DIR / "confusion_matrix.png")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # słowa z największymi wagami - pokazują, czego model się nauczył
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    top = pd.DataFrame({
        "negatywne": vocab[order[:30]], "waga_neg": coef[order[:30]].round(2),
        "pozytywne": vocab[order[::-1][:30]], "waga_poz": coef[order[::-1][:30]].round(2),
    })
    top.to_csv(RESULTS_DIR / "top_features.csv", index=False)

    joblib.dump({"pipeline": model, "normalization": NORMALIZATION}, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Zapisano {MODEL_PATH.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
