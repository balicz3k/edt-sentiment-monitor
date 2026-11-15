"""Porównanie własnego modelu z gotowym transformerem na tym samym zbiorze testowym.

Wynik: results/transformer_comparison.json, results/confusion_matrix_transformer.png
Uruchomienie: python scripts/compare_transformer.py [--sample 1000] [--cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sentiment import OwnModel, TransformerModel  # noqa: E402
from train_model import RESULTS_DIR, evaluate, load_clean, save_confusion_matrix  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="0 = cały zbiór testowy")
    parser.add_argument("--cpu", action="store_true", help="wymuś CPU (np. gdy GPU zajęte)")
    args = parser.parse_args()

    test = load_clean("test")
    if args.sample:
        test = test.sample(args.sample, random_state=42)
    texts = test["text"].tolist()
    y = test["label"].to_numpy()

    own = OwnModel()
    t0 = time.time()
    own_labels, _ = own.predict(texts)  # liczę razem z czyszczeniem - tak jak w GUI
    own_time = time.time() - t0

    trans = TransformerModel(device="cpu" if args.cpu else None)
    t0 = time.time()
    tr_labels, tr_conf = trans.predict(texts)
    tr_time = time.time() - t0
    # SST-2 zwraca pewność dla przewidzianej klasy - przeliczam na P(pozytywna) do AUC
    tr_score = [c if lab == 1 else 1 - c for lab, c in zip(tr_labels, tr_conf)]

    result = {
        "n_test": len(test),
        "device_transformer": trans.device,
        "tfidf_logreg": {**evaluate(y, own_labels),
                         "ms_per_text": round(1000 * own_time / len(test), 2)},
        "distilbert_sst2": {**evaluate(y, tr_labels, tr_score),
                            "ms_per_text": round(1000 * tr_time / len(test), 2)},
        "agreement": round(float((own_labels == tr_labels).mean()), 4),
    }
    (RESULTS_DIR / "transformer_comparison.json").write_text(json.dumps(result, indent=2))
    save_confusion_matrix(y, tr_labels, "DistilBERT SST-2 (test)",
                          RESULTS_DIR / "confusion_matrix_transformer.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
