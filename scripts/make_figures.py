"""Wykres do prezentacji z plików wyników (results/) -> docs/img/porownanie_modeli.png."""
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    out = ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((RES / "metrics.json").read_text())
    t = json.loads((RES / "transformer_comparison.json").read_text())
    live = pd.read_csv(RES / "live_eval.csv").query("gra == 'RAZEM'").set_index("model")["f1"]
    data = pd.DataFrame({
        "test (2 689 recenzji)": [m["tfidf_logreg"]["f1"], m["tfidf_linear_svc"]["f1"],
                                  m["tfidf_naive_bayes"]["f1"], t["distilbert_sst2"]["f1"]],
        "na żywo, inne gry": [live["own"], None, None, live["transformer"]],
    }, index=["TF-IDF + LogReg", "TF-IDF + LinearSVC", "TF-IDF + NB", "DistilBERT SST-2"])
    ax = data.plot.bar(figsize=(6.4, 3.4), rot=0, width=0.75)
    ax.set_ylim(0.6, 0.9); ax.set_ylabel("F1")
    ax.set_title("F1 modeli: zbiór testowy i recenzje pobrane na żywo")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    plt.xticks(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "porownanie_modeli.png", dpi=200); plt.close()
    print("zapisano docs/img/porownanie_modeli.png")


if __name__ == "__main__":
    main()
