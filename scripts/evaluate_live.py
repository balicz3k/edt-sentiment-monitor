"""Ocena obu modeli na świeżych recenzjach z Internetu (gry spoza treningu).

Steam podaje, czy autor poleca grę (voted_up) - traktuję to jako prawdziwą
etykietę. Dla każdej gry z games.PREDICT_GAMES pobieram po N recenzji
pozytywnych i negatywnych. Wynik: results/live_eval.csv
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from games import PREDICT_GAMES  # noqa: E402
from scraper import SteamError, fetch_reviews, make_client  # noqa: E402
from sentiment import OwnModel, TransformerModel  # noqa: E402
from train_model import RESULTS_DIR, evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = []
    with make_client() as client:
        for appid, name in PREDICT_GAMES.items():
            for review_type in ("positive", "negative"):
                try:
                    batch = fetch_reviews(appid, n=args.per_class, review_type=review_type,
                                          client=client)
                except SteamError as e:
                    print(f"pomijam {name}: {e}")
                    continue
                for r in batch:
                    r["game"] = name
                rows.extend(batch)
    df = pd.DataFrame(rows)

    own = OwnModel()
    df["clean"] = own.clean(df["text"].tolist())
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)  # puste teksty pomijam
    df["own"], _ = own.predict_clean(df["clean"].tolist())
    df["transformer"], _ = TransformerModel(device="cpu" if args.cpu else None).predict(
        df["text"].tolist())
    df["y"] = df["voted_up"].astype(int)

    out = []
    for game, part in list(df.groupby("game")) + [("RAZEM", df)]:
        for model in ("own", "transformer"):
            m = evaluate(part["y"], part[model])
            out.append({"gra": game, "model": model, "n": len(part),
                        "accuracy": m["accuracy"], "f1": m["f1"]})
    res = pd.DataFrame(out)
    res["data_pobrania"] = date.today().isoformat()
    res.to_csv(RESULTS_DIR / "live_eval.csv", index=False)
    print(res.pivot(index="gra", columns="model", values="f1").round(3))


if __name__ == "__main__":
    main()
