"""Pobiera zbiór treningowy: recenzje Steam (EN) z gier z games.TRAIN_GAMES.

Dla każdej gry osobno pobieram recenzje pozytywne i negatywne, żeby zbiór
był zbalansowany (na Steamie ok. 80% recenzji jest pozytywnych).
Wynik: data/reviews_raw.csv oraz podział data/train.csv, val.csv, test.csv.

Uruchomienie (z katalogu projektu):
    python scripts/download_data.py            # 400 + 400 recenzji na grę
    python scripts/download_data.py --per-class 100   # szybka wersja
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from games import TRAIN_GAMES  # noqa: E402
from scraper import SteamError, fetch_reviews, make_client  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
MIN_WORDS = 3      # krótsze recenzje ("good", "10/10") są zbyt niejednoznaczne
MAX_CHARS = 5000   # bardzo długie recenzje to często ASCII-art albo kopiowane teksty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=400,
                        help="liczba recenzji pozytywnych i negatywnych na grę")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    rows = []
    with make_client() as client:
        for appid, name in TRAIN_GAMES.items():
            for review_type in ("positive", "negative"):
                try:
                    reviews = fetch_reviews(appid, n=args.per_class, review_type=review_type,
                                            client=client)
                except SteamError as e:
                    print(f"  pomijam {name} ({review_type}): {e}")
                    continue
                for r in reviews:
                    r["game"] = name
                rows.extend(reviews)
                print(f"{name:28s} {review_type:8s} {len(reviews):4d}")

    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").str.strip()
    df = df.drop_duplicates(subset="text")
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    df = df[df["text"].str.len() <= MAX_CHARS]
    df["label"] = df["voted_up"].astype(int)  # 1 = pozytywna, 0 = negatywna
    df.to_csv(DATA_DIR / "reviews_raw.csv", index=False)

    # podział 70/15/15, stratyfikacja po etykiecie, stałe ziarno
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    for part, name in ((train, "train"), (val, "val"), (test, "test")):
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} recenzji, pozytywnych {part['label'].mean():.1%}")


if __name__ == "__main__":
    main()
