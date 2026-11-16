# EDT Projekt 1 - Inteligentny system monitorowania opinii

Projekt z przedmiotu Eksploracja Danych Tekstowych (sem. 2, 2026/2027).
Aplikacja pobiera na żywo recenzje gier ze Steama, czyści tekst, ocenia sentyment
(pozytywny / negatywny) własnym modelem TF-IDF + regresja logistyczna albo gotowym
transformerem i pokazuje wyniki w GUI (Streamlit).

## Wymagania projektu i gdzie są spełnione

| Wymaganie | Realizacja |
|---|---|
| pobieranie tekstów z Internetu (API / scraping) | `scraper.py` - oficjalne Web API Steama, w GUI na żywo |
| automatyczna analiza sentymentu poz./neg. | `sentiment.py`, model trenowany w `train_model.py` |
| prosty GUI | `app.py` (Streamlit) |
| dane treningowe i do predykcji spójne | recenzje Steam po angielsku: trening na 30 grach, predykcja na innych grach |

## Dane

**Źródło: recenzje gier ze Steama (język angielski).** Wybrałem je, bo:
- każda recenzja ma etykietę od autora ("polecam" / "nie polecam"), więc mogę wytrenować
  model i sprawdzić trafność także na danych pobranych na żywo,
- dane treningowe i do predykcji mają tę samą strukturę i charakter (krótkie, potoczne
  opinie graczy), czego wymaga treść projektu,
- API jest publiczne i nie wymaga klucza.

Pobieranie: metoda `IUserReviewsService/GetAppReviews` z oficjalnego Web API Steama
([dokumentacja](https://partner.steamgames.com/doc/webapi/IUserReviewsService)).
Starszy adres `store.steampowered.com/appreviews/<appid>?json=1` Valve wyłączyło
22.10.2026 (zwraca przekierowanie na stronę główną), dlatego go nie używam.
`api.steampowered.com/robots.txt` blokuje roboty indeksujące, ale ta metoda jest
udokumentowanym interfejsem dla programów; ograniczam się do małej liczby zapytań
(przerwa 1 s między stronami, limit 50 żądań na wywołanie, ponawianie z backoffem
przy HTTP 429/5xx). Wyszukiwarka gier w GUI korzysta z `store.steampowered.com/api/storesearch/`
(ścieżka dozwolona w robots.txt). Nagłówek User-Agent jest ogólny, bez danych osobowych.

- **trening** (`scripts/download_data.py`): 30 popularnych gier (`games.py: TRAIN_GAMES`),
  po 400 najnowszych recenzji pozytywnych i 400 negatywnych na grę (zbiór zbalansowany,
  bo na Steamie ok. 80% recenzji jest pozytywnych). Po usunięciu duplikatów, recenzji
  krótszych niż 3 słowa i dłuższych niż 5000 znaków zostaje 17 923 recenzji,
  podzielonych 70/15/15 (train/val/test, ziarno 42, stratyfikacja po etykiecie).
- **predykcja** (GUI i `scripts/evaluate_live.py`): inne gry (`PREDICT_GAMES`) albo
  dowolna gra wyszukana po nazwie.

Dane nie są w repozytorium (`data/` w `.gitignore`) - pobiera je skrypt.
Recenzje są treściami użytkowników Steama; używam ich wyłącznie do celów dydaktycznych
i nie rozpowszechniam.

## Instalacja i uruchomienie

Sprawdzone na Pythonie 3.12 (Kubuntu, czysty venv).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm    # lematyzacja w porównaniu w notatniku
```

Gotowy model `models/tfidf_logreg.joblib` (0,6 MB) jest w repozytorium, więc aplikację
można uruchomić od razu:

```bash
streamlit run app.py
```

Pełne odtworzenie wyników (ok. 10 minut, w tym kilka minut transformera na CPU):

```bash
python scripts/download_data.py          # ok. 5 min, dane do data/
python train_model.py                     # model + results/metrics.json, macierz pomyłek
python scripts/compare_transformer.py     # porównanie z DistilBERT (opcja --cpu)
python scripts/evaluate_live.py           # ocena na świeżych recenzjach z innych gier
pytest -q                                 # testy czyszczenia tekstu
jupyter notebook notebooks/01_model.ipynb # analiza i wnioski
```

Przy pierwszym użyciu transformera biblioteka `transformers` pobiera model
(ok. 270 MB) do `~/.cache/huggingface`.

## Struktura

```
app.py                     GUI (Streamlit)
scraper.py                 pobieranie recenzji (Steam Web API), wyjątki, retry, limit żądań
games.py                   listy gier: treningowe i do predykcji
text_cleaning.py           czyszczenie tekstu (pipeline z zajęć)
train_model.py             TF-IDF + LogReg, dobór C, ewaluacja, zapis modelu
sentiment.py               wspólny interfejs: własny model i DistilBERT
scripts/download_data.py   zbiór treningowy
scripts/compare_transformer.py, scripts/evaluate_live.py
notebooks/01_model.ipynb   eksperymenty i wnioski
results/                   metryki, macierze pomyłek, wagi słów, ocena na żywo
tests/test_cleaning.py     testy czyszczenia
docs/NAUKA.md              opis "po ludzku" + pytania na prezentację
docs/prezentacja.pdf       slajdy
```

## Czyszczenie tekstu (`text_cleaning.py`)

1. usunięcie HTML i BBCode (`[b]`, `[spoiler]` - recenzje Steam używają BBCode),
2. usunięcie adresów URL i e-maili,
3. rozwinięcie negacji: `don't` -> `do not`, `can't` -> `can not`,
4. małe litery,
5. tokenizacja: słowa, liczby, oceny typu `2/10` jako jeden token, `!` i `?`
   (powtórzenia zwinięte do jednego), emoji,
6. usunięcie stopwords **bez słów negacji** (`not`, `no`, `never`, `nor`, `without`...),
7. stemming (Porter).

Wybór stemmingu: w notatniku porównałem brak normalizacji, stemming i lematyzację spaCy
na zbiorze walidacyjnym. Różnice są małe, stemming miał najwyższe F1 i jest kilka razy
szybszy od lematyzacji. Usunięcie negacji razem ze stopwords obniża F1 - to pułapka
z Prezentacji 2 ("To nie jest dobry film" -> "dobry film").

## Model i wyniki

TF-IDF z unigramami i bigramami (`min_df=2`, `sublinear_tf`), regresja logistyczna.
Parametr C dobrany 5-krotną walidacją krzyżową na zbiorze treningowym (F1) - wyszło C = 3.

Zbiór testowy (2 689 recenzji, `results/metrics.json`, `results/transformer_comparison.json`):

| model | accuracy | precision | recall | F1 | ROC AUC | czas [ms/tekst] |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg (własny) | 0,819 | 0,811 | 0,766 | **0,788** | 0,903 | 0,22 |
| TF-IDF + LinearSVC | 0,808 | 0,790 | 0,768 | 0,779 | - | - |
| TF-IDF + Naive Bayes | 0,788 | 0,890 | 0,592 | 0,711 | - | - |
| DistilBERT SST-2 (gotowy) | 0,807 | 0,845 | 0,687 | 0,758 | 0,884 | 92 (CPU) |

Recenzje pobrane na żywo z 9 gier spoza treningu (1 768 recenzji, 16.11.2026,
`results/live_eval.csv`): F1 własnego modelu 0,831, DistilBERT 0,818.

Wnioski: własny model jest nieco lepszy od transformera, bo uczył się na tej samej
domenie (żargon graczy: "refund", "unplayable", "10/10", "peak"), a DistilBERT był
dostrajany na zdaniach z recenzji filmów. Jest też ok. 400 razy szybszy na CPU.
Transformer ma wyższą precyzję, ale częściej oznacza pozytywne recenzje jako negatywne.

## GUI (`streamlit run app.py`)

- **Analiza recenzji**: wybór gry (lista, wyszukiwanie po nazwie albo appid), liczby
  i rodzaju recenzji oraz modelu. Wynik: odsetek pozytywnych, porównanie z oceną autora
  na Steamie (accuracy, F1, macierz pomyłek), rozkład sentymentu, trend w kolejnych
  dniach, chmury słów osobno dla pozytywnych i negatywnych, tabela przykładów
  (z filtrem, np. tylko niezgodne ze Steamem), eksport CSV.
- **Pojedynczy tekst**: wynik każdego kroku czyszczenia i predykcja dla wpisanego zdania.
- **O modelu**: metryki, macierze pomyłek, ocena na żywo i słowa o największych wagach.

Puste po czyszczeniu recenzje są pomijane (ich liczba jest pokazana), a nie oznaczane
jako negatywne. Błędy sieci są pokazywane jako ostrzeżenie w GUI.

## Ograniczenia

- Etykieta "polecam" nie zawsze odpowiada wydźwiękowi tekstu: żarty, sarkazm
  ("0/10 would buy again"), recenzje chwalące grę, ale krytykujące wydawcę.
- Model jest tylko dla angielskiego i dla recenzji gier; na innych tekstach będzie słabszy.
- Bag-of-words nie widzi szyku słów dalej niż bigram (np. "not that bad at all").
- Część recenzji (np. ASCII-art) po czyszczeniu jest pusta i jest pomijana.
