# EDT Projekt 1 — Inteligentny System Monitorowania Opinii

Projekt zaliczeniowy z przedmiotu **Eksploracja Danych Tekstowych** (Sem. 2, 2025/2026).

## Opis

Aplikacja webowa monitoruje sentyment w artykułach i komentarzach z portalu **HackerNews** w czasie rzeczywistym. Pobiera aktualne top stories, analizuje ich tytuły i komentarze modelem DistilBERT i wyświetla wyniki w przejrzystym dashboardzie.

## Jak działa

1. **Scraping** — dane pobierane są z [HackerNews API](https://github.com/HackerNews/API) (publiczne, bez klucza)
2. **Model** — analiza sentymentu opiera się na modelu `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face), wytrenowanym na zbiorze SST-2 (recenzje filmowe)
3. **GUI** — interfejs Streamlit z tabelą wyników, wykresem i widokiem komentarzy

## Stack technologiczny

| Komponent | Technologia |
|-----------|-------------|
| GUI | Streamlit >= 1.40 |
| Model NLP | Transformers (DistilBERT SST-2) |
| DL framework | PyTorch >= 2.3 |
| HTTP client | httpx >= 0.27 |
| Parsing HTML | BeautifulSoup4 |
| Dane live | HackerNews Firebase API |

## Struktura projektu

```
edt-sentiment-monitor/
├── app.py           # Streamlit GUI — główna aplikacja
├── scraper.py       # HackerNews API scraper
├── sentiment.py     # Pipeline analizy sentymentu (DistilBERT)
├── requirements.txt # Zależności Python
└── README.md
```

## Uruchomienie

### 1. Utwórz środowisko wirtualne i zainstaluj zależności

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Uruchom aplikację

```bash
streamlit run app.py
```

Aplikacja otworzy się pod adresem **http://localhost:8501**

> **Uwaga:** Przy pierwszym uruchomieniu model (~260 MB) zostanie automatycznie pobrany z Hugging Face Hub i zapisany w `~/.cache/huggingface/`. Kolejne uruchomienia są szybkie.

## Funkcje interfejsu

- **Przycisk "Pobierz najnowsze"** — pobiera aktualne top stories z HN i analizuje sentyment tytułów
- **Metryki** — łączna liczba stories, podział POSITIVE/NEGATIVE, średnia pewność modelu
- **Wykres** — rozkład sentymentów
- **Tabela** — lista stories z etykietą sentymentu i poziomem pewności
- **Filtr** — pokazuj wszystkie / tylko POSITIVE / tylko NEGATIVE
- **Komentarze** — kliknij story, by rozwinąć komentarze z ich sentymentami
- **Timestamp** — czas ostatniego odświeżenia

## Źródła danych

- **Dane live (predykcja):** HackerNews Top Stories API — `https://hacker-news.firebaseio.com/v0/topstories.json`
- **Model (trening):** DistilBERT wytrenowany na zbiorze Stanford Sentiment Treebank v2 (SST-2), dostępny na [Hugging Face Hub](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

## Autor

