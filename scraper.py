"""Pobieranie recenzji gier ze Steama.

Korzystam z oficjalnej metody Web API IUserReviewsService/GetAppReviews
(dokumentacja: https://partner.steamgames.com/doc/webapi/IUserReviewsService).
Stary adres store.steampowered.com/appreviews/<appid>?json=1 został
wycofany przez Valve 22.10.2026, dlatego go nie używam.
Klucz API nie jest potrzebny, ale anonimowe zapytania mają limit (HTTP 429),
więc robię przerwy między stronami i ograniczam liczbę żądań.
"""
import json
import logging
import time
from datetime import datetime

import httpx

REVIEWS_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
USER_AGENT = "edt-sentiment-monitor/1.0 (projekt studencki, python-httpx)"

# typy recenzji wg dokumentacji API
REVIEW_TYPES = {"all": 0, "positive": 1, "negative": 2}
# sortowanie: 0 = najbardziej pomocne, 1 = najnowsze
FILTER_RECENT = 1

MAX_REQUESTS = 50      # twardy limit żądań na jedno wywołanie fetch_reviews
PAUSE_SECONDS = 1.0    # przerwa między kolejnymi stronami

log = logging.getLogger(__name__)


class SteamError(Exception):
    """Błąd pobierania danych ze Steama (pokazywany potem w GUI)."""


def make_client() -> httpx.Client:
    return httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT})


def get_json(client: httpx.Client, url: str, params: dict, retries: int = 3) -> dict:
    """GET z ponawianiem (backoff 1 s, 2 s, 4 s) dla timeoutów, 429 i błędów 5xx."""
    delay = 1.0
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as e:
            last_error = e
            log.warning("Timeout (próba %d/%d): %s", attempt, retries, url)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            # 4xx (poza 429) nie ma sensu ponawiać - np. złe appid
            if code != 429 and code < 500:
                raise SteamError(f"Serwer zwrócił błąd HTTP {code}") from e
            last_error = e
            log.warning("HTTP %d (próba %d/%d)", code, attempt, retries)
        except httpx.HTTPError as e:
            # brak sieci, zerwane połączenie itp.
            last_error = e
            log.warning("Błąd sieci (próba %d/%d): %s", attempt, retries, e)
        except json.JSONDecodeError as e:
            raise SteamError("Odpowiedź serwera nie jest poprawnym JSON-em") from e
        time.sleep(delay)
        delay *= 2
    raise SteamError(f"Nie udało się pobrać danych po {retries} próbach ({last_error})")


def parse_review(raw: dict, appid: int) -> dict:
    author = raw.get("author", {})
    return {
        "review_id": raw.get("recommendationid"),
        "appid": appid,
        "text": raw.get("review", ""),
        "voted_up": bool(raw.get("voted_up")),
        "created": datetime.fromtimestamp(raw.get("timestamp_created", 0)),
        "playtime_h": round(author.get("playtime_at_review", 0) / 60, 1),
        "votes_up": raw.get("votes_up", 0),
    }


def fetch_reviews(appid: int, n: int = 100, review_type: str = "all",
                  language: str = "english", client: httpx.Client | None = None,
                  max_requests: int = MAX_REQUESTS) -> list[dict]:
    """Pobiera do n najnowszych recenzji gry (stronicowanie kursorem, max 100 na stronę)."""
    own_client = client is None
    if own_client:
        client = make_client()
    reviews = []
    seen_ids = set()
    cursor = "*"
    try:
        for i in range(max_requests):
            query = {
                "appid": int(appid),
                "filter": FILTER_RECENT,
                "languages": [language],
                "review_type": REVIEW_TYPES[review_type],
                "purchase_type": 1,  # wszystkie zakupy, nie tylko na Steamie
                "num_per_page": min(100, n - len(reviews)),
                "cursor": cursor,
            }
            data = get_json(client, REVIEWS_URL, {"input_json": json.dumps(query)})
            batch = data.get("response", {}).get("reviews", [])
            new = [r for r in batch if r.get("recommendationid") not in seen_ids]
            if not new:
                break  # koniec recenzji albo kursor się powtarza
            for raw in new:
                seen_ids.add(raw.get("recommendationid"))
                reviews.append(parse_review(raw, int(appid)))
            if len(reviews) >= n:
                break
            cursor = data["response"].get("cursor", "")
            if not cursor:
                break
            time.sleep(PAUSE_SECONDS)
    finally:
        if own_client:
            client.close()
    return reviews[:n]


def search_games(term: str, client: httpx.Client | None = None) -> list[dict]:
    """Wyszukiwanie gry po nazwie (publiczne API wyszukiwarki sklepu)."""
    own_client = client is None
    if own_client:
        client = make_client()
    try:
        data = get_json(client, SEARCH_URL, {"term": term, "l": "english", "cc": "us"})
    finally:
        if own_client:
            client.close()
    return [{"appid": item["id"], "name": item["name"]} for item in data.get("items", [])]
