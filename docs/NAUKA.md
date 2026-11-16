# Jak to działa - notatki do nauki i obrony

## 1. Przepływ danych w jednym zdaniu

Steam API -> lista recenzji (tekst + "polecam/nie polecam") -> czyszczenie tekstu ->
wektor TF-IDF -> regresja logistyczna -> prawdopodobieństwo, że recenzja jest pozytywna
-> wykresy, chmury słów i tabela w Streamlit.

## 2. Pobieranie danych (`scraper.py`)

- Wysyłam GET na `api.steampowered.com/IUserReviewsService/GetAppReviews/v1/`
  z parametrem `input_json` (appid gry, język, sortowanie "najnowsze", typ recenzji,
  liczba na stronę - max 100, kursor).
- Odpowiedź ma listę recenzji i `cursor` - przekazuję go w kolejnym zapytaniu, żeby dostać
  następną stronę (stronicowanie kursorem). Kończę, gdy mam tyle recenzji, ile trzeba,
  albo gdy nie przychodzi nic nowego.
- Obsługa błędów: `httpx.TimeoutException` i błędy sieci (`httpx.HTTPError`) oraz
  HTTP 429/5xx - ponawiam do 3 razy z rosnącą przerwą (1 s, 2 s, 4 s); inne 4xx
  (np. złe appid) - od razu błąd; `json.JSONDecodeError` - błąd "odpowiedź nie jest JSON-em".
  Wszystko zamieniam na `SteamError`, którą GUI pokazuje jako ostrzeżenie.
- Cache: `st.cache_data(ttl=600)` - te same parametry przez 10 minut nie wysyłają
  nowych zapytań.

## 3. Czyszczenie tekstu (`text_cleaning.py`)

Kolejność ma znaczenie:
1. HTML/BBCode, URL, e-maile -> spacja (to szum, nie opinia).
2. `don't` -> `do not` **przed** tokenizacją, inaczej "n't" zniknęłoby razem z interpunkcją.
3. Małe litery - dla bag-of-words "GREAT" i "great" to ta sama cecha. Pułapka z zajęć:
   tracimy "krzyk" pisany wielkimi literami (i informację dla NER), ale tutaj zysk
   z mniejszego słownika jest większy.
4. Tokenizacja wyrażeniem regularnym: słowa, liczby, `2/10` jako jeden token (ocena!),
   `!` i `?`, emoji. Nie usuwam "wszystkiego, co nie jest literą" - pułapki z prezentacji
   (waluty, emoji, liczby niosą informację).
5. Stopwords z listy scikit-learn, ale **bez negacji** (`not`, `no`, `never`, `nor`,
   `none`, `nothing`, `without`, `cannot`...). Usuwam je przed stemmingiem, bo stemmer
   zmienia "this" w "thi" i takie słowo nie pasowałoby już do listy.
6. Stemming Portera: obcina końcówki (`crashes` -> `crash`, `amazing` -> `amaz`).

**Stemming a lematyzacja.** Stemming tnie końcówki regułami - szybki, wynik nie musi być
słowem (`studies` -> `studi`). Lematyzacja (spaCy) sprowadza do formy słownikowej
z użyciem słownika i części mowy (`studies` -> `study`, `was` -> `be`) - dokładniejsza,
ale kilka razy wolniejsza. W notatniku na walidacji różnica F1 była mała, stemming
minimalnie lepszy, więc wybrałem stemming.

## 4. TF-IDF

Każda recenzja to wektor o długości słownika (tu słowa i pary słów - bigramy).
Wartość dla słowa t w dokumencie d:

    tfidf(t, d) = tf(t, d) * idf(t),   idf(t) = ln((1 + N) / (1 + df(t))) + 1

- `tf` - ile razy słowo jest w recenzji (u mnie `sublinear_tf`: 1 + ln(tf), żeby
  "bad bad bad bad" nie ważyło 4 razy więcej niż "bad"),
- `idf` - słowa obecne w wielu recenzjach ("game") dostają małą wagę, rzadkie większą,
- wektor jest normalizowany do długości 1.
- Bigramy łapią krótki kontekst: "not good", "not bad", "not recommend".
- `min_df=2` - słowo/bigram musi wystąpić w co najmniej 2 recenzjach (literówki odpadają).

## 5. Regresja logistyczna

Liczy sumę ważoną cech `z = w·x + b` i zamienia ją na prawdopodobieństwo
`p = 1 / (1 + e^-z)`. Jeśli p >= 0,5 -> pozytywna. Wagi uczy się, minimalizując
log-loss z regularyzacją L2; parametr C to odwrotność siły regularyzacji
(małe C - prostszy model). C wybrałem walidacją krzyżową 5-krotną na zbiorze
treningowym. Zaleta: wagi da się odczytać - `results/top_features.csv` pokazuje
np. "not recommend", "refund", "unplay" po stronie negatywnej i "10/10", "best", "peak"
po pozytywnej.

## 6. Metryki

- accuracy - odsetek poprawnych odpowiedzi,
- precision (dla klasy pozytywnej) - ile z recenzji oznaczonych jako pozytywne naprawdę
  jest pozytywnych,
- recall - ile pozytywnych recenzji model znalazł,
- F1 - średnia harmoniczna precision i recall (wysoka tylko, gdy obie są wysokie),
- macierz pomyłek - liczby TN, FP, FN, TP,
- ROC AUC - jak dobrze model porządkuje recenzje wg prawdopodobieństwa, niezależnie od progu 0,5.

## 7. Transformer (DistilBERT SST-2)

Gotowy model `distilbert-base-uncased-finetuned-sst-2-english`: mniejsza wersja BERT-a,
dostrojona na zdaniach z recenzji filmów (SST-2). Dostaje surowy tekst (ma własny
tokenizator WordPiece i rozumie kontekst całego zdania). Nie dotrenowywałem go.
Wynik: trochę słabszy od mojego modelu na recenzjach gier i ok. 400 razy wolniejszy na CPU.

## 8. GUI

Streamlit uruchamia skrypt `app.py` od góry przy każdej akcji użytkownika;
`st.cache_resource` trzyma wczytane modele, `st.cache_data` - pobrane recenzje,
`st.session_state` - ostatni wynik. WordCloud rysuję z częstości słów po czyszczeniu
(bez stemmingu, żeby słowa były czytelne, i bez słów z tytułu gry, które dominowały
obie chmury).

## 9. Pytania na prezentację (z odpowiedziami)

1. **Skąd są dane do predykcji i treningu?** Oba z recenzji Steam (EN), przez oficjalne
   Web API. Trening: 30 gier, zbalansowane 400+400 na grę. Predykcja: inne gry, pobierane
   na żywo w GUI. Dzięki temu dane są spójne strukturą i charakterem.
2. **Dlaczego nie HackerNews (pierwsza wersja)?** Tytuły i komentarze techniczne nie mają
   wyraźnego wydźwięku i nie ma do nich etykiet, a model z recenzji filmów do nich nie pasował.
3. **Dlaczego nie usuwasz "not"?** Bo zmienia znaczenie: "not good" po usunięciu "not"
   to "good". W notatniku usunięcie negacji obniża F1.
4. **Stemming czy lematyzacja i dlaczego?** Stemming - na walidacji minimalnie lepszy
   i kilka razy szybszy; lematyzacja daje poprawne słowa, ale tu nie poprawiła wyniku.
5. **Co to jest TF-IDF?** Waga słowa = częstość w dokumencie razy "rzadkość" w korpusie;
   wzór w punkcie 4.
6. **Po co bigramy?** Żeby złapać krótkie zwroty z negacją i frazy: "not bad",
   "not recommend", "waste money".
7. **Dlaczego regresja logistyczna?** Szybka, dobrze działa na rzadkich wektorach TF-IDF,
   daje prawdopodobieństwo (pewność w GUI) i interpretowalne wagi. LinearSVC wyszedł podobnie,
   Naive Bayes słabiej (niski recall).
8. **Jak dobrałeś C?** GridSearchCV, 5 foldów, metryka F1, tylko na zbiorze treningowym;
   test użyty raz na końcu.
9. **Dlaczego F1, a nie tylko accuracy?** Klasy nie są idealnie równe (44% pozytywnych),
   a F1 pokazuje balans między fałszywymi alarmami i pominięciami.
10. **Czemu transformer jest gorszy?** Był uczony na innej domenie (filmy), nie zna żargonu
    graczy ("refund", "unplayable", "peak"); nie dotrenowywałem go. Za to rozumie kontekst
    zdania, więc przy dotrenowaniu pewnie by wygrał - kosztem czasu.
11. **Jak sprawdzasz model na danych z Internetu?** Steam podaje, czy autor poleca grę -
    w GUI porównuję predykcję z tą etykietą (accuracy, F1, macierz pomyłek), a skrypt
    `evaluate_live.py` robi to dla 9 gier spoza treningu.
12. **Co pokazuje WordCloud?** Najczęstsze słowa (po czyszczeniu) w recenzjach, które model
    uznał za pozytywne / negatywne - widać, o czym gracze piszą (np. "crash",
    "performance" po stronie negatywnej).
13. **Jakie błędy robi model?** Sarkazm i żarty ("0/10 would buy again"), recenzje
    z wulgaryzmami u zadowolonych graczy, krytyka wydawcy przy poleceniu gry.
14. **Co z pustymi tekstami?** Po czyszczeniu są pomijane i liczone osobno - nie dostają
    etykiety "negatywny" (to był błąd pierwszej wersji).
15. **Co gdy API nie odpowiada?** Ponawiam 3 razy z backoffem, potem GUI pokazuje
    ostrzeżenie z przyczyną; aplikacja się nie wywraca.
