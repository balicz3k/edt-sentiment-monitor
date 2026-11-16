"""Monitor opinii o grach ze Steama - GUI w Streamlit.

Uruchomienie: streamlit run app.py
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score
from wordcloud import WordCloud

from games import PREDICT_GAMES, TRAIN_GAMES
from scraper import SteamError, fetch_reviews, search_games
from sentiment import LABEL_NAMES, OwnModel, TransformerModel
from text_cleaning import clean_many, explain_steps

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# słowa, które dominują w obu chmurach i nic nie mówią o sentymencie
WORDCLOUD_SKIP = {"game", "play", "!", "?", "s"}

st.set_page_config(page_title="Monitor opinii Steam", layout="wide")


@st.cache_resource
def load_own_model():
    return OwnModel()


@st.cache_resource
def load_transformer():
    return TransformerModel()


# cache w sesji: te same parametry przez 10 minut nie wysyłają nowych zapytań
@st.cache_data(ttl=600, show_spinner=False)
def cached_reviews(appid: int, n: int, review_type: str) -> pd.DataFrame:
    return pd.DataFrame(fetch_reviews(appid, n=n, review_type=review_type))


@st.cache_data(ttl=600, show_spinner=False)
def cached_search(term: str) -> list[dict]:
    return search_games(term)


def analyze(df: pd.DataFrame, model_choice: str) -> pd.DataFrame:
    own = load_own_model()
    df = df.copy()
    df["clean"] = own.clean(df["text"].tolist())
    # do chmur słów biorę tekst bez stemmingu - "amaz" czyta się gorzej niż "amazing"
    df["words"] = clean_many(df["text"].tolist(), normalize=None)
    # puste po czyszczeniu (np. same linki albo ASCII-art) - pomijam zamiast zgadywać
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        return df
    if model_choice == OwnModel.name:
        labels, conf = own.predict_clean(df["clean"].tolist())
    else:
        labels, conf = load_transformer().predict(df["text"].tolist())
    df["pred"] = labels
    df["pewnosc"] = conf.round(3)
    df["steam"] = df["voted_up"].astype(int)
    return df


def wordcloud_image(texts: pd.Series, game: str):
    # słowa z tytułu gry ("monster", "hunter") są w obu chmurach - pomijam je
    skip = WORDCLOUD_SKIP | set(game.lower().split())
    counts = Counter(t for text in texts for t in text.split() if t not in skip)
    if not counts:
        return None
    wc = WordCloud(width=600, height=320, background_color="white", max_words=80,
                   colormap="viridis", random_state=0)
    return wc.generate_from_frequencies(counts).to_array()


def show_results(res: dict):
    df = res["df"]
    st.subheader(f"{res['game']} - {len(df)} recenzji, model: {res['model']}")
    st.caption(f"Pobrano: {res['fetched_at']}. Pominięto pustych po czyszczeniu: {res['dropped']}.")

    pos_share = df["pred"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pozytywne wg modelu", f"{pos_share:.0%}")
    c2.metric("Polecający wg Steam", f"{df['steam'].mean():.0%}")
    c3.metric("Zgodność ze Steam (accuracy)", f"{accuracy_score(df['steam'], df['pred']):.1%}")
    c4.metric("F1 względem Steam", f"{f1_score(df['steam'], df['pred'], zero_division=0):.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Rozkład sentymentu**")
        dist = pd.DataFrame({
            "model": df["pred"].map(LABEL_NAMES).value_counts(),
            "Steam": df["steam"].map(LABEL_NAMES).value_counts(),
        }).fillna(0)
        st.bar_chart(dist, stack=False)
    with right:
        st.markdown("**Macierz pomyłek: model vs ocena autora na Steam**")
        cm = pd.crosstab(df["steam"].map(LABEL_NAMES), df["pred"].map(LABEL_NAMES),
                         rownames=["Steam"], colnames=["model"])
        st.dataframe(cm)
        days = df["created"].dt.date.nunique()
        if days > 1:
            st.markdown("**Udział pozytywnych recenzji w kolejnych dniach (wg modelu)**")
            day = df["created"].dt.strftime("%Y-%m-%d")
            trend = df.groupby(day)["pred"].mean().rename("udział pozytywnych")
            st.line_chart(trend)

    st.markdown("**Chmury słów (tekst po czyszczeniu, bez stopwords i słów z tytułu gry)**")
    w1, w2 = st.columns(2)
    for col, label, title in ((w1, 1, "Recenzje pozytywne"), (w2, 0, "Recenzje negatywne")):
        with col:
            st.markdown(title)
            img = wordcloud_image(df.loc[df["pred"] == label, "words"], res["game"])
            if img is None:
                st.info("Brak recenzji w tej grupie.")
            else:
                st.image(img, use_container_width=True)

    st.markdown("**Przykłady**")
    which = st.radio("Pokaż", ["wszystkie", "pozytywne", "negatywne",
                               "niezgodne ze Steam"], horizontal=True)
    view = df
    if which == "pozytywne":
        view = df[df["pred"] == 1]
    elif which == "negatywne":
        view = df[df["pred"] == 0]
    elif which == "niezgodne ze Steam":
        view = df[df["pred"] != df["steam"]]
    table = pd.DataFrame({
        "recenzja": view["text"].str.slice(0, 300),
        "sentyment": view["pred"].map(LABEL_NAMES),
        "pewność": view["pewnosc"],
        "Steam": view["steam"].map({1: "poleca", 0: "nie poleca"}),
        "data": view["created"].dt.strftime("%Y-%m-%d"),
        "godziny gry": view["playtime_h"],
    })
    st.dataframe(table, use_container_width=True, hide_index=True, height=380)
    st.download_button("Pobierz wyniki jako CSV",
                       df.drop(columns=["clean", "words"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"sentyment_{res['appid']}.csv", mime="text/csv")


def tab_reviews():
    with st.sidebar:
        st.header("Źródło danych")
        how = st.radio("Wybór gry", ["z listy", "wyszukaj po nazwie", "podaj appid"])
        appid, game = None, None
        if how == "z listy":
            names = list(PREDICT_GAMES.values())
            # domyślnie gra o mieszanych opiniach - ciekawszy rozkład niż przy samych pochwałach
            game = st.selectbox("Gra (spoza zbioru treningowego)", names,
                                index=names.index("Monster Hunter Wilds"))
            appid = {v: k for k, v in PREDICT_GAMES.items()}[game]
        elif how == "wyszukaj po nazwie":
            term = st.text_input("Nazwa gry", "witcher")
            if term.strip():
                try:
                    found = cached_search(term.strip())
                except SteamError as e:
                    st.warning(f"Wyszukiwanie nie działa: {e}")
                    found = []
                if found:
                    pick = st.selectbox("Wynik", [f"{g['name']} ({g['appid']})" for g in found])
                    appid = int(pick.rsplit("(", 1)[1].rstrip(")"))
                    game = pick.rsplit(" (", 1)[0]
                else:
                    st.info("Brak wyników.")
        else:
            appid = st.number_input("appid ze Steama", min_value=1, value=620, step=1)
            game = PREDICT_GAMES.get(appid, f"appid {appid}")
        if appid in TRAIN_GAMES:
            st.warning("Ta gra była w zbiorze treningowym - wynik będzie zawyżony.")

        n = st.slider("Liczba recenzji", 20, 300, 100, step=20)
        kind = st.selectbox("Rodzaj recenzji", ["najnowsze (wszystkie)", "tylko polecające",
                                                "tylko niepolecające"])
        review_type = {"najnowsze (wszystkie)": "all", "tylko polecające": "positive",
                       "tylko niepolecające": "negative"}[kind]
        model_choice = st.radio("Model", [OwnModel.name, TransformerModel.name])
        run = st.button("Pobierz i analizuj", type="primary", disabled=appid is None)

    if run:
        try:
            with st.spinner("Pobieram recenzje ze Steama..."):
                raw = cached_reviews(int(appid), n, review_type)
        except SteamError as e:
            st.warning(f"Nie udało się pobrać recenzji: {e}")
            return
        if raw.empty:
            st.warning("Ta gra nie ma recenzji w języku angielskim (albo appid jest błędne).")
            return
        with st.spinner("Czyszczenie tekstu i predykcja..."):
            df = analyze(raw, model_choice)
        if df.empty:
            st.warning("Po czyszczeniu nie została żadna recenzja.")
            return
        st.session_state["results"] = {
            "df": df, "game": game, "appid": appid, "model": model_choice,
            "dropped": len(raw) - len(df),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    if "results" in st.session_state:
        show_results(st.session_state["results"])
    else:
        st.info("Wybierz grę w panelu po lewej i kliknij \"Pobierz i analizuj\". "
                "Recenzje są pobierane na żywo z oficjalnego API Steama.")


def tab_single_text():
    st.markdown("Wpisz własny tekst, żeby zobaczyć kolejne kroki czyszczenia i predykcję.")
    text = st.text_area("Tekst", "The developers keep breaking the game!!! It's NOT fun anymore, "
                                 "don't buy it. 2/10 [b]Refund[/b] https://example.com")
    if not text.strip():
        return
    steps = explain_steps(text, normalize=load_own_model().normalization)
    st.table(pd.DataFrame(steps, columns=["krok", "wynik"]))
    own = load_own_model()
    labels, conf = own.predict([text])
    st.write(f"{OwnModel.name}: **{LABEL_NAMES[int(labels[0])]}** (pewność {conf[0]:.2f})")
    if st.button("Sprawdź też transformerem"):
        labels, conf = load_transformer().predict([text])
        st.write(f"{TransformerModel.name}: **{LABEL_NAMES[int(labels[0])]}** "
                 f"(pewność {conf[0]:.2f})")


def tab_about():
    metrics_file = RESULTS_DIR / "metrics.json"
    if not metrics_file.exists():
        st.info("Brak wyników - uruchom python train_model.py")
        return
    m = json.loads(metrics_file.read_text())
    st.markdown(f"Zbiór treningowy: {m['n_train']} recenzji z {len(TRAIN_GAMES)} gier, "
                f"testowy: {m['n_test']}. Normalizacja: {m['normalization']}, "
                f"najlepsze C = {m['best_C']}.")
    rows = {name: m[name] for name in ("tfidf_logreg", "tfidf_linear_svc", "tfidf_naive_bayes")}
    comp_file = RESULTS_DIR / "transformer_comparison.json"
    if comp_file.exists():
        rows["distilbert_sst2"] = json.loads(comp_file.read_text())["distilbert_sst2"]
    table = pd.DataFrame(rows).T[["accuracy", "precision", "recall", "f1"]]
    st.markdown("**Wyniki na zbiorze testowym**")
    st.dataframe(table)
    c1, c2 = st.columns(2)
    for col, name in ((c1, "confusion_matrix.png"), (c2, "confusion_matrix_transformer.png")):
        if (RESULTS_DIR / name).exists():
            col.image(str(RESULTS_DIR / name))
    live = RESULTS_DIR / "live_eval.csv"
    if live.exists():
        st.markdown("**F1 na recenzjach pobranych z Internetu (gry spoza treningu)**")
        st.dataframe(pd.read_csv(live).pivot(index="gra", columns="model", values="f1"))
    top = RESULTS_DIR / "top_features.csv"
    if top.exists():
        st.markdown("**Słowa i bigramy o największych wagach w regresji logistycznej**")
        st.dataframe(pd.read_csv(top).head(20), hide_index=True)


st.title("Monitor opinii o grach ze Steama")
t1, t2, t3 = st.tabs(["Analiza recenzji", "Pojedynczy tekst", "O modelu"])
with t1:
    tab_reviews()
with t2:
    tab_single_text()
with t3:
    tab_about()
