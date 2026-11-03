"""Streamlit GUI — Inteligentny System Monitorowania Opinii (HackerNews)."""
import streamlit as st
from datetime import datetime
import pandas as pd

from scraper import fetch_top_stories, fetch_comments
from sentiment import analyze

st.set_page_config(
    page_title="EDT Sentiment Monitor",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Inteligentny System Monitorowania Opinii")
st.caption("Źródło danych: HackerNews Top Stories | Model: distilbert-base-uncased-finetuned-sst-2-english")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Ustawienia")
    n_stories = st.slider("Liczba stories do pobrania", 5, 50, 30, 5)
    n_comments = st.slider("Komentarzy na story", 3, 20, 10, 1)
    filter_opt = st.selectbox("Filtr sentymentu", ["Wszystkie", "POSITIVE", "NEGATIVE"])
    st.divider()
    st.markdown("**EDT Projekt 1** — Eksploracja Danych Tekstowych")
    st.markdown("Politechnika / Sem. 2 2025/2026")

# --- Session state ---
if "stories_df" not in st.session_state:
    st.session_state.stories_df = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "raw_stories" not in st.session_state:
    st.session_state.raw_stories = []

# --- Fetch button ---
col_btn, col_info = st.columns([1, 3])
with col_btn:
    fetch_clicked = st.button("🔄 Pobierz najnowsze", type="primary", use_container_width=True)

with col_info:
    if st.session_state.last_refresh:
        st.info(f"Ostatnie odświeżenie: {st.session_state.last_refresh}")

if fetch_clicked:
    with st.spinner("Pobieranie stories z HackerNews..."):
        stories = fetch_top_stories(n=n_stories)
        st.session_state.raw_stories = stories

    if not stories:
        st.error("Nie udało się pobrać stories. Sprawdź połączenie z internetem.")
    else:
        with st.spinner(f"Analizuję sentyment {len(stories)} tytułów... (pierwsze uruchomienie pobiera model ~260 MB)"):
            titles = [s["title"] for s in stories]
            sentiments = analyze(titles)

        rows = []
        for s, sent in zip(stories, sentiments):
            rows.append({
                "id": s["id"],
                "Tytuł": s["title"],
                "URL": s["url"],
                "Wynik HN": s["score"],
                "Sentyment": sent["label"],
                "Pewność": round(sent["score"] * 100, 1),
                "kids": s["kids"],
            })

        st.session_state.stories_df = pd.DataFrame(rows)
        st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"Pobrano i przeanalizowano {len(rows)} stories.")

# --- Display results ---
if st.session_state.stories_df is not None:
    df = st.session_state.stories_df.copy()

    # Filtrowanie
    if filter_opt != "Wszystkie":
        df = df[df["Sentyment"] == filter_opt]

    if df.empty:
        st.warning(f"Brak stories z sentymentem: {filter_opt}")
    else:
        # Metryki ogólne
        total = len(st.session_state.stories_df)
        pos = (st.session_state.stories_df["Sentyment"] == "POSITIVE").sum()
        neg = total - pos

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Łącznie stories", total)
        m2.metric("POSITIVE", pos, delta=f"{pos/total*100:.0f}%")
        m3.metric("NEGATIVE", neg, delta=f"-{neg/total*100:.0f}%", delta_color="inverse")
        m4.metric("Średnia pewność", f"{st.session_state.stories_df['Pewność'].mean():.1f}%")

        st.divider()

        # Wykres + tabela
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            st.subheader("📊 Rozkład sentymentów")
            chart_data = st.session_state.stories_df["Sentyment"].value_counts().reset_index()
            chart_data.columns = ["Sentyment", "Liczba"]
            st.bar_chart(chart_data.set_index("Sentyment"), color=["#00cc44"])

        with col_table:
            st.subheader(f"📋 Stories ({filter_opt})")
            display_df = df[["Tytuł", "Sentyment", "Pewność", "Wynik HN"]].copy()
            display_df["Sentyment"] = display_df["Sentyment"].apply(
                lambda x: f"{'✅' if x == 'POSITIVE' else '❌'} {x}"
            )
            display_df["Pewność"] = display_df["Pewność"].apply(lambda x: f"{x}%")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )

        st.divider()

        # Komentarze — expander per story
        st.subheader("💬 Komentarze — kliknij story, by rozwinąć")
        for _, row in df.iterrows():
            label = "POSITIVE" if "POSITIVE" in str(row["Sentyment"]) else "NEGATIVE"
            sent_icon = "✅" if label == "POSITIVE" else "❌"
            with st.expander(f"{sent_icon} [{label} {row['Pewność']}%] {row['Tytuł'][:80]}"):
                if row["URL"]:
                    st.markdown(f"[🔗 Otwórz artykuł]({row['URL']})")

                kids = row["kids"] if isinstance(row["kids"], list) else []
                if not kids:
                    st.info("Brak komentarzy do tego story.")
                else:
                    with st.spinner("Pobieranie komentarzy..."):
                        comments = fetch_comments(int(row["id"]), limit=n_comments)

                    if not comments:
                        st.info("Brak dostępnych komentarzy.")
                    else:
                        with st.spinner("Analizuję komentarze..."):
                            comment_sents = analyze(comments)

                        for comment, csent in zip(comments, comment_sents):
                            icon = "✅" if csent["label"] == "POSITIVE" else "❌"
                            conf = round(csent["score"] * 100, 1)
                            st.markdown(
                                f"""{icon} **{csent['label']}** ({conf}%)  \n> {comment[:300]}{'...' if len(comment) > 300 else ''}""",
                                unsafe_allow_html=False,
                            )
                            st.divider()

    # Timestamp na dole
    st.caption(f"⏱️ Ostatnie odświeżenie danych: {st.session_state.last_refresh}")

else:
    st.info("Kliknij **Pobierz najnowsze**, aby pobrać i przeanalizować stories z HackerNews.")
    st.markdown(
        """
        ### Jak to działa?
        1. Aplikacja pobiera top stories z **HackerNews API** (bez klucza API)
        2. Tytuły i komentarze są analizowane modelem **DistilBERT** (fine-tuned SST-2)
        3. Każdy tekst otrzymuje etykietę **POSITIVE** lub **NEGATIVE** z poziomem pewności
        4. Możesz filtrować wyniki i przeglądać komentarze do każdego story
        """
    )
