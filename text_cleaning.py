"""Czyszczenie recenzji przed TF-IDF (na podstawie Prezentacji 2 z zajęć).

Kolejność kroków:
 1. HTML i BBCode ([b], [spoiler], [url=...]) - recenzje Steam używają BBCode
 2. adresy URL i e-maile
 3. rozwinięcie skrótów z negacją: don't -> do not, can't -> can not
 4. małe litery (pułapka: tracimy "CAPS" jako wyraz emocji, ale dla
    bag-of-words "GREAT" i "great" to to samo słowo - opis w docs/NAUKA.md)
 5. tokenizacja: słowa, liczby, oceny typu 10/10, znaki ! i ?, emoji
    (nie usuwam wszystkich znaków specjalnych - patrz pułapki z zajęć)
 6. stopwords BEZ słów negacji - "not good" musi zostać "not good"
 7. stemming (Porter z NLTK) albo lematyzacja (spaCy) albo nic
"""
import html
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# słowa, których NIE wolno usuwać jako stopwords (pułapka 1 z prezentacji)
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody",
             "neither", "nowhere", "cannot", "without"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS

HTML_TAG_RE = re.compile(r"<[^>]+>")
BBCODE_RE = re.compile(r"\[/?[a-z*0-9]+(=[^\]]*)?\]", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\w+")
# emoji: najczęstsze bloki Unicode (buźki, symbole, dingbaty)
EMOJI = "\U0001F300-\U0001FAFF☀-➿"
TOKEN_RE = re.compile(rf"\d+/\d+|[a-z0-9]+(?:'[a-z]+)?|[!?]|[{EMOJI}]")

_stemmer = PorterStemmer()


def expand_negations(text: str) -> str:
    text = text.replace("’", "'")  # typograficzny apostrof z telefonów
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bain't\b", "is not", text, flags=re.IGNORECASE)
    text = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
    return text


def basic_clean(text: str, lowercase: bool = True) -> str:
    """Kroki 1-4: usuwa śmieci, ale nie zmienia jeszcze słów."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = BBCODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = expand_negations(text)
    if lowercase:
        text = text.lower()
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    # "!!!" daje trzy tokeny "!" - zostawiam jeden, żeby nie dominowały
    out = []
    for t in tokens:
        if t in "!?" and out and out[-1] == t:
            continue
        out.append(t)
    return out


@lru_cache(maxsize=1)
def get_spacy():
    import spacy
    # parser i NER nie są potrzebne do lematów - wyłączam, jest kilka razy szybciej
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize_many(token_lists: list[list[str]]) -> list[list[str]]:
    nlp = get_spacy()
    texts = [" ".join(t) for t in token_lists]
    result = []
    for doc in nlp.pipe(texts, batch_size=256):
        result.append([tok.lemma_.lower() for tok in doc if not tok.is_space])
    return result


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def clean_many(texts: list[str], normalize: str | None = "lemma",
               stopwords: bool = True) -> list[str]:
    """Cały pipeline dla listy tekstów. Zwraca teksty z tokenami oddzielonymi spacją.

    normalize: "lemma", "stem" albo None.
    """
    token_lists = [tokenize(basic_clean(t)) for t in texts]
    # stopwords usuwam PRZED stemmingiem - po nim "this" to "thi" i nie pasuje do listy
    if stopwords:
        token_lists = [remove_stopwords(toks) for toks in token_lists]
    if normalize == "lemma":
        token_lists = lemmatize_many(token_lists)
        if stopwords:   # lematyzacja tworzy nowe stopwords, np. "was" -> "be"
            token_lists = [remove_stopwords(toks) for toks in token_lists]
    elif normalize == "stem":
        token_lists = [[_stemmer.stem(t) for t in toks] for toks in token_lists]
    elif normalize is not None:
        raise ValueError(f"Nieznana normalizacja: {normalize}")
    return [" ".join(toks) for toks in token_lists]


def clean_text(text: str, normalize: str | None = "lemma", stopwords: bool = True) -> str:
    return clean_many([text], normalize, stopwords)[0]


def explain_steps(text: str, normalize: str = "lemma") -> list[tuple[str, str]]:
    """Wynik po każdym kroku - do pokazania w GUI i w notatniku."""
    steps = [("Tekst oryginalny", text)]
    cleaned = basic_clean(text)
    steps.append(("HTML/BBCode/URL usunięte, negacje rozwinięte, małe litery", cleaned))
    tokens = tokenize(cleaned)
    steps.append(("Tokeny", ", ".join(tokens)))
    tokens = remove_stopwords(tokens)
    steps.append(("Bez stopwords (negacje zostają)", ", ".join(tokens)))
    if normalize == "lemma":
        tokens = remove_stopwords(lemmatize_many([tokens])[0])
        steps.append(("Lematyzacja (spaCy)", ", ".join(tokens)))
    elif normalize == "stem":
        tokens = [_stemmer.stem(t) for t in tokens]
        steps.append(("Stemming (Porter)", ", ".join(tokens)))
    return steps
