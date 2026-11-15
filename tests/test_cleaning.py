import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_cleaning import basic_clean, clean_text, tokenize  # noqa: E402


def test_negation_not_removed():
    # pułapka 1 z prezentacji - "not" nie może zniknąć razem ze stopwords
    assert clean_text("This is not good", normalize=None).split() == ["not", "good"]


def test_contraction_expanded():
    assert "not" in clean_text("I don't like it", normalize=None).split()
    assert "not" in clean_text("It won’t start", normalize=None).split()


def test_html_bbcode_and_url_removed():
    text = "[b]Great[/b] game <br> see https://example.com/x or mail me@test.com"
    cleaned = basic_clean(text)
    assert "[b]" not in cleaned and "<br>" not in cleaned
    assert "http" not in cleaned and "@" not in cleaned
    assert "great" in cleaned


def test_exclamation_and_emoji_kept():
    tokens = tokenize(basic_clean("Love it!!! \U0001F60D"))
    assert tokens == ["love", "it", "!", "\U0001F60D"]


def test_score_kept_as_one_token():
    # liczby niosą informację (pułapka 4) - ocena 2/10 to sygnał negatywny
    assert "2/10" in tokenize("solid 2/10 would not buy")


def test_stemming_and_lemmatization_differ():
    stem = clean_text("The studies were running", normalize="stem").split()
    lemma = clean_text("The studies were running", normalize="lemma").split()
    assert "studi" in stem          # stem nie musi być słowem
    assert "study" in lemma         # lemat jest formą słownikową
    assert "run" in lemma


def test_empty_and_none():
    assert clean_text("", normalize=None) == ""
    assert basic_clean(None) == ""
