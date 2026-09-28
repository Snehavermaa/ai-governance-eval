import pytest

from governance.text import content_terms, numbers, split_sentences


@pytest.mark.parametrize(
    "text,expected",
    [
        ("One. Two! Three?", ["One.", "Two!", "Three?"]),
        ("Use e.g. a chair. Then stop.", ["Use e.g. a chair.", "Then stop."]),
        ("The limit is 2.5 days. Next.", ["The limit is 2.5 days.", "Next."]),
        ("   ", []),
        ("no terminal punctuation", ["no terminal punctuation"]),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("24 days and 10 days", {"24", "10"}),
        ("15,000 rupees", {"15000"}),
        ("2.50 hours", {"2.5"}),
        ("two weeks", {"2"}),
        ("no digits here", set()),
    ],
)
def test_numbers(text, expected):
    assert numbers(text) == expected


def test_content_terms_drop_stopwords_and_stem():
    assert content_terms("The days are approved") == {"day", "approv"}
