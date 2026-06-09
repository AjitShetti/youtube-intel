import pytest
from src.preprocessing.clean_text import clean_comment

def test_clean_comment_empty():
    assert clean_comment(None) == ""
    assert clean_comment("") == ""

def test_clean_comment_urls():
    text = "Check out this link https://example.com and this one www.test.com"
    # Should remove URLs and extra spaces, lemma for "check", "link"
    cleaned = clean_comment(text)
    assert "https" not in cleaned
    assert "www" not in cleaned
    assert "example" not in cleaned

def test_clean_comment_mentions():
    text = "Hello @username, how are you?"
    cleaned = clean_comment(text)
    assert "username" not in cleaned
    assert "hello" in cleaned

def test_clean_comment_emojis_and_punctuation():
    text = "Wow! This is amazing!!! 😊🔥 #awesome"
    cleaned = clean_comment(text)
    assert "😊" not in cleaned
    assert "🔥" not in cleaned
    assert "!" not in cleaned
    assert "awesome" in cleaned

def test_clean_comment_lemmatization_and_stopwords():
    text = "The cats are running quickly to the stores."
    cleaned = clean_comment(text)
    # the, are, to, the -> stopwords
    # cats -> cat, running -> run, stores -> store
    # Since we use spacy lemma and stopword removal
    assert "cat" in cleaned
    assert "run" in cleaned
    assert "store" in cleaned
    assert "the" not in cleaned
