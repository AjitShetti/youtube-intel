import pytest
from src.scraper.youtube_scraper import get_video_id

def test_get_video_id_standard_url():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert get_video_id(url) == "dQw4w9WgXcQ"

def test_get_video_id_short_url():
    url = "https://youtu.be/dQw4w9WgXcQ"
    assert get_video_id(url) == "dQw4w9WgXcQ"

def test_get_video_id_with_params():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s"
    assert get_video_id(url) == "dQw4w9WgXcQ"

def test_get_video_id_short_with_params():
    url = "https://youtu.be/dQw4w9WgXcQ?t=10"
    assert get_video_id(url) == "dQw4w9WgXcQ"
