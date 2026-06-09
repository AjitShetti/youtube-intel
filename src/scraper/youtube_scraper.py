import os
import json
from googleapiclient.discovery import build
from src.config import get_config
from src.utils.logger import setup_logger
logger = setup_logger(__name__)



API_KEY = get_config("youtube.api_key")
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_video_id(url: str) -> str:
    """Extracts video ID from YT URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    
    video_id = url.split("/")[-1]
    return video_id.split("?")[0]  # Remove anything after ?

def fetch_comments(video_url: str, save_path="data/raw"):
    video_id = get_video_id(video_url)
    comments = []
    next_page_token = None

    max_results = get_config("youtube.max_results", 1000)
    
    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=next_page_token,
            order="relevance"
        )
        try:
            response = request.execute()
        except Exception as e:
            logger.info(f"Error fetching comments: {e}")
            if len(comments) > 0:
                logger.info(f"Returning {len(comments)} comments already fetched.")
                break
            return None

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "author": snippet.get("authorDisplayName"),
                "text_raw": snippet.get("textDisplay"),
                "published_at": snippet.get("publishedAt"),
                "like_count": snippet.get("likeCount")
            })
            if len(comments) >= max_results:
                break
        
        next_page_token = response.get("nextPageToken")
        if next_page_token is None or len(comments) >= max_results:
            break
    
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{video_id}.json"
    with open(file_path, "w") as f:
        json.dump(comments, f, indent=2)
    return file_path

if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    logger.info(fetch_comments(url))