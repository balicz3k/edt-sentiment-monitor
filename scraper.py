"""Pobiera tytuły i komentarze z HackerNews API."""
import httpx
import re

HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"


def fetch_top_stories(n: int = 30) -> list[dict]:
    """Pobiera n top stories z HN. Zwraca listę {title, url, score, text, id}."""
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(HN_TOP)
        resp.raise_for_status()
        ids = resp.json()[:n]

        stories = []
        for story_id in ids:
            try:
                item_resp = client.get(HN_ITEM.format(id=story_id))
                item_resp.raise_for_status()
                item = item_resp.json()
                if not item or item.get("type") != "story":
                    continue
                stories.append({
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0),
                    "text": item.get("text", ""),
                    "kids": item.get("kids", []),
                })
            except Exception:
                continue
        return stories


def fetch_comments(story_id: int, limit: int = 10) -> list[str]:
    """Pobiera pierwsze `limit` komentarzy do story (bez HTML)."""
    with httpx.Client(timeout=15.0) as client:
        try:
            item_resp = client.get(HN_ITEM.format(id=story_id))
            item_resp.raise_for_status()
            story = item_resp.json()
        except Exception:
            return []

        kids = story.get("kids", [])[:limit]
        comments = []
        for kid_id in kids:
            try:
                comment_resp = client.get(HN_ITEM.format(id=kid_id))
                comment_resp.raise_for_status()
                comment = comment_resp.json()
                text = comment.get("text", "")
                if text and not comment.get("deleted") and not comment.get("dead"):
                    clean = re.sub(r"<[^>]+>", " ", text)
                    clean = clean.replace("&#x27;", "'").replace("&quot;", '"')
                    clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                    clean = " ".join(clean.split())
                    if clean.strip():
                        comments.append(clean.strip())
            except Exception:
                continue
        return comments
