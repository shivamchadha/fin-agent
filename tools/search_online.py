from newspaper import Article,Config
from duckduckgo_search import DDGS
from datetime import datetime
import dateutil.parser
from functools import lru_cache
import time

config = Config()
config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
config.request_timeout = 15


ddgs = DDGS( timeout=60)

@lru_cache(maxsize=128)
def search_ddg(query, max_results=10):
    time.sleep(30)
    results = ddgs.text(query, max_results=max_results)
    return results

@lru_cache(maxsize=128)
def search_ddg_news(query, max_results=10):
    time.sleep(30)
    try:
        results = ddgs.news(keywords=query, region="in-en", safesearch="off", timelimit="w", max_results=max_results)
    except Exception as e:
        print(f"News search error: {str(e)}")
        return search_ddg(query + " news", max_results)
    return results


def extract_article_text(ddg_result):
    """
    Enhanced extraction that:
    1. Uses DDG metadata as fallback
    2. Preserves all original fields
    Returns unified article format with source tracking
    """
    base_result = {
        "title": ddg_result.get("title", ""),
        "text": ddg_result.get("body", ""),
        "publish_date": ddg_result.get("date"),
        "authors": [],
        "url": ddg_result.get("url", ""),
        "image": ddg_result.get("image", ""),
        "publisher": ddg_result.get("source", ""),
        "method": "ddg_metadata"  # Default to metadata
    }

    # Only attempt full extraction if we have a valid URL
    if not base_result["url"]:
        return base_result

    try:
        article = Article(base_result["url"], config=config)
        article.download()
        article.parse()
        
        # Only override if we got meaningful content
        if article.text and len(article.text.strip()) > 200:
            return {
                **base_result,
                "title": article.title or base_result["title"],
                "text": article.text,
                "publish_date": article.publish_date or base_result["date"],
                "authors": article.authors,
                "method": "full_extraction"
            }
            
    except Exception as e:
        print(f"⚠️ Extraction failed, using metadata: {str(e)}")
    
    return base_result

