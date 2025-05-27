from newspaper import Article  
from duckduckgo_search import DDGS

def search_ddg(query, max_results=10):
    results = DDGS().text(query, max_results=max_results)
    return results

def search_ddg_news(query, max_results=10):
    results = DDGS().news(keywords=query, region="in-en", safesearch="off", timelimit="30s", max_results=max_results)
    return results


def extract_article_text(url):
    try:
        article = Article(url)
        article.download()  
        article.parse()     
        
        return {
            "title": article.title,
            "text": article.text,
            "publish_date": article.publish_date,
            "authors": article.authors,  
            "source": url
        }
    except Exception as e:
        print(f"⚠️ Error extracting {url}: {str(e)}")
        return None  