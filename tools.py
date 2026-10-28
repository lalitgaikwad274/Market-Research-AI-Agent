import sys
import warnings

# Suppress the ddgs rename runtime warning from duckduckgo_search
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*has been renamed to `ddgs`.*")

try:
    import ddgs  # noqa: F401
except ImportError:
    try:
        import duckduckgo_search
        sys.modules["ddgs"] = duckduckgo_search
    except ImportError:
        pass

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

try:
    from langchain_core.tools import Tool
except ImportError:
    from langchain.tools import Tool

from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

# DuckDuckGo wrapper. time=None doesn't restrict results to the past year.
search_wrapper = DuckDuckGoSearchAPIWrapper(time=None, max_results=5)

# Domains that are useless to scrape for lead qualification
SKIP_DOMAINS = ("duckduckgo.com", "facebook.com", "instagram.com", "linkedin.com",
                "youtube.com", "twitter.com", "x.com", "yelp.", "wikipedia.org")


def save_to_txt(data: str, filename: str = "leads_output.txt") -> str:
    """Append data to a text file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Leads Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"


def scrape_website(url: str) -> str:
    """Fetch a page and return up to 5000 chars of cleaned text."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:5000]
    except Exception as e:
        return f"Error scraping website: {e}"


def _search(query: str) -> list[dict]:
    """Return DuckDuckGo results as [{title, snippet, link}, ...]."""
    try:
        return search_wrapper.results(query, max_results=5)
    except Exception as e:
        return [{"title": "error", "snippet": f"Search failed: {e}", "link": ""}]


def _format_results(results: list[dict]) -> str:
    return "\n".join(
        f"- {r.get('title', '')}: {r.get('snippet', '')} ({r.get('link', '')})"
        for r in results
    )


def _first_scrapable(results: list[dict]) -> str:
    """Scrape the first result that isn't a skipped domain and loads fine."""
    for r in results:
        link = r.get("link", "")
        if link and not any(d in link for d in SKIP_DOMAINS):
            scraped = scrape_website(link)
            if not scraped.startswith("Error"):
                return f"Source: {link}\n{scraped}"
    return ""


def search_web(query: str) -> str:
    """Search tool: returns titles, snippets AND links (the default Run tool drops links)."""
    results = _search(query)
    return _format_results(results) if results else "No results found."


def search_and_scrape(target: str) -> str:
    """Scrape a URL directly, or search for a business/topic and scrape the best result."""
    target = target.strip()

    # Direct URL
    if target.startswith(("http://", "https://")):
        return scrape_website(target)

    # General discovery query (e.g. "small businesses in Vancouver")
    if any(t in target.lower() for t in ["small business", "businesses", "vancouver", "directory", "list"]):
        results = _search(target)
        scraped = _first_scrapable(results)
        out = _format_results(results)
        return f"{out}\n\nScraped content:\n{scraped}" if scraped else out

    # Specific company: look for IT-related info
    parts = []
    for keyword in ["IT services", "managed IT"]:
        results = _search(f"{target} {keyword}")
        scraped = _first_scrapable(results)
        parts.append(scraped or _format_results(results))
    combined = "\n\n".join(p for p in parts if p)
    return combined or "No information found."


from langchain_core.tools import tool

# Defining all the tools using @tool decorator with clean argument schemas
@tool("search")
def search_tool(query: str) -> str:
    """Search the web with DuckDuckGo. Input is a search query string. Returns titles, snippets and URLs."""
    return search_web(query)


@tool("scrape")
def scrape_tool(target: str) -> str:
    """Scrapes a URL directly, or searches for a business/topic and scrapes the result. Input is either a URL or a search query / business name."""
    return search_and_scrape(target)


@tool("save_to_text")
def save_tool(data: str) -> str:
    """Saves the given text data to a text file."""
    return save_to_txt(data)