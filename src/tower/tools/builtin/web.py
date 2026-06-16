"""网络工具 —— 抓取网页内容和网络搜索。"""

import json
import os
import urllib.request
import urllib.error
import urllib.parse
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


# ============================================================
# Web Fetch
# ============================================================


@tool
def web_fetch(url: str) -> dict:
    """Fetch the content of a web page and return it as plain text.

    Use for: reading documentation, fetching API references, checking
    package versions on PyPI/npm, reading GitHub READMEs or issues.

    Args:
        url: The URL to fetch (must start with http:// or https://).
    """
    if not url.startswith(("http://", "https://")):
        return {"error": f"URL must start with http:// or https://, got: {url}"}

    # 限制可访问的域名类型（避免 SSRF 到内网）
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""

    # 拒绝内网地址
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return {"error": f"Cannot fetch localhost URL: {url}"}
    if hostname.startswith("10.") or hostname.startswith("192.168.") or hostname.startswith("172.16."):
        return {"error": f"Cannot fetch private network URL: {url}"}

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Tower-Agent/0.1.0 (fetching documentation)",
                "Accept": "text/html,text/plain,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            # 读取最多 500KB
            body = resp.read(500 * 1024)
            content_type = resp.headers.get("Content-Type", "")

            # 尝试解码
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            try:
                text = body.decode(charset)
            except (UnicodeDecodeError, LookupError):
                text = body.decode("utf-8", errors="replace")

            # 简单 HTML→text 转换（去掉标签，保留文本）
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&amp;", "&", text)
            text = re.sub(r"&lt;", "<", text)
            text = re.sub(r"&gt;", ">", text)
            text = re.sub(r"&quot;", '"', text)
            text = re.sub(r"&#\d+;", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            max_len = 30000
            truncated = len(text) > max_len
            text = text[:max_len]

            return {
                "url": url,
                "status": resp.status,
                "content": text,
                "content_length": len(text),
                "truncated": truncated,
            }
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}", "url": url}
    except urllib.error.URLError as e:
        return {"error": f"Failed to fetch URL: {e.reason}", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


# ============================================================
# Web Search
# ============================================================


@tool
def web_search(query: str) -> dict:
    """Search the web for information. Returns titles, URLs, and snippets
    from search results.

    Use for: finding documentation, checking latest versions, searching
    for error messages, finding code examples, looking up APIs.

    Args:
        query: The search query string.
    """
    if not TAVILY_API_KEY:
        return {
            "error": (
                "TAVILY_API_KEY not configured. Set it in .env file. "
                "Get a free key at https://tavily.com"
            )
        }

    try:
        data = json.dumps({
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 5,
            "search_depth": "basic",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())

        results = result.get("results", [])
        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content", "") or r.get("snippet", ""))[:300],
            }
            for r in results
        ]

        return {
            "query": query,
            "results": formatted,
            "count": len(formatted),
        }
    except urllib.error.HTTPError as e:
        return {"error": f"Search API error: HTTP {e.code}", "query": query}
    except Exception as e:
        return {"error": str(e), "query": query}
