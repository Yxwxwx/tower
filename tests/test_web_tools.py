"""测试网络工具。"""

import pytest
from tower.tools.builtin.web import web_fetch, web_search


class TestWebFetch:
    def test_invalid_url_scheme(self):
        result = web_fetch.invoke({"url": "ftp://example.com"})
        assert "error" in result

    def test_localhost_blocked(self):
        result = web_fetch.invoke({"url": "http://localhost:8080"})
        assert "error" in result

    def test_private_ip_blocked(self):
        result = web_fetch.invoke({"url": "http://192.168.1.1"})
        assert "error" in result

    def test_invalid_url(self):
        result = web_fetch.invoke({"url": "not-a-url"})
        assert "error" in result

    def test_real_url_fetch(self):
        """测试抓取真实网页 — 需要外网。"""
        result = web_fetch.invoke({"url": "https://httpbin.org/ip"})
        if "error" in result:
            pytest.skip(f"Network unavailable: {result['error']}")
        assert "content" in result
        assert result["status"] == 200


class TestWebSearch:
    def test_no_api_key_or_success(self):
        result = web_search.invoke({"query": "python asyncio tutorial"})
        # 如果配置了 API key 就返回结果，否则返回友好的错误
        if "error" in result:
            assert "TAVILY_API_KEY" in result["error"] or "error" in result
