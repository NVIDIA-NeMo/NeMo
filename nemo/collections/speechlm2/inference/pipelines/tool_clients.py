# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP client helpers for S2S built-in tool functions.

Each class encapsulates one external API: the base URL, request headers,
and raw HTTP fetch logic.  Tool functions in streaming_s2s_pipeline.py call
these helpers and handle only the response shaping / error formatting.
"""

import json
import urllib.request
import urllib.parse


class WeatherAPIClient:
    """Open-Meteo current-conditions endpoint."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    WMO_CODES = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "depositing rime fog",
        51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
        61: "slight rain", 63: "moderate rain", 65: "heavy rain",
        71: "slight snow", 73: "moderate snow", 75: "heavy snow",
        80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
    }

    @classmethod
    def get_current(cls, lat: float, lon: float, timeout: int = 5) -> dict:
        """Fetch current weather for the given coordinates. Returns parsed JSON dict."""
        url = (
            f"{cls.BASE_URL}?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            f"&temperature_unit=fahrenheit"
        )
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    @classmethod
    def describe_code(cls, code: int) -> str:
        return cls.WMO_CODES.get(code, "unknown")


class StockAPIClient:
    """Yahoo Finance chart endpoint for current quote data."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    _HEADERS = {"User-Agent": "Mozilla/5.0"}

    @classmethod
    def get_quote(cls, symbol: str, timeout: int = 5) -> dict:
        """Fetch a 1-day chart for *symbol*. Returns parsed JSON dict."""
        url = f"{cls.BASE_URL}/{urllib.parse.quote(symbol)}?range=1d&interval=1d"
        req = urllib.request.Request(url, headers=cls._HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())


class HuggingFacePapersClient:
    """HuggingFace daily-papers API."""

    BASE_URL = "https://huggingface.co/api/daily_papers"
    _HEADERS = {"User-Agent": "Mozilla/5.0"}

    @classmethod
    def get_top(cls, date_str: str, limit: int = 1, timeout: int = 10) -> list:
        """Return list of papers for *date_str* (YYYY-MM-DD). May be empty."""
        url = f"{cls.BASE_URL}?date={urllib.parse.quote(date_str)}&limit={limit}"
        req = urllib.request.Request(url, headers=cls._HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())


class GoogleNewsClient:
    """Google News RSS feed."""

    BASE_URL = "https://news.google.com/rss"
    _HEADERS = {"User-Agent": "Mozilla/5.0"}

    # Google News topic tokens
    TOPIC_IDS = {
        "business":      "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        "technology":    "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
        "science":       "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB",
        "health":        "CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ",
        "sports":        "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB",
        "entertainment": "CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB",
    }

    @classmethod
    def get_feed(cls, topic: str = "", timeout: int = 10) -> bytes:
        """Return raw RSS XML bytes for *topic* (or top headlines if empty)."""
        if topic and topic in cls.TOPIC_IDS:
            url = f"{cls.BASE_URL}/topics/{cls.TOPIC_IDS[topic]}?hl=en-US&gl=US&ceid=US:en"
        else:
            url = f"{cls.BASE_URL}?hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers=cls._HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
