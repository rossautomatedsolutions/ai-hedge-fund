import datetime
import json
import logging
import os
import pandas as pd
import requests
import time
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)
from src.offline_demo_data import has_offline_demo_fixture, load_offline_demo_fixture

# Global cache instance
_cache = get_cache()

FINANCIAL_DATASETS_API_KEY_ENV_VAR = "FINANCIAL_DATASETS_API_KEY"
FINANCIAL_DATASETS_BASE_URL = "https://api.financialdatasets.ai"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15
DEFAULT_REQUEST_MAX_ATTEMPTS = 3
DEFAULT_SKIP_OPTIONAL_SLOW_DATA = False

_request_timeout_seconds = DEFAULT_REQUEST_TIMEOUT_SECONDS
_request_max_attempts = DEFAULT_REQUEST_MAX_ATTEMPTS
_skip_optional_slow_data = DEFAULT_SKIP_OPTIONAL_SLOW_DATA
_offline_demo_data_mode = False


@dataclass(frozen=True)
class FinancialDatasetsRequestSpec:
    name: str
    method: str
    url: str
    json_data: dict | None = None


@dataclass(frozen=True)
class FinancialDataRequestSettings:
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_REQUEST_MAX_ATTEMPTS
    skip_optional_slow_data: bool = DEFAULT_SKIP_OPTIONAL_SLOW_DATA


def get_financial_datasets_api_key(api_key: str | None = None) -> str | None:
    return api_key or os.environ.get(FINANCIAL_DATASETS_API_KEY_ENV_VAR)


def get_financial_data_request_settings() -> FinancialDataRequestSettings:
    return FinancialDataRequestSettings(
        timeout_seconds=_request_timeout_seconds,
        max_attempts=_request_max_attempts,
        skip_optional_slow_data=_skip_optional_slow_data,
    )


def is_offline_demo_data_mode_enabled() -> bool:
    return _offline_demo_data_mode


@contextmanager
def financial_data_request_settings(
    *,
    timeout_seconds: int | None = None,
    max_attempts: int | None = None,
    skip_optional_slow_data: bool | None = None,
):
    global _request_timeout_seconds, _request_max_attempts, _skip_optional_slow_data

    previous = get_financial_data_request_settings()
    if timeout_seconds is not None:
        _request_timeout_seconds = timeout_seconds
    if max_attempts is not None:
        _request_max_attempts = max_attempts
    if skip_optional_slow_data is not None:
        _skip_optional_slow_data = skip_optional_slow_data

    try:
        yield get_financial_data_request_settings()
    finally:
        _request_timeout_seconds = previous.timeout_seconds
        _request_max_attempts = previous.max_attempts
        _skip_optional_slow_data = previous.skip_optional_slow_data


@contextmanager
def offline_demo_data_mode(enabled: bool = False):
    global _offline_demo_data_mode

    previous = _offline_demo_data_mode
    _offline_demo_data_mode = enabled
    try:
        yield enabled
    finally:
        _offline_demo_data_mode = previous


def _offline_demo_fixture_payload(ticker: str) -> dict | None:
    if not _offline_demo_data_mode:
        return None
    normalized_ticker = ticker.upper()
    if not has_offline_demo_fixture(normalized_ticker):
        return None
    return load_offline_demo_fixture(normalized_ticker)


def build_financial_datasets_headers(api_key: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    financial_api_key = get_financial_datasets_api_key(api_key)
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key
    return headers


def get_financial_datasets_request_specs(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    financial_metrics_period: str = "ttm",
    financial_metrics_limit: int = 5,
    line_items: list[str] | None = None,
    line_item_period: str = "ttm",
    line_item_limit: int = 5,
) -> list[FinancialDatasetsRequestSpec]:
    requested_line_items = line_items or ["revenue", "net_income", "free_cash_flow", "outstanding_shares"]
    return [
        FinancialDatasetsRequestSpec(
            name="prices",
            method="GET",
            url=f"{FINANCIAL_DATASETS_BASE_URL}/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}",
        ),
        FinancialDatasetsRequestSpec(
            name="financial_metrics",
            method="GET",
            url=f"{FINANCIAL_DATASETS_BASE_URL}/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={financial_metrics_limit}&period={financial_metrics_period}",
        ),
        FinancialDatasetsRequestSpec(
            name="line_items",
            method="POST",
            url=f"{FINANCIAL_DATASETS_BASE_URL}/financials/search/line-items",
            json_data={
                "tickers": [ticker],
                "line_items": requested_line_items,
                "end_date": end_date,
                "period": line_item_period,
                "limit": line_item_limit,
            },
        ),
        FinancialDatasetsRequestSpec(
            name="company_facts",
            method="GET",
            url=f"{FINANCIAL_DATASETS_BASE_URL}/company/facts/?ticker={ticker}",
        ),
    ]


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int | None = None) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    normalized_method = method.upper()
    cache_key = _build_request_cache_key(normalized_method, url, json_data=json_data)
    cached_payload = _cache.get_request_response(cache_key)
    if _is_cacheable_payload(cached_payload):
        logger.debug("Using cached financial data response for %s %s", normalized_method, url)
        return _build_cached_response(url, cached_payload)

    effective_max_retries = max_retries if max_retries is not None else max(_request_max_attempts - 1, 0)
    max_attempts = effective_max_retries + 1
    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        try:
            if normalized_method == "POST":
                response = requests.post(url, headers=headers, json=json_data, timeout=_request_timeout_seconds)
            else:
                response = requests.get(url, headers=headers, timeout=_request_timeout_seconds)
        except requests.exceptions.RequestException as exc:
            last_exception = exc
            if _should_retry_exception(exc) and attempt < max_attempts - 1:
                delay = _retry_backoff_delay_seconds(attempt)
                logger.warning(
                    "Transient financial data request failure for %s %s on attempt %s/%s: %s. Retrying in %.1fs.",
                    normalized_method,
                    url,
                    attempt + 1,
                    max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            raise

        if _should_retry_response(response) and attempt < max_attempts - 1:
            delay = _retry_backoff_delay_seconds(attempt)
            logger.warning(
                "Transient financial data response for %s %s on attempt %s/%s: HTTP %s. Retrying in %.1fs.",
                normalized_method,
                url,
                attempt + 1,
                max_attempts,
                response.status_code,
                delay,
            )
            time.sleep(delay)
            continue

        if response.status_code == 200:
            try:
                parsed_payload = response.json()
            except ValueError:
                logger.debug("Skipping cache for %s %s because response JSON could not be parsed.", normalized_method, url)
            else:
                if _is_cacheable_payload(parsed_payload):
                    _cache.set_request_response(cache_key, parsed_payload)
        return response

    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"Financial data request exhausted retries without a response: {normalized_method} {url}")


def _normalize_json_for_cache(json_data: dict | None) -> str:
    """Return a stable string representation for POST request bodies."""
    if json_data is None:
        return ""
    return json.dumps(json_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_request_cache_key(method: str, url: str, json_data: dict | None = None) -> str:
    """Build an exact-match cache key for a request."""
    return f"{method.upper()} {url} {_normalize_json_for_cache(json_data)}"


def _is_cacheable_payload(payload: object) -> bool:
    """Allow only JSON-serializable parsed payloads in the request cache."""
    if payload is None:
        return False
    try:
        json.dumps(payload, ensure_ascii=True)
    except (TypeError, ValueError):
        return False
    return True


def _build_cached_response(url: str, payload: dict | list | str | int | float | bool | None) -> requests.Response:
    """Create a lightweight Response object from cached JSON payload data."""
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response.encoding = "utf-8"
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    return response


def _retry_backoff_delay_seconds(attempt: int) -> float:
    """Return exponential backoff delay for zero-based retry attempt."""
    return float(2**attempt)


def _should_retry_response(response: requests.Response) -> bool:
    """Retry only transient HTTP status codes."""
    return response.status_code == 429 or 500 <= response.status_code <= 599


def _should_retry_exception(exc: requests.exceptions.RequestException) -> bool:
    """Retry transient network/SSL/timeouts, but not deterministic request failures."""
    if isinstance(exc, requests.exceptions.SSLError):
        return True
    if isinstance(exc, (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        message = str(exc).lower()
        if "10054" in message or "forcibly closed" in message or "connectionreseterror" in message:
            return True
        return True
    return False


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    if fixture_payload := _offline_demo_fixture_payload(ticker):
        return [Price(**price) for price in fixture_payload.get("prices", [])]

    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # If not in cache, fetch from API
    headers = build_financial_datasets_headers(api_key)
    url = f"{FINANCIAL_DATASETS_BASE_URL}/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        return []

    # Parse response with Pydantic model
    try:
        price_response = PriceResponse(**response.json())
        prices = price_response.prices
    except Exception as e:
        logger.warning("Failed to parse price response for %s: %s", ticker, e)
        return []

    if not prices:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    if fixture_payload := _offline_demo_fixture_payload(ticker):
        metrics = [FinancialMetrics(**metric) for metric in fixture_payload.get("financial_metrics", [])]
        return metrics[:limit]

    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # If not in cache, fetch from API
    headers = build_financial_datasets_headers(api_key)
    url = f"{FINANCIAL_DATASETS_BASE_URL}/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        return []

    # Parse response with Pydantic model
    try:
        metrics_response = FinancialMetricsResponse(**response.json())
        financial_metrics = metrics_response.financial_metrics
    except Exception as e:
        logger.warning("Failed to parse financial metrics response for %s: %s", ticker, e)
        return []

    if not financial_metrics:
        return []

    # Cache the results as dicts using the comprehensive cache key
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API."""
    if fixture_payload := _offline_demo_fixture_payload(ticker):
        line_item_models = [LineItem(**line_item) for line_item in fixture_payload.get("line_items", [])]
        return line_item_models[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = build_financial_datasets_headers(api_key)
    url = f"{FINANCIAL_DATASETS_BASE_URL}/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response.status_code != 200:
        return []
    
    try:
        data = response.json()
        response_model = LineItemResponse(**data)
        search_results = response_model.search_results
    except Exception as e:
        logger.warning("Failed to parse line items response for %s: %s", ticker, e)
        return []
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    if fixture_payload := _offline_demo_fixture_payload(ticker):
        return [InsiderTrade(**trade) for trade in fixture_payload.get("insider_trades", [])][:limit]

    if _skip_optional_slow_data:
        logger.info("Skipping insider trades for %s because optional slow data is disabled.", ticker)
        return []

    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # If not in cache, fetch from API
    headers = build_financial_datasets_headers(api_key)

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"{FINANCIAL_DATASETS_BASE_URL}/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except Exception as e:
            logger.warning("Failed to parse insider trades response for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    if fixture_payload := _offline_demo_fixture_payload(ticker):
        return [CompanyNews(**news) for news in fixture_payload.get("company_news", [])][:limit]

    if _skip_optional_slow_data:
        logger.info("Skipping company news for %s because optional slow data is disabled.", ticker)
        return []

    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # If not in cache, fetch from API
    headers = build_financial_datasets_headers(api_key)

    all_news = []
    current_end_date = end_date

    while True:
        url = f"{FINANCIAL_DATASETS_BASE_URL}/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except Exception as e:
            logger.warning("Failed to parse company news response for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from the API."""
    if fixture_payload := _offline_demo_fixture_payload(ticker):
        company_facts = fixture_payload.get("company_facts") or {}
        market_cap = company_facts.get("market_cap")
        if market_cap is not None:
            return float(market_cap)
        metrics = fixture_payload.get("financial_metrics") or []
        if metrics:
            metric_market_cap = metrics[0].get("market_cap")
            if metric_market_cap is not None:
                return float(metric_market_cap)
        return None

    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = build_financial_datasets_headers(api_key)
        url = f"{FINANCIAL_DATASETS_BASE_URL}/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
