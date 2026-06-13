from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import requests

from src.tools.api import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    FINANCIAL_DATASETS_API_KEY_ENV_VAR,
    FinancialDatasetsRequestSpec,
    build_financial_datasets_headers,
    get_financial_datasets_api_key,
    get_financial_datasets_request_specs,
)


FAILURE_TYPES = {
    "missing_api_key",
    "invalid_api_key",
    "unauthorized_401",
    "ssl_error",
    "connection_reset",
    "timeout",
    "missing_data",
    "partial_data",
    "unknown_error",
}


@dataclass
class EndpointCheckResult:
    name: str
    method: str
    url: str
    ok: bool
    classification: str
    diagnosis: str
    http_status: int | None = None
    record_count: int | None = None
    error: str | None = None


@dataclass
class TickerDataCheckResult:
    ticker: str
    ok: bool
    partial_ok: bool
    classification: str
    diagnosis: str
    env_var: str
    checked_at: str
    checks: list[EndpointCheckResult]


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {"raw": payload}
    except ValueError:
        return {}


def _extract_record_count(spec: FinancialDatasetsRequestSpec, response: requests.Response) -> int | None:
    payload = _safe_json(response)
    if spec.name == "prices":
        return len(payload.get("prices") or [])
    if spec.name == "financial_metrics":
        return len(payload.get("financial_metrics") or [])
    if spec.name == "line_items":
        return len(payload.get("search_results") or [])
    if spec.name == "company_facts":
        return 1 if payload.get("company_facts") else 0
    return None


def _extract_error_text(response: requests.Response) -> str:
    payload = _safe_json(response)
    text_parts: list[str] = []
    for key in ("message", "error", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())
    raw_text = (response.text or "").strip()
    if raw_text:
        text_parts.append(raw_text)
    return " | ".join(text_parts)


def _classify_response(
    spec: FinancialDatasetsRequestSpec,
    response: requests.Response,
    *,
    has_api_key: bool,
) -> tuple[str, str, int | None]:
    status_code = response.status_code
    if status_code == 200:
        record_count = _extract_record_count(spec, response)
        if record_count == 0:
            return (
                "missing_data",
                f"{spec.name} responded successfully but returned no records. The ticker may be unsupported or data may be unavailable.",
                record_count,
            )
        return ("ok", f"{spec.name} returned usable data.", record_count)

    if status_code == 401:
        error_text = _extract_error_text(response)
        lowered = error_text.lower()
        if "missing api key" in lowered and not has_api_key:
            return (
                "missing_api_key",
                f"`{FINANCIAL_DATASETS_API_KEY_ENV_VAR}` is not set, so {spec.name} cannot be authenticated.",
                None,
            )
        if has_api_key:
            return (
                "unauthorized_401",
                f"{spec.name} returned HTTP 401 with an API key present. The key may be invalid, expired, or unauthorized for this endpoint.",
                None,
            )
        return (
            "invalid_api_key",
            f"{spec.name} returned HTTP 401 without a usable public response. This likely requires a valid `{FINANCIAL_DATASETS_API_KEY_ENV_VAR}`.",
            None,
        )
    if status_code in {402, 403, 429}:
        return (
            "unknown_error",
            f"{spec.name} returned HTTP {status_code}. This may indicate a free-tier limitation, plan restriction, or temporary provider throttling.",
            None,
        )
    if status_code == 404:
        return (
            "missing_data",
            f"{spec.name} returned HTTP 404. The ticker may be unsupported or this dataset may be unavailable.",
            None,
        )
    if status_code >= 400:
        return (
            "unknown_error",
            f"{spec.name} returned HTTP {status_code}. Response body may contain more detail.",
            None,
        )
    return ("unknown_error", f"{spec.name} returned an unexpected HTTP status: {status_code}.", None)


def _classify_exception(spec: FinancialDatasetsRequestSpec, exc: Exception, *, has_api_key: bool) -> tuple[str, str]:
    if isinstance(exc, requests.exceptions.SSLError):
        return ("ssl_error", f"{spec.name} failed with an SSL/TLS error while contacting the provider.")
    if isinstance(exc, (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.Timeout)):
        return ("timeout", f"{spec.name} timed out while contacting the provider.")
    if isinstance(exc, requests.exceptions.ConnectionError):
        message = str(exc)
        if "10054" in message or "forcibly closed" in message or "ConnectionResetError" in message:
            return ("connection_reset", f"{spec.name} connection was reset by the remote host.")
        return ("unknown_error", f"{spec.name} hit a network connection error: {message}")
    return ("unknown_error", f"{spec.name} failed with an unexpected error: {exc}")


def _run_request_check(spec: FinancialDatasetsRequestSpec, *, api_key: str | None) -> EndpointCheckResult:
    headers = build_financial_datasets_headers(api_key)
    has_api_key = bool(get_financial_datasets_api_key(api_key))
    try:
        if spec.method.upper() == "POST":
            response = requests.post(spec.url, headers=headers, json=spec.json_data, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
        else:
            response = requests.get(spec.url, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
        classification, diagnosis, record_count = _classify_response(spec, response, has_api_key=has_api_key)
        return EndpointCheckResult(
            name=spec.name,
            method=spec.method,
            url=spec.url,
            ok=classification == "ok",
            classification=classification,
            diagnosis=diagnosis,
            http_status=response.status_code,
            record_count=record_count,
            error=None if classification == "ok" else response.text[:300] or None,
        )
    except Exception as exc:
        classification, diagnosis = _classify_exception(spec, exc, has_api_key=has_api_key)
        return EndpointCheckResult(
            name=spec.name,
            method=spec.method,
            url=spec.url,
            ok=False,
            classification=classification,
            diagnosis=diagnosis,
            http_status=None,
            record_count=None,
            error=str(exc),
        )


def run_ticker_data_check(ticker: str, start_date: str, end_date: str, *, api_key: str | None = None) -> TickerDataCheckResult:
    specs = get_financial_datasets_request_specs(ticker, start_date, end_date)
    checks = [_run_request_check(spec, api_key=api_key) for spec in specs]
    successes = [check for check in checks if check.ok]
    failures = [check for check in checks if not check.ok]

    if successes and not failures:
        ok = True
        partial_ok = False
        classification = "ok"
        diagnosis = "All required price/fundamental/financial data checks succeeded."
    elif successes and failures:
        ok = False
        partial_ok = True
        classification = "partial_data"
        diagnosis = "Some required data endpoints succeeded while others failed. Review per-endpoint results for coverage gaps."
    else:
        ok = False
        partial_ok = False
        failure_kinds = {check.classification for check in failures}
        if failure_kinds == {"missing_api_key"}:
            classification = "missing_api_key"
            diagnosis = f"All required endpoints failed because `{FINANCIAL_DATASETS_API_KEY_ENV_VAR}` is not set."
        elif failure_kinds <= {"unauthorized_401", "invalid_api_key"}:
            classification = "unauthorized_401"
            diagnosis = "All required endpoints returned HTTP 401 with authentication failures."
        elif failure_kinds == {"missing_data"}:
            classification = "missing_data"
            diagnosis = "All required endpoints responded without usable records for this ticker."
        else:
            primary = failures[0]
            classification = primary.classification
            diagnosis = primary.diagnosis

    return TickerDataCheckResult(
        ticker=ticker,
        ok=ok,
        partial_ok=partial_ok,
        classification=classification,
        diagnosis=diagnosis,
        env_var=FINANCIAL_DATASETS_API_KEY_ENV_VAR,
        checked_at=datetime.now().isoformat(),
        checks=checks,
    )


def ticker_data_check_to_dict(result: TickerDataCheckResult) -> dict[str, Any]:
    return asdict(result)


def format_ticker_data_check(result: TickerDataCheckResult) -> str:
    return json.dumps(ticker_data_check_to_dict(result), indent=2, ensure_ascii=True)
