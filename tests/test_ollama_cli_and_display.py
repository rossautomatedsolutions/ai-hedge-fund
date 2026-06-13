from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src.basket_runner import BasketRunConfig, run_basket
from src.cli.input import resolve_model_selection, select_model
from src.data_diagnostics import run_ticker_data_check
from src.utils.display import print_trading_output


def test_resolve_model_selection_prefers_ollama_registry_for_ollama_flag() -> None:
    model_name, provider = resolve_model_selection(True, "qwen3:4b")
    assert model_name == "qwen3:4b"
    assert provider == "Ollama"


def test_resolve_model_selection_accepts_llama_alias_for_ollama() -> None:
    model_name, provider = resolve_model_selection(True, "llama3.1:8b")
    assert model_name == "llama3.1:latest"
    assert provider == "Ollama"


def test_select_model_non_interactive_uses_resolved_ollama_model(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[str, bool]] = []

    def fake_ensure(model_name: str, *, interactive: bool = True) -> bool:
        observed.append((model_name, interactive))
        return True

    monkeypatch.setattr("src.cli.input.ensure_ollama_and_model", fake_ensure)

    model_name, provider = select_model(True, "llama3.1", no_interactive=True)

    assert (model_name, provider) == ("llama3.1:latest", "Ollama")
    assert observed == [("llama3.1:latest", False)]


def test_print_trading_output_handles_malformed_payloads(capsys: pytest.CaptureFixture[str]) -> None:
    result = {
        "decisions": {
            "BB": {
                "action": None,
                "quantity": None,
                "confidence": None,
                "reasoning": ["line1", "line2"],
            }
        },
        "analyst_signals": {
            "fundamentals_analyst_agent": {
                "BB": {
                    "signal": None,
                    "confidence": None,
                    "reasoning": {"detail": "missing pieces"},
                }
            },
            "technical_analyst_agent": {
                "BB": ["unexpected", "list"],
            },
        },
    }

    print_trading_output(result)
    output = capsys.readouterr().out

    assert "UNKNOWN" in output
    assert "N/A" in output
    assert "WARNING" in output
    assert "BB" in output


def test_basket_runner_dry_run_writes_manifest_and_summary(tmp_path: Path) -> None:
    config = BasketRunConfig(
        tickers=["BB", "GME"],
        basket_name="speculative",
        model="llama3.1:latest",
        output_dir=str(tmp_path),
        max_symbols=None,
        continue_on_error=False,
        show_reasoning=False,
        start_date=None,
        end_date=None,
        dry_run=True,
        data_check_only=False,
        analysts=["fundamentals_analyst", "technical_analyst", "valuation_analyst"],
    )

    run_dir = run_basket(config)

    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.md"

    assert manifest_path.exists()
    assert summary_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tickers"] == ["BB", "GME"]
    assert manifest["dry_run"] is True
    assert (run_dir / "failures.json").exists()
    assert (run_dir / "data_check.json").exists()


def test_basket_runner_continue_on_error_writes_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.basket_runner.ensure_ollama_and_model", lambda model, interactive=False: True)
    monkeypatch.setattr("src.basket_runner.print_trading_output", lambda result: None)
    monkeypatch.setattr(
        "src.basket_runner.run_ticker_data_check",
        lambda ticker, start_date, end_date: type(
            "Check",
            (),
            {
                "ticker": ticker,
                "ok": True,
                "classification": "ok",
                "diagnosis": "All good",
                "env_var": "FINANCIAL_DATASETS_API_KEY",
                "checked_at": "2026-06-13T00:00:00",
                "checks": [],
            },
        )(),
    )
    monkeypatch.setattr(
        "src.basket_runner.ticker_data_check_to_dict",
        lambda result: {
            "ticker": result.ticker,
            "ok": result.ok,
            "classification": result.classification,
            "diagnosis": result.diagnosis,
            "env_var": result.env_var,
            "checked_at": result.checked_at,
            "checks": result.checks,
        },
    )
    monkeypatch.setattr("src.basket_runner.format_ticker_data_check", lambda result: json.dumps({"ticker": result.ticker}))

    calls = {"count": 0}

    def fake_run_hedge_fund(**kwargs):
        calls["count"] += 1
        ticker = kwargs["tickers"][0]
        if calls["count"] == 1:
            raise RuntimeError(f"boom-{ticker}")
        return {
            "decisions": {
                ticker: {
                    "action": "hold",
                    "quantity": 0,
                    "confidence": 50,
                    "reasoning": "ok",
                }
            },
            "analyst_signals": {},
        }

    monkeypatch.setattr("src.basket_runner.run_hedge_fund", fake_run_hedge_fund)

    config = BasketRunConfig(
        tickers=["BB", "GME"],
        basket_name="speculative",
        model="llama3.1:latest",
        output_dir=str(tmp_path),
        max_symbols=None,
        continue_on_error=True,
        show_reasoning=False,
        start_date=None,
        end_date=None,
        dry_run=False,
        data_check_only=False,
        analysts=["fundamentals_analyst"],
    )

    run_dir = run_basket(config)

    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    summary = (run_dir / "run_summary.md").read_text(encoding="utf-8")
    combined_csv = (run_dir / "combined_decisions.csv").read_text(encoding="utf-8")

    assert failures[0]["ticker"] == "BB"
    assert failures[0]["error"] == "boom-BB"
    assert failures[0]["classification"] == "unknown_error"
    assert "Failures" in summary
    assert "GME" in combined_csv


def test_run_ticker_data_check_classifies_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)

    def fake_get(url, headers, timeout):
        response = Mock()
        response.status_code = 401
        response.text = '{"error":"Missing API key","message":"Please include an X-API-KEY"}'
        response.json.return_value = {"error": "Missing API key", "message": "Please include an X-API-KEY"}
        return response

    monkeypatch.setattr("src.data_diagnostics.requests.get", fake_get)
    monkeypatch.setattr("src.data_diagnostics.requests.post", lambda url, headers, json, timeout: fake_get(url, headers, timeout))

    result = run_ticker_data_check("BB", "2026-01-01", "2026-06-01")

    assert result.ok is False
    assert result.classification == "missing_api_key"
    assert "FINANCIAL_DATASETS_API_KEY" in result.diagnosis


def test_run_ticker_data_check_accepts_public_http_200_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)

    def fake_get(url, headers, timeout):
        response = Mock()
        response.status_code = 200
        response.text = "ok"
        if "prices" in url:
            response.json.return_value = {"prices": [{"time": "2026-01-01T00:00:00Z"}]}
        elif "financial-metrics" in url:
            response.json.return_value = {"financial_metrics": [{"market_cap": 1}]}
        elif "company/facts" in url:
            response.json.return_value = {"company_facts": {"market_cap": 1}}
        else:
            response.json.return_value = {}
        return response

    def fake_post(url, headers, json, timeout):
        response = Mock()
        response.status_code = 200
        response.text = "ok"
        response.json.return_value = {"search_results": [{"ticker": "AAPL"}]}
        return response

    monkeypatch.setattr("src.data_diagnostics.requests.get", fake_get)
    monkeypatch.setattr("src.data_diagnostics.requests.post", fake_post)

    result = run_ticker_data_check("AAPL", "2026-01-01", "2026-06-01")

    assert result.ok is True
    assert result.partial_ok is False
    assert result.classification == "ok"


def test_run_ticker_data_check_classifies_unauthorized_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", "bad-key")

    def fake_get(url, headers, timeout):
        response = Mock()
        response.status_code = 401
        response.text = "Unauthorized"
        response.json.return_value = {"detail": "Unauthorized"}
        return response

    monkeypatch.setattr("src.data_diagnostics.requests.get", fake_get)
    monkeypatch.setattr("src.data_diagnostics.requests.post", lambda url, headers, json, timeout: fake_get(url, headers, timeout))

    result = run_ticker_data_check("BB", "2026-01-01", "2026-06-01")

    assert result.ok is False
    assert result.classification == "unauthorized_401"


def test_run_ticker_data_check_classifies_partial_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)

    def fake_get(url, headers, timeout):
        response = Mock()
        if "prices" in url:
            response.status_code = 200
            response.text = "ok"
            response.json.return_value = {"prices": [{"time": "2026-01-01T00:00:00Z"}]}
        elif "financial-metrics" in url:
            response.status_code = 401
            response.text = '{"error":"Missing API key","message":"Please include an X-API-KEY"}'
            response.json.return_value = {"error": "Missing API key", "message": "Please include an X-API-KEY"}
        elif "company/facts" in url:
            response.status_code = 401
            response.text = '{"error":"Missing API key","message":"Please include an X-API-KEY"}'
            response.json.return_value = {"error": "Missing API key", "message": "Please include an X-API-KEY"}
        else:
            response.status_code = 200
            response.text = "ok"
            response.json.return_value = {}
        return response

    def fake_post(url, headers, json, timeout):
        response = Mock()
        response.status_code = 200
        response.text = "ok"
        response.json.return_value = {"search_results": [{"ticker": "GOOGL"}]}
        return response

    monkeypatch.setattr("src.data_diagnostics.requests.get", fake_get)
    monkeypatch.setattr("src.data_diagnostics.requests.post", fake_post)

    result = run_ticker_data_check("GOOGL", "2026-01-01", "2026-06-01")

    assert result.ok is False
    assert result.partial_ok is True
    assert result.classification == "partial_data"


def test_run_ticker_data_check_classifies_empty_success_payload_as_missing_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)

    def fake_get(url, headers, timeout):
        response = Mock()
        response.status_code = 200
        response.text = "ok"
        if "prices" in url:
            response.json.return_value = {"prices": []}
        elif "financial-metrics" in url:
            response.json.return_value = {"financial_metrics": []}
        elif "company/facts" in url:
            response.json.return_value = {"company_facts": None}
        else:
            response.json.return_value = {}
        return response

    def fake_post(url, headers, json, timeout):
        response = Mock()
        response.status_code = 200
        response.text = "ok"
        response.json.return_value = {"search_results": []}
        return response

    monkeypatch.setattr("src.data_diagnostics.requests.get", fake_get)
    monkeypatch.setattr("src.data_diagnostics.requests.post", fake_post)

    result = run_ticker_data_check("ZZZZ", "2026-01-01", "2026-06-01")

    assert result.ok is False
    assert result.partial_ok is False
    assert result.classification == "missing_data"


def test_run_ticker_data_check_classifies_connection_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", "test-key")

    def fake_get(url, headers, timeout):
        raise requests.exceptions.ConnectionError("ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host')")

    monkeypatch.setattr("src.data_diagnostics.requests.get", fake_get)
    monkeypatch.setattr("src.data_diagnostics.requests.post", lambda url, headers, json, timeout: fake_get(url, headers, timeout))

    result = run_ticker_data_check("BB", "2026-01-01", "2026-06-01")

    assert result.ok is False
    assert result.classification == "connection_reset"


def test_basket_runner_data_check_only_writes_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.basket_runner.run_ticker_data_check",
        lambda ticker, start_date, end_date: type(
            "Check",
            (),
            {
                "ticker": ticker,
                "ok": ticker == "GME",
                "partial_ok": False,
                "classification": "missing_data" if ticker == "BB" else "ok",
                "diagnosis": "ticker unsupported" if ticker == "BB" else "All good",
                "env_var": "FINANCIAL_DATASETS_API_KEY",
                "checked_at": "2026-06-13T00:00:00",
                "checks": [],
            },
        )(),
    )
    monkeypatch.setattr(
        "src.basket_runner.ticker_data_check_to_dict",
        lambda result: {
            "ticker": result.ticker,
            "ok": result.ok,
            "partial_ok": result.partial_ok,
            "classification": result.classification,
            "diagnosis": result.diagnosis,
            "env_var": result.env_var,
            "checked_at": result.checked_at,
            "checks": result.checks,
        },
    )
    monkeypatch.setattr("src.basket_runner.format_ticker_data_check", lambda result: json.dumps({"ticker": result.ticker}))

    config = BasketRunConfig(
        tickers=["BB", "GME"],
        basket_name="speculative",
        model="llama3.1:latest",
        output_dir=str(tmp_path),
        max_symbols=None,
        continue_on_error=True,
        show_reasoning=False,
        start_date=None,
        end_date=None,
        dry_run=False,
        data_check_only=True,
        analysts=["fundamentals_analyst"],
    )

    run_dir = run_basket(config)
    data_check = json.loads((run_dir / "data_check.json").read_text(encoding="utf-8"))
    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    summary = (run_dir / "run_summary.md").read_text(encoding="utf-8")

    assert len(data_check) == 2
    assert failures[0]["classification"] == "missing_data"
    assert "Data checks failed" in summary
