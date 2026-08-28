from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from src.basket_runner import BasketRunConfig, main, run_basket
from src.signal_artifacts import SIGNAL_CAPTURE_FILENAME, export_signal_ledger_bundle


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_source_run_dir(
    base_dir: Path,
    *,
    offline_demo_data: bool = False,
    end_date: str = "2020-01-15",
) -> Path:
    run_dir = base_dir / "source_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "created_at": "2026-08-28T12:00:00",
            "basket_name": "large-cap",
            "tickers": ["AAPL"],
            "model": "llama3.1:latest",
            "model_provider": "Ollama",
            "analysts": ["fundamentals_analyst", "technical_analyst", "valuation_analyst"],
            "continue_on_error": True,
            "show_reasoning": False,
            "start_date": "2019-10-15",
            "end_date": end_date,
            "dry_run": False,
            "data_check_only": False,
            "analyst_preset": "core",
            "request_timeout_seconds": 15,
            "max_data_retries": 3,
            "fast_data_mode": False,
            "offline_demo_data": offline_demo_data,
            "skip_optional_slow_data": False,
        },
    )
    _write_json(
        run_dir / SIGNAL_CAPTURE_FILENAME,
        {
            "run_id": "source_run",
            "basket_name": "large-cap",
            "analyst_preset": "core",
            "tickers": ["AAPL"],
            "analysts": ["fundamentals_analyst", "technical_analyst", "valuation_analyst"],
            "records": [
                {
                    "ticker": "AAPL",
                    "run_status": "success",
                    "data_check": {
                        "ticker": "AAPL",
                        "ok": True,
                        "classification": "ok",
                        "diagnosis": "All good",
                    },
                    "failure": None,
                    "result": {
                        "decisions": {
                            "AAPL": {
                                "action": "buy",
                                "quantity": 10,
                                "confidence": 82,
                                "reasoning": "Momentum and fundamentals align",
                            }
                        },
                        "analyst_signals": {
                            "fundamentals_analyst_agent": {
                                "AAPL": {
                                    "signal": "bullish",
                                    "confidence": 90,
                                    "reasoning": {"summary": "Strong profitability"},
                                }
                            },
                            "technical_analyst_agent": {
                                "AAPL": {
                                    "signal": "bearish",
                                    "confidence": 40,
                                    "reasoning": {"summary": "Momentum fading"},
                                }
                            },
                        },
                    },
                }
            ],
        },
    )
    return run_dir


def test_run_basket_writes_signal_capture_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
                "checked_at": "2026-08-28T12:00:00",
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
    monkeypatch.setattr(
        "src.basket_runner.run_hedge_fund",
        lambda **kwargs: {
            "decisions": {
                "AAPL": {
                    "action": "buy",
                    "quantity": 10,
                    "confidence": 82,
                    "reasoning": "Momentum and fundamentals align",
                }
            },
            "analyst_signals": {
                "fundamentals_analyst_agent": {"AAPL": {"signal": "bullish", "confidence": 90}},
                "technical_analyst_agent": {"AAPL": {"signal": "bearish", "confidence": 40}},
            },
        },
    )

    config = BasketRunConfig(
        tickers=["AAPL"],
        basket_name="large-cap",
        model="llama3.1:latest",
        output_dir=str(tmp_path),
        max_symbols=None,
        continue_on_error=True,
        show_reasoning=False,
        start_date="2020-01-01",
        end_date="2020-01-15",
        dry_run=False,
        data_check_only=False,
        analyst_preset="core",
        analysts=["fundamentals_analyst", "technical_analyst", "valuation_analyst"],
        request_timeout_seconds=15,
        max_data_retries=3,
        fast_data_mode=False,
    )

    run_dir = run_basket(config)

    payload = json.loads((run_dir / SIGNAL_CAPTURE_FILENAME).read_text(encoding="utf-8"))
    assert payload["run_id"] == run_dir.name
    assert payload["analyst_preset"] == "core"
    assert payload["records"][0]["run_status"] == "success"
    assert payload["records"][0]["result"]["decisions"]["AAPL"]["action"] == "buy"
    assert payload["records"][0]["result"]["analyst_signals"]["technical_analyst_agent"]["AAPL"]["signal"] == "bearish"


def test_export_signal_ledger_bundle_writes_explicit_abstains_and_ineligible_rows(tmp_path: Path) -> None:
    run_dir = _write_source_run_dir(tmp_path)

    artifacts = export_signal_ledger_bundle(run_dir=run_dir)

    records = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    rows_by_family = {row["analyst_family"]: row for row in records}

    assert set(rows_by_family) == {
        "fundamentals_analyst",
        "technical_analyst",
        "valuation_analyst",
        "portfolio_manager",
    }
    assert rows_by_family["fundamentals_analyst"]["signal"] == "bullish"
    assert rows_by_family["fundamentals_analyst"]["signal_score"] == 1
    assert rows_by_family["fundamentals_analyst"]["source_signal_label"] == "bullish"
    assert rows_by_family["fundamentals_analyst"]["data_mode"] == "historical_live_research_unverified_pit"
    assert rows_by_family["fundamentals_analyst"]["data_cutoff"] is None
    assert rows_by_family["fundamentals_analyst"]["is_backtest_eligible"] is False
    assert rows_by_family["fundamentals_analyst"]["backtest_ineligibility_reason"] == "historical_window_not_proven_point_in_time_safe"

    assert rows_by_family["technical_analyst"]["signal"] == "bearish"
    assert rows_by_family["technical_analyst"]["price_data_cutoff"] is None

    assert rows_by_family["valuation_analyst"]["signal"] == "abstain"
    assert rows_by_family["valuation_analyst"]["evidence_available"] is False
    assert rows_by_family["valuation_analyst"]["ticker_status"] == "missing_signal"
    assert rows_by_family["valuation_analyst"]["partial_failure"] is True
    assert "did not emit a signal" in rows_by_family["valuation_analyst"]["failure_reason"]

    assert rows_by_family["portfolio_manager"]["signal"] == "bullish"
    assert rows_by_family["portfolio_manager"]["source_signal_label"] == "buy"
    assert rows_by_family["portfolio_manager"]["partial_failure"] is True
    assert rows_by_family["portfolio_manager"]["backtest_ineligibility_reason"] == "partial_or_incomplete_signal_evidence"

    trades = list(csv.DictReader(artifacts.trading_foundation_trades_path.open(encoding="utf-8", newline="")))
    assert trades == []

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 4
    assert manifest["eligible_record_count"] == 0
    assert manifest["selected_integration_path"]["selected_engine_entrypoint"] == "backtesting.run_backtest_from_trades.run"


def test_export_signal_ledger_bundle_marks_offline_demo_rows_ineligible(tmp_path: Path) -> None:
    run_dir = _write_source_run_dir(tmp_path, offline_demo_data=True)

    artifacts = export_signal_ledger_bundle(run_dir=run_dir)
    records = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    fundamentals_row = next(row for row in records if row["analyst_family"] == "fundamentals_analyst")

    assert fundamentals_row["data_mode"] == "offline_demo"
    assert fundamentals_row["is_backtest_eligible"] is False
    assert fundamentals_row["backtest_ineligibility_reason"] == "offline_demo_fixture_not_historical_evidence"


def test_main_signal_ledger_source_run_dir_exports_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _write_source_run_dir(tmp_path)
    output_dir = tmp_path / "exported_bundle"

    monkeypatch.setattr(
        "src.basket_runner.run_basket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_basket should not run in export-only mode")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--export-signal-ledger",
            "--signal-ledger-source-run-dir",
            str(run_dir),
            "--signal-ledger-output",
            str(output_dir),
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signal_ledger_csv"] == (output_dir / "signal_ledger.csv").as_posix()
    assert payload["signal_ledger_record_count"] == 4
    assert payload["trading_foundation_prepared_trade_row_count"] == 0
