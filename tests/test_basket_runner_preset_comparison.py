from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from src.basket_runner import (
    BasketRunConfig,
    DECISION_SUMMARY_DISCLAIMER,
    build_arg_parser,
    main,
    parse_compare_presets,
    run_preset_comparison,
)
from src.offline_demo_data import OFFLINE_DEMO_DISCLAIMER


def _base_config(tmp_path: Path) -> BasketRunConfig:
    return BasketRunConfig(
        tickers=["BB"],
        basket_name="speculative",
        model="llama3.1:latest",
        output_dir=str(tmp_path),
        max_symbols=None,
        continue_on_error=True,
        show_reasoning=False,
        start_date="2026-03-01",
        end_date="2026-03-09",
        dry_run=False,
        data_check_only=False,
        analyst_preset="all",
        analysts=["fundamentals_analyst"],
        request_timeout_seconds=15,
        max_data_retries=3,
        fast_data_mode=True,
        offline_demo_data=True,
    )


def _write_combined_decisions(run_dir: Path, rows: list[dict[str, str]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (run_dir / "combined_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "action",
                "quantity",
                "confidence",
                "reasoning",
                "analyst_vote_summary",
                "analyst_consensus",
                "report_note",
                "analyst_preset",
                "model",
                "provider",
                "data_status",
                "failure_classification",
                "bullish_count",
                "bearish_count",
                "neutral_count",
                "run_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_arg_parser_accepts_compare_presets() -> None:
    args = build_arg_parser().parse_args(
        [
            "--tickers",
            "BB",
            "--compare-presets",
            "technical-only,core,no-news,all",
        ]
    )

    assert args.compare_presets == "technical-only,core,no-news,all"
    assert parse_compare_presets(args.compare_presets) == ["technical-only", "core", "no-news", "all"]


def test_run_preset_comparison_writes_rows_and_disclaimers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)

    def fake_run_basket(run_config: BasketRunConfig) -> Path:
        run_dir = Path(run_config.output_dir) / "run"
        row_by_preset = {
            "technical-only": {
                "ticker": "BB",
                "action": "buy",
                "quantity": "40",
                "confidence": "71",
                "reasoning": "Technical momentum only.",
                "analyst_vote_summary": "bullish=1, bearish=0, neutral=0",
                "analyst_consensus": "bullish",
                "report_note": OFFLINE_DEMO_DISCLAIMER,
                "analyst_preset": "technical-only",
                "model": "llama3.1:latest",
                "provider": "Ollama",
                "data_status": "offline_demo",
                "failure_classification": "",
                "bullish_count": "1",
                "bearish_count": "0",
                "neutral_count": "0",
                "run_status": "success",
            },
            "core": {
                "ticker": "BB",
                "action": "buy",
                "quantity": "40",
                "confidence": "67",
                "reasoning": "Core preset balanced outcome.",
                "analyst_vote_summary": "bullish=1, bearish=1, neutral=1",
                "analyst_consensus": "mixed",
                "report_note": OFFLINE_DEMO_DISCLAIMER,
                "analyst_preset": "core",
                "model": "llama3.1:latest",
                "provider": "Ollama",
                "data_status": "offline_demo",
                "failure_classification": "",
                "bullish_count": "1",
                "bearish_count": "1",
                "neutral_count": "1",
                "run_status": "success",
            },
            "no-news": {
                "ticker": "BB",
                "action": "short",
                "quantity": "85",
                "confidence": "82",
                "reasoning": "Removing news tilted bearish.",
                "analyst_vote_summary": "bullish=0, bearish=2, neutral=1",
                "analyst_consensus": "bearish",
                "report_note": OFFLINE_DEMO_DISCLAIMER,
                "analyst_preset": "no-news",
                "model": "llama3.1:latest",
                "provider": "Ollama",
                "data_status": "offline_demo",
                "failure_classification": "",
                "bullish_count": "0",
                "bearish_count": "2",
                "neutral_count": "1",
                "run_status": "success",
            },
            "all": {
                "ticker": "BB",
                "action": "sell",
                "quantity": "80",
                "confidence": "79",
                "reasoning": "All analysts together leaned bearish.",
                "analyst_vote_summary": "bullish=1, bearish=3, neutral=1",
                "analyst_consensus": "bearish",
                "report_note": OFFLINE_DEMO_DISCLAIMER,
                "analyst_preset": "all",
                "model": "llama3.1:latest",
                "provider": "Ollama",
                "data_status": "offline_demo",
                "failure_classification": "",
                "bullish_count": "1",
                "bearish_count": "3",
                "neutral_count": "1",
                "run_status": "success",
            },
        }
        _write_combined_decisions(run_dir, [row_by_preset[run_config.analyst_preset]])
        return run_dir

    monkeypatch.setattr("src.basket_runner.run_basket", fake_run_basket)

    artifacts = run_preset_comparison(config, ["technical-only", "core", "no-news", "all"])

    rows = list(csv.DictReader(artifacts.csv_path.open(encoding="utf-8", newline="")))
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert [row["analyst_preset"] for row in rows] == ["technical-only", "core", "no-news", "all"]
    assert rows[0]["action"] == "buy"
    assert rows[1]["analyst_consensus"] == "mixed"
    assert rows[2]["action"] == "short"
    assert rows[3]["action"] == "sell"
    assert DECISION_SUMMARY_DISCLAIMER in markdown
    assert OFFLINE_DEMO_DISCLAIMER in markdown
    assert "- technical-only: buy / 40 / bullish" in markdown
    assert "- core: buy / 40 / mixed" in markdown
    assert "- no-news: short / 85 / bearish" in markdown
    assert "- all: sell / 80 / bearish" in markdown


def test_run_preset_comparison_records_failure_rows_without_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)

    def fake_run_basket(run_config: BasketRunConfig) -> Path:
        if run_config.analyst_preset == "core":
            raise RuntimeError("preset exploded")
        run_dir = Path(run_config.output_dir) / "run"
        _write_combined_decisions(
            run_dir,
            [
                {
                    "ticker": "BB",
                    "action": "buy",
                    "quantity": "10",
                    "confidence": "60",
                    "reasoning": "ok",
                    "analyst_vote_summary": "bullish=1, bearish=0, neutral=0",
                    "analyst_consensus": "bullish",
                    "report_note": "",
                    "analyst_preset": run_config.analyst_preset,
                    "model": "llama3.1:latest",
                    "provider": "Ollama",
                    "data_status": "offline_demo",
                    "failure_classification": "",
                    "bullish_count": "1",
                    "bearish_count": "0",
                    "neutral_count": "0",
                    "run_status": "success",
                }
            ],
        )
        return run_dir

    monkeypatch.setattr("src.basket_runner.run_basket", fake_run_basket)

    artifacts = run_preset_comparison(config, ["technical-only", "core", "all"])
    rows = list(csv.DictReader(artifacts.csv_path.open(encoding="utf-8", newline="")))

    assert [row["analyst_preset"] for row in rows] == ["technical-only", "core", "all"]
    assert rows[1]["run_status"] == "failed"
    assert rows[1]["failure_classification"] == "unknown_error"
    assert rows[2]["run_status"] == "success"


def test_main_without_compare_presets_keeps_single_preset_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_run_basket(config: BasketRunConfig) -> Path:
        observed["analyst_preset"] = config.analyst_preset
        observed["analysts"] = list(config.analysts)
        return tmp_path / "single-run"

    def fail_compare(*args, **kwargs):
        raise AssertionError("compare path should not run")

    monkeypatch.setattr("src.basket_runner.run_basket", fake_run_basket)
    monkeypatch.setattr("src.basket_runner.run_preset_comparison", fail_compare)
    monkeypatch.setattr(sys, "argv", ["basket_runner.py", "--tickers", "BB", "--analyst-preset", "core"])

    assert main() == 0
    assert observed["analyst_preset"] == "core"
    assert observed["analysts"] == ["fundamentals_analyst", "technical_analyst", "valuation_analyst"]
