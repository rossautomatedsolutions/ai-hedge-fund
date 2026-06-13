from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.basket_runner import BasketRunConfig, run_basket
from src.cli.input import resolve_model_selection, select_model
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


def test_basket_runner_continue_on_error_writes_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.basket_runner.ensure_ollama_and_model", lambda model, interactive=False: True)
    monkeypatch.setattr("src.basket_runner.print_trading_output", lambda result: None)

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
        analysts=["fundamentals_analyst"],
    )

    run_dir = run_basket(config)

    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    summary = (run_dir / "run_summary.md").read_text(encoding="utf-8")
    combined_csv = (run_dir / "combined_decisions.csv").read_text(encoding="utf-8")

    assert failures == [{"ticker": "BB", "error": "boom-BB"}]
    assert "Failures" in summary
    assert "GME" in combined_csv
