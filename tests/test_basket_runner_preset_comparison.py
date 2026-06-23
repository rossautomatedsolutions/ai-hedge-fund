from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from src.basket_runner import (
    BasketRunConfig,
    DECISION_SUMMARY_DISCLAIMER,
    RESEARCH_JOURNAL_FIELDNAMES,
    append_research_journal,
    build_arg_parser,
    main,
    parse_compare_presets,
    run_preset_comparison,
    write_research_packet,
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


def _comparison_rows_for_tickers(run_dir: Path, tickers: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    row_templates = {
        "technical-only": {
            "action": "buy",
            "confidence": "71",
            "analyst_vote_summary": "bullish=1, bearish=0, neutral=0",
            "analyst_consensus": "bullish",
            "comparison_note": "",
            "reasoning": "Technical breakout remains intact.",
        },
        "core": {
            "action": "buy",
            "confidence": "67",
            "analyst_vote_summary": "bullish=1, bearish=1, neutral=1",
            "analyst_consensus": "mixed",
            "comparison_note": "Action is directional despite mixed analyst votes.",
            "reasoning": "Balanced preset still shows hesitation.",
        },
        "no-news": {
            "action": "short",
            "confidence": "82",
            "analyst_vote_summary": "bullish=0, bearish=2, neutral=1",
            "analyst_consensus": "bearish",
            "comparison_note": "Reasoning should be read as portfolio-manager rationale, not analyst vote majority.",
            "reasoning": "Broader non-news inputs remain defensive.",
        },
        "all": {
            "action": "sell",
            "confidence": "79",
            "analyst_vote_summary": "bullish=1, bearish=3, neutral=1",
            "analyst_consensus": "bearish",
            "comparison_note": "",
            "reasoning": "All analysts together leaned bearish.",
        },
    }
    for ticker in tickers:
        for preset, template in row_templates.items():
            rows.append(
                {
                    "ticker": ticker,
                    "analyst_preset": preset,
                    "action": template["action"],
                    "quantity": "40",
                    "confidence": template["confidence"],
                    "analyst_vote_summary": template["analyst_vote_summary"],
                    "analyst_consensus": template["analyst_consensus"],
                    "comparison_note": template["comparison_note"],
                    "reasoning": template["reasoning"],
                    "data_status": "offline_demo",
                    "run_status": "success",
                    "failure_classification": "",
                    "run_dir": (run_dir / "preset_runs" / preset / ticker / "run").as_posix(),
                    "log_path": (run_dir / "preset_runs" / preset / ticker / "run" / "logs" / f"{ticker}.log").as_posix(),
                }
            )
    return rows


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


def test_build_arg_parser_accepts_research_packet_flag() -> None:
    args = build_arg_parser().parse_args(["--tickers", "BB", "--research-packet"])

    assert args.research_packet is True


def test_build_arg_parser_accepts_research_journal_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--tickers",
            "BB",
            "--research-packet",
            "--append-research-journal",
            "--research-journal-path",
            "outputs/research_journal.csv",
        ]
    )

    assert args.append_research_journal is True
    assert args.research_journal_path == "outputs/research_journal.csv"


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
                "reasoning": "majority bullish signals, moderate confidence",
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
    assert rows[0]["comparison_note"] == ""
    assert rows[1]["comparison_note"] == "Action is directional despite mixed analyst votes."
    assert rows[2]["comparison_note"] == "Reasoning should be read as portfolio-manager rationale, not analyst vote majority."
    assert rows[3]["comparison_note"] == ""
    assert DECISION_SUMMARY_DISCLAIMER in markdown
    assert OFFLINE_DEMO_DISCLAIMER in markdown
    assert "- technical-only: buy / 40 / bullish" in markdown
    assert "- core: buy / 40 / mixed / Action is directional despite mixed analyst votes." in markdown
    assert "- no-news: short / 85 / bearish / Reasoning should be read as portfolio-manager rationale, not analyst vote majority." in markdown
    assert "- all: sell / 80 / bearish" in markdown
    assert "| Preset | Action | Quantity | Confidence | Analyst Votes | Consensus | Comparison Note | Data Status | Run Status | Failure | Reasoning | Log |" in markdown


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


def test_write_research_packet_from_comparison_rows_includes_required_sections(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "comparison-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ticker": "BB",
            "analyst_preset": "technical-only",
            "action": "buy",
            "quantity": "40",
            "confidence": "71",
            "analyst_vote_summary": "bullish=1, bearish=0, neutral=0",
            "analyst_consensus": "bullish",
            "comparison_note": "",
            "reasoning": "Technical breakout remains intact.",
            "data_status": "offline_demo",
            "run_status": "success",
            "failure_classification": "",
            "run_dir": (run_dir / "preset_runs" / "technical-only" / "run").as_posix(),
            "log_path": (run_dir / "preset_runs" / "technical-only" / "run" / "logs" / "BB.log").as_posix(),
        },
        {
            "ticker": "BB",
            "analyst_preset": "core",
            "action": "buy",
            "quantity": "40",
            "confidence": "67",
            "analyst_vote_summary": "bullish=1, bearish=1, neutral=1",
            "analyst_consensus": "mixed",
            "comparison_note": "Action is directional despite mixed analyst votes.",
            "reasoning": "Balanced preset still shows hesitation.",
            "data_status": "offline_demo",
            "run_status": "success",
            "failure_classification": "",
            "run_dir": (run_dir / "preset_runs" / "core" / "run").as_posix(),
            "log_path": (run_dir / "preset_runs" / "core" / "run" / "logs" / "BB.log").as_posix(),
        },
        {
            "ticker": "BB",
            "analyst_preset": "no-news",
            "action": "short",
            "quantity": "85",
            "confidence": "82",
            "analyst_vote_summary": "bullish=0, bearish=2, neutral=1",
            "analyst_consensus": "bearish",
            "comparison_note": "Reasoning should be read as portfolio-manager rationale, not analyst vote majority.",
            "reasoning": "Broader non-news inputs remain defensive.",
            "data_status": "offline_demo",
            "run_status": "success",
            "failure_classification": "",
            "run_dir": (run_dir / "preset_runs" / "no-news" / "run").as_posix(),
            "log_path": (run_dir / "preset_runs" / "no-news" / "run" / "logs" / "BB.log").as_posix(),
        },
        {
            "ticker": "BB",
            "analyst_preset": "all",
            "action": "sell",
            "quantity": "80",
            "confidence": "79",
            "analyst_vote_summary": "bullish=1, bearish=3, neutral=1",
            "analyst_consensus": "bearish",
            "comparison_note": "",
            "reasoning": "All analysts together leaned bearish.",
            "data_status": "offline_demo",
            "run_status": "success",
            "failure_classification": "",
            "run_dir": (run_dir / "preset_runs" / "all" / "run").as_posix(),
            "log_path": (run_dir / "preset_runs" / "all" / "run" / "logs" / "BB.log").as_posix(),
        },
    ]

    artifacts = write_research_packet(
        run_dir,
        config=config,
        comparison_rows=rows,
        preset_run_dirs={row["analyst_preset"]: row["run_dir"] for row in rows},
    )

    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = artifacts.payload
    bb_packet = payload["tickers"][0]

    assert artifacts.markdown_path.name == "research_packet.md"
    assert artifacts.json_path.name == "research_packet.json"
    assert DECISION_SUMMARY_DISCLAIMER in markdown
    assert OFFLINE_DEMO_DISCLAIMER in markdown
    assert "This packet is based on static fixture data and should not be treated as current market analysis." in markdown
    assert "| technical-only | buy | 71 | bullish=1, bearish=0, neutral=0 | bullish | - | success |" in markdown
    assert "Technical-only is bullish while broader presets are mixed or bearish" in markdown
    assert "Action is directional despite mixed analyst votes." in markdown
    assert "Reasoning should be read as portfolio-manager rationale, not analyst vote majority." in markdown
    assert "### Bull Case" in markdown
    assert "### Bear Case" in markdown
    assert "### Key Disagreement Points" in markdown
    assert "### Data Limitations" in markdown
    assert "### What To Check Next Manually" in markdown
    assert "### Notable Risks / Reasons Not To Act" in markdown
    assert bb_packet["bull_case"]
    assert bb_packet["bear_case"]
    assert bb_packet["key_disagreement_points"]
    assert bb_packet["data_limitations"]
    assert bb_packet["what_to_check_next_manually"]
    assert bb_packet["notable_risks_or_reasons_not_to_act"]
    assert bb_packet["action_by_preset"]["technical-only"] == "buy"
    assert bb_packet["confidence_by_preset"]["all"] == "79"


def test_write_research_packet_single_preset_uses_existing_combined_decisions(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.analyst_preset = "core"
    run_dir = tmp_path / "single-run"
    _write_combined_decisions(
        run_dir,
        [
            {
                "ticker": "BB",
                "action": "hold",
                "quantity": "0",
                "confidence": "55",
                "reasoning": "Single-preset research angle only.",
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
            }
        ],
    )

    artifacts = write_research_packet(run_dir, config=config)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert "Presets analyzed: `core`" in markdown
    assert "Only one analyst preset was analyzed, so cross-preset disagreement is limited." in markdown


def test_research_journal_is_not_written_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "single-run"
    _write_combined_decisions(
        run_dir,
        [
            {
                "ticker": "BB",
                "action": "hold",
                "quantity": "0",
                "confidence": "55",
                "reasoning": "Single-preset research angle only.",
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
            }
        ],
    )
    default_journal = tmp_path / "research_journal.csv"

    monkeypatch.setattr("src.basket_runner.run_basket", lambda config: run_dir)
    monkeypatch.setattr("src.basket_runner._default_research_journal_path", lambda: default_journal)
    monkeypatch.setattr(
        sys,
        "argv",
        ["basket_runner.py", "--tickers", "BB", "--research-packet", "--output-dir", str(tmp_path)],
    )

    assert main() == 0
    assert not default_journal.exists()


def test_append_research_journal_creates_headers_when_requested(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "comparison-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_research_packet(run_dir, config=config, comparison_rows=_comparison_rows_for_tickers(run_dir, ["BB"]))
    journal_path = tmp_path / "journal.csv"

    append_research_journal(artifacts, journal_path=journal_path)

    rows = list(csv.DictReader(journal_path.open(encoding="utf-8", newline="")))
    header_line = journal_path.read_text(encoding="utf-8").splitlines()[0]

    assert journal_path.exists()
    assert header_line.split(",") == RESEARCH_JOURNAL_FIELDNAMES
    assert len(rows) == 1
    assert rows[0]["ticker"] == "BB"


def test_second_research_journal_append_adds_rows_without_duplicating_headers(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    journal_path = tmp_path / "journal.csv"

    first_run = tmp_path / "comparison-run-1"
    second_run = tmp_path / "comparison-run-2"
    first_run.mkdir(parents=True, exist_ok=True)
    second_run.mkdir(parents=True, exist_ok=True)

    first_artifacts = write_research_packet(first_run, config=config, comparison_rows=_comparison_rows_for_tickers(first_run, ["BB"]))
    second_artifacts = write_research_packet(second_run, config=config, comparison_rows=_comparison_rows_for_tickers(second_run, ["GME"]))

    append_research_journal(first_artifacts, journal_path=journal_path)
    append_research_journal(second_artifacts, journal_path=journal_path)

    lines = journal_path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(journal_path.open(encoding="utf-8", newline="")))

    assert sum(1 for line in lines if line.startswith("generated_at,")) == 1
    assert [row["ticker"] for row in rows] == ["BB", "GME"]


def test_multiple_tickers_append_multiple_journal_rows(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "comparison-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_research_packet(
        run_dir,
        config=config,
        comparison_rows=_comparison_rows_for_tickers(run_dir, ["BB", "GME"]),
    )
    journal_path = tmp_path / "journal.csv"

    journal_artifacts = append_research_journal(artifacts, journal_path=journal_path)
    rows = list(csv.DictReader(journal_path.open(encoding="utf-8", newline="")))

    assert journal_artifacts.rows_written == 2
    assert [row["ticker"] for row in rows] == ["BB", "GME"]


def test_custom_research_journal_path_is_used(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "comparison-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_research_packet(run_dir, config=config, comparison_rows=_comparison_rows_for_tickers(run_dir, ["BB"]))
    journal_path = tmp_path / "custom" / "research_journal.csv"

    journal_artifacts = append_research_journal(artifacts, journal_path=journal_path)

    assert journal_artifacts.journal_path == journal_path
    assert journal_path.exists()


def test_research_journal_json_and_list_fields_are_valid_json_strings(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "comparison-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_research_packet(run_dir, config=config, comparison_rows=_comparison_rows_for_tickers(run_dir, ["BB"]))
    journal_path = tmp_path / "journal.csv"

    append_research_journal(artifacts, journal_path=journal_path)

    row = next(csv.DictReader(journal_path.open(encoding="utf-8", newline="")))
    assert json.loads(row["presets_analyzed"]) == ["technical-only", "core", "no-news", "all"]
    assert isinstance(json.loads(row["action_by_preset"]), dict)
    assert isinstance(json.loads(row["confidence_by_preset"]), dict)
    assert isinstance(json.loads(row["consensus_by_preset"]), dict)
    assert isinstance(json.loads(row["comparison_notes"]), list)
    assert isinstance(json.loads(row["key_disagreement_points"]), list)
    assert isinstance(json.loads(row["data_limitations"]), list)
    assert isinstance(json.loads(row["what_to_check_next_manually"]), list)
    assert isinstance(json.loads(row["notable_risks_or_reasons_not_to_act"]), list)


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
    monkeypatch.setattr("src.basket_runner.write_research_packet", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("research packet should not run")))
    monkeypatch.setattr(sys, "argv", ["basket_runner.py", "--tickers", "BB", "--analyst-preset", "core"])

    assert main() == 0
    assert observed["analyst_preset"] == "core"
    assert observed["analysts"] == ["fundamentals_analyst", "technical_analyst", "valuation_analyst"]
