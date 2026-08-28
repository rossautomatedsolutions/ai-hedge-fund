from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.basket_runner import (
    BasketRunConfig,
    DECISION_SUMMARY_DISCLAIMER,
    HUMAN_REVIEW_LOG_FIELDNAMES,
    RESEARCH_JOURNAL_FIELDNAMES,
    append_research_journal,
    build_research_watchlist,
    build_validation_checklist,
    build_arg_parser,
    main,
    parse_compare_presets,
    record_human_review,
    review_human_reviews,
    review_research_journal,
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


def _journal_row(
    *,
    ticker: str,
    generated_at: str,
    run_dir: Path,
    research_packet_name: str = "research_packet",
    model: str = "llama3.1:latest",
    data_mode: str = "offline_demo",
    offline_demo_data: bool = True,
    action_by_preset: dict[str, str] | None = None,
    confidence_by_preset: dict[str, str] | None = None,
    consensus_by_preset: dict[str, str] | None = None,
    comparison_notes: list[str] | None = None,
    bull_case: str = "Improving momentum; valuation support",
    bear_case: str = "Execution risk; weak demand",
    key_disagreement_points: list[str] | None = None,
    what_to_check_next_manually: list[str] | None = None,
    notable_risks_or_reasons_not_to_act: list[str] | None = None,
) -> dict[str, str]:
    if action_by_preset is None:
        action_by_preset = {"core": "buy", "all": "hold"}
    if confidence_by_preset is None:
        confidence_by_preset = {"core": "67", "all": "55"}
    if consensus_by_preset is None:
        consensus_by_preset = {"core": "bullish", "all": "mixed"}
    if comparison_notes is None:
        comparison_notes = ["Core was constructive while all stayed mixed."]
    if key_disagreement_points is None:
        key_disagreement_points = ["Timing disagreement", "Valuation disagreement"]
    if what_to_check_next_manually is None:
        what_to_check_next_manually = ["Check earnings date", "Verify volume trend"]
    if notable_risks_or_reasons_not_to_act is None:
        notable_risks_or_reasons_not_to_act = ["Macro volatility", "Thin conviction"]
    return {
        "generated_at": generated_at,
        "ticker": ticker,
        "model": model,
        "data_mode": data_mode,
        "offline_demo_data": str(offline_demo_data),
        "run_dir": run_dir.as_posix(),
        "presets_analyzed": json.dumps(sorted(action_by_preset.keys())),
        "action_by_preset": json.dumps(action_by_preset),
        "confidence_by_preset": json.dumps(confidence_by_preset),
        "consensus_by_preset": json.dumps(consensus_by_preset),
        "comparison_notes": json.dumps(comparison_notes),
        "bull_case": bull_case,
        "bear_case": bear_case,
        "key_disagreement_points": json.dumps(key_disagreement_points),
        "data_limitations": json.dumps(["Offline fixture data only"]),
        "what_to_check_next_manually": json.dumps(what_to_check_next_manually),
        "notable_risks_or_reasons_not_to_act": json.dumps(notable_risks_or_reasons_not_to_act),
        "research_packet_md_path": (run_dir / f"{research_packet_name}.md").as_posix(),
        "research_packet_json_path": (run_dir / f"{research_packet_name}.json").as_posix(),
    }


def _write_journal_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESEARCH_JOURNAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _repo_review_output_paths() -> list[Path]:
    return [
        Path("outputs") / "research_journal_review.md",
        Path("outputs") / "research_journal_review_BB.md",
        Path("outputs") / "research_journal_review.json",
        Path("outputs") / "research_journal_review_BB.json",
    ]


def _snapshot_repo_review_outputs() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in _repo_review_output_paths()}


def _assert_repo_review_outputs_unchanged(snapshot: dict[Path, bytes | None]) -> None:
    for path, original_bytes in snapshot.items():
        if original_bytes is None:
            assert not path.exists()
        else:
            assert path.exists()
            assert path.read_bytes() == original_bytes


def _repo_journal_output_paths() -> list[Path]:
    return [Path("outputs") / "research_journal.csv"]


def _snapshot_repo_journal_outputs() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in _repo_journal_output_paths()}


def _assert_repo_journal_outputs_unchanged(snapshot: dict[Path, bytes | None]) -> None:
    for path, original_bytes in snapshot.items():
        if original_bytes is None:
            assert not path.exists()
        else:
            assert path.exists()
            assert path.read_bytes() == original_bytes


def _repo_watchlist_output_paths() -> list[Path]:
    return [
        Path("outputs") / "research_watchlist.md",
        Path("outputs") / "research_watchlist.json",
    ]


def _snapshot_repo_watchlist_outputs() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in _repo_watchlist_output_paths()}


def _assert_repo_watchlist_outputs_unchanged(snapshot: dict[Path, bytes | None]) -> None:
    for path, original_bytes in snapshot.items():
        if original_bytes is None:
            assert not path.exists()
        else:
            assert path.exists()
            assert path.read_bytes() == original_bytes


def _repo_validation_output_paths() -> list[Path]:
    return [
        Path("outputs") / "validation_checklist_BB.md",
        Path("outputs") / "validation_checklist_BB.json",
    ]


def _snapshot_repo_validation_outputs() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in _repo_validation_output_paths()}


def _assert_repo_validation_outputs_unchanged(snapshot: dict[Path, bytes | None]) -> None:
    for path, original_bytes in snapshot.items():
        if original_bytes is None:
            assert not path.exists()
        else:
            assert path.exists()
            assert path.read_bytes() == original_bytes


def _repo_human_review_log_paths() -> list[Path]:
    return [Path("outputs") / "human_review_log.csv"]


def _snapshot_repo_human_review_logs() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in _repo_human_review_log_paths()}


def _assert_repo_human_review_logs_unchanged(snapshot: dict[Path, bytes | None]) -> None:
    for path, original_bytes in snapshot.items():
        if original_bytes is None:
            assert not path.exists()
        else:
            assert path.exists()
            assert path.read_bytes() == original_bytes


def _repo_human_review_summary_paths() -> list[Path]:
    return [
        Path("outputs") / "human_review_summary.md",
        Path("outputs") / "human_review_summary.json",
    ]


def _snapshot_repo_human_review_summary_outputs() -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in _repo_human_review_summary_paths()}


def _assert_repo_human_review_summary_outputs_unchanged(snapshot: dict[Path, bytes | None]) -> None:
    for path, original_bytes in snapshot.items():
        if original_bytes is None:
            assert not path.exists()
        else:
            assert path.exists()
            assert path.read_bytes() == original_bytes


def _write_watchlist_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def test_build_arg_parser_accepts_full_research_workflow_flag() -> None:
    args = build_arg_parser().parse_args(["--full-research-workflow", "--tickers", "BB,GME"])

    assert args.full_research_workflow is True
    assert args.tickers == "BB,GME"


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


def test_build_arg_parser_accepts_review_research_journal_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--review-research-journal",
            "--ticker",
            "BB",
            "--research-journal-path",
            "outputs/research_journal.csv",
            "--journal-review-output",
            "outputs/research_journal_review_BB.md",
        ]
    )

    assert args.review_research_journal is True
    assert args.ticker == "BB"
    assert args.journal_review_output == "outputs/research_journal_review_BB.md"


def test_build_arg_parser_accepts_research_watchlist_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--research-watchlist",
            "--research-journal-path",
            "outputs/research_journal.csv",
            "--watchlist-output",
            "outputs/research_watchlist.md",
        ]
    )

    assert args.research_watchlist is True
    assert args.research_journal_path == "outputs/research_journal.csv"
    assert args.watchlist_output == "outputs/research_watchlist.md"


def test_build_arg_parser_accepts_validation_checklist_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--validation-checklist",
            "--ticker",
            "BB",
            "--research-journal-path",
            "outputs/research_journal.csv",
            "--watchlist-path",
            "outputs/research_watchlist.json",
            "--validation-output",
            "outputs/validation_checklist_BB.md",
        ]
    )

    assert args.validation_checklist is True
    assert args.ticker == "BB"
    assert args.watchlist_path == "outputs/research_watchlist.json"
    assert args.validation_output == "outputs/validation_checklist_BB.md"


def test_build_arg_parser_accepts_record_human_review_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--record-human-review",
            "--ticker",
            "BB",
            "--human-status",
            "Watchlist",
            "--review-notes",
            "Needs current data.",
            "--validation-checklist-path",
            "outputs/validation_checklist_BB.json",
            "--human-review-log-path",
            "outputs/human_review_log.csv",
        ]
    )

    assert args.record_human_review is True
    assert args.ticker == "BB"
    assert args.human_status == "Watchlist"
    assert args.validation_checklist_path == "outputs/validation_checklist_BB.json"
    assert args.human_review_log_path == "outputs/human_review_log.csv"


def test_build_arg_parser_accepts_review_human_reviews_flags() -> None:
    args = build_arg_parser().parse_args(
        [
            "--review-human-reviews",
            "--ticker",
            "BB",
            "--human-review-log-path",
            "outputs/human_review_log.csv",
            "--human-review-summary-output",
            "outputs/human_review_summary.md",
        ]
    )

    assert args.review_human_reviews is True
    assert args.ticker == "BB"
    assert args.human_review_log_path == "outputs/human_review_log.csv"
    assert args.human_review_summary_output == "outputs/human_review_summary.md"


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


def test_review_research_journal_one_ticker(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "research_journal_review_BB.md"
    snapshot = _snapshot_repo_review_outputs()
    run_dir = tmp_path / "runs" / "bb_latest"
    _write_journal_csv(
        journal_path,
        [
            _journal_row(
                ticker="BB",
                generated_at="2026-06-20T09:00:00",
                run_dir=tmp_path / "runs" / "bb_older",
                bull_case="Improving momentum; product catalysts",
                bear_case="Funding risk; execution risk",
            ),
            _journal_row(
                ticker="BB",
                generated_at="2026-06-21T09:00:00",
                run_dir=run_dir,
                comparison_notes=["Core buy; all mixed"],
                what_to_check_next_manually=["Check earnings date", "Review short interest"],
                notable_risks_or_reasons_not_to_act=["Macro volatility", "Financing uncertainty"],
            ),
        ],
    )

    artifacts = review_research_journal(journal_path=journal_path, ticker="BB", output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.entries_reviewed == 2
    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert DECISION_SUMMARY_DISCLAIMER in markdown
    assert f"- Source journal path: `{journal_path.as_posix()}`" in markdown
    assert "- Scope: `BB`" in markdown
    assert f"- Latest run directory: `{run_dir.as_posix()}`" in markdown
    assert "## BB" in markdown
    assert "| Generated At | Model | Data Mode | Action By Preset | Consensus By Preset | Comparison Notes | Research Packet |" in markdown
    assert "Core buy; all mixed" in markdown
    assert "Recurring bull-case themes:" in markdown
    assert payload["scope"]["ticker"] == "BB"
    assert payload["entries_reviewed"] == 2
    assert payload["tickers"][0]["ticker"] == "BB"
    _assert_repo_review_outputs_unchanged(snapshot)


def test_review_research_journal_all_tickers_grouped(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "research_journal_review.md"
    snapshot = _snapshot_repo_review_outputs()
    _write_journal_csv(
        journal_path,
        [
            _journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb"),
            _journal_row(
                ticker="GME",
                generated_at="2026-06-21T10:30:00",
                run_dir=tmp_path / "runs" / "gme",
                action_by_preset={"core": "hold"},
                confidence_by_preset={"core": "51"},
                consensus_by_preset={"core": "mixed"},
                bull_case="Retail interest; squeeze potential",
                bear_case="Volatility risk; weak fundamentals",
            ),
        ],
    )

    artifacts = review_research_journal(journal_path=journal_path, output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert "- Scope: `All tickers`" in markdown
    assert "## BB" in markdown
    assert "## GME" in markdown
    assert [ticker["ticker"] for ticker in payload["tickers"]] == ["BB", "GME"]
    _assert_repo_review_outputs_unchanged(snapshot)


def test_review_research_journal_custom_journal_path(tmp_path: Path) -> None:
    journal_path = tmp_path / "nested" / "custom_journal.csv"
    output_path = tmp_path / "reports" / "custom_journal_review_BB.md"
    snapshot = _snapshot_repo_review_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    artifacts = review_research_journal(journal_path=journal_path, ticker="BB", output_path=output_path)

    assert artifacts.journal_path == journal_path
    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert artifacts.markdown_path.exists()
    assert artifacts.json_path.exists()
    _assert_repo_review_outputs_unchanged(snapshot)


def test_review_research_journal_custom_output_path(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "bb_review.md"
    snapshot = _snapshot_repo_review_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    artifacts = review_research_journal(journal_path=journal_path, ticker="BB", output_path=output_path)

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert artifacts.markdown_path.exists()
    assert artifacts.json_path.exists()
    _assert_repo_review_outputs_unchanged(snapshot)


def test_review_research_journal_malformed_json_adds_warning_without_crashing(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "malformed_review_BB.md"
    snapshot = _snapshot_repo_review_outputs()
    row = _journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")
    row["action_by_preset"] = '{"core":"buy"'
    _write_journal_csv(journal_path, [row])

    artifacts = review_research_journal(journal_path=journal_path, ticker="BB", output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.warnings
    assert "Malformed JSON in `action_by_preset`" in markdown
    assert payload["tickers"][0]["entries"][0]["action_by_preset"]["_malformed"] is True
    assert payload["tickers"][0]["entries"][0]["action_by_preset"]["_raw"] == '{"core":"buy"'
    _assert_repo_review_outputs_unchanged(snapshot)


def test_review_mode_does_not_run_analyst_or_data_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "cli_review_BB.md"
    snapshot = _snapshot_repo_review_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    monkeypatch.setattr(
        "src.basket_runner.run_basket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_basket should not run in review mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preset comparison should not run in review mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.resolve_analysts_for_preset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysts should not resolve in review mode")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--review-research-journal",
            "--ticker",
            "BB",
            "--research-journal-path",
            str(journal_path),
            "--journal-review-output",
            str(output_path),
        ],
    )

    assert main() == 0
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    _assert_repo_review_outputs_unchanged(snapshot)


def test_review_research_journal_missing_file_gives_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(SystemExit, match="Research journal file not found:"):
        review_research_journal(journal_path=missing_path, ticker="BB")


def test_research_watchlist_report_generation_from_sample_journal(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "research_watchlist.md"
    snapshot = _snapshot_repo_watchlist_outputs()
    _write_journal_csv(
        journal_path,
        [
            _journal_row(
                ticker="BB",
                generated_at="2026-06-20T09:00:00",
                run_dir=tmp_path / "runs" / "bb1",
                action_by_preset={"technical-only": "buy", "core": "buy", "all": "hold"},
                consensus_by_preset={"technical-only": "bullish", "core": "bullish", "all": "mixed"},
                comparison_notes=[],
                bull_case="Momentum improving; catalyst setup",
                bear_case="Execution risk; demand uncertainty",
            ),
            _journal_row(
                ticker="BB",
                generated_at="2026-06-21T09:00:00",
                run_dir=tmp_path / "runs" / "bb2",
                action_by_preset={"technical-only": "buy", "core": "buy", "all": "hold"},
                consensus_by_preset={"technical-only": "bullish", "core": "bullish", "all": "mixed"},
                comparison_notes=[],
                bull_case="Trend intact; setup improving",
                bear_case="Execution risk remains",
            ),
            _journal_row(
                ticker="GME",
                generated_at="2026-06-21T10:00:00",
                run_dir=tmp_path / "runs" / "gme1",
                action_by_preset={"technical-only": "buy", "core": "hold", "all": "sell"},
                consensus_by_preset={"technical-only": "bullish", "core": "mixed", "all": "bearish"},
                comparison_notes=["Preset conflict requires manual review."],
            ),
            _journal_row(
                ticker="NVDA",
                generated_at="2026-06-21T11:00:00",
                run_dir=tmp_path / "runs" / "nvda1",
                action_by_preset={"core": "sell", "all": "sell", "no-news": "short"},
                consensus_by_preset={"core": "bearish", "all": "bearish", "no-news": "bearish"},
                comparison_notes=[],
                bull_case="AI demand still exists",
                bear_case="Crowded positioning; valuation risk",
            ),
            _journal_row(
                ticker="AAPL",
                generated_at="2026-06-21T12:00:00",
                run_dir=tmp_path / "runs" / "aapl1",
                action_by_preset={"core": "buy"},
                consensus_by_preset={"core": "bullish"},
                comparison_notes=[],
            ),
        ],
    )

    artifacts = build_research_watchlist(journal_path=journal_path, output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert DECISION_SUMMARY_DISCLAIMER in markdown
    assert f"- Source journal path: `{journal_path.as_posix()}`" in markdown
    assert "## Strong Follow-Up Candidates" in markdown
    assert "## Mixed / Disagreement Candidates" in markdown
    assert "## Bearish / Avoid For Now Candidates" in markdown
    assert "## Insufficient History / Needs More Runs" in markdown
    assert "Scoring heuristic:" in markdown
    assert "| Ticker | Latest Generated At | Score | Category | Latest Actions | Latest Consensus | Disagreement Flags | Latest Bull Case | Latest Bear Case | Latest Packet Path |" in markdown
    _assert_repo_watchlist_outputs_unchanged(snapshot)


def test_research_watchlist_json_companion_generation(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "research_watchlist.md"
    snapshot = _snapshot_repo_watchlist_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    artifacts = build_research_watchlist(journal_path=journal_path, output_path=output_path)
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.json_path.exists()
    assert payload["source_journal_path"] == journal_path.as_posix()
    assert payload["rows_reviewed"] == 1
    assert payload["ticker_count"] == 1
    assert payload["scoring_rules"]
    assert payload["per_ticker"][0]["ticker"] == "BB"
    assert "categories" in payload
    _assert_repo_watchlist_outputs_unchanged(snapshot)


def test_research_watchlist_categorization_sections(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "watchlist.md"
    snapshot = _snapshot_repo_watchlist_outputs()
    _write_journal_csv(
        journal_path,
        [
            _journal_row(
                ticker="BB",
                generated_at="2026-06-20T09:00:00",
                run_dir=tmp_path / "runs" / "bb1",
                data_mode="live",
                offline_demo_data=False,
                action_by_preset={"technical-only": "buy", "core": "buy", "all": "buy"},
                consensus_by_preset={"technical-only": "bullish", "core": "bullish", "all": "bullish"},
                comparison_notes=[],
            ),
            _journal_row(
                ticker="BB",
                generated_at="2026-06-21T09:00:00",
                run_dir=tmp_path / "runs" / "bb2",
                data_mode="live",
                offline_demo_data=False,
                action_by_preset={"technical-only": "buy", "core": "buy", "all": "buy"},
                consensus_by_preset={"technical-only": "bullish", "core": "bullish", "all": "bullish"},
                comparison_notes=[],
            ),
            _journal_row(
                ticker="GME",
                generated_at="2026-06-21T10:00:00",
                run_dir=tmp_path / "runs" / "gme1",
                data_mode="live",
                offline_demo_data=False,
                action_by_preset={"technical-only": "buy", "core": "hold", "all": "hold"},
                consensus_by_preset={"technical-only": "bullish", "core": "mixed", "all": "mixed"},
                comparison_notes=["Disagreement"],
            ),
            _journal_row(
                ticker="GME",
                generated_at="2026-06-22T10:00:00",
                run_dir=tmp_path / "runs" / "gme2",
                data_mode="live",
                offline_demo_data=False,
                action_by_preset={"technical-only": "buy", "core": "hold", "all": "hold"},
                consensus_by_preset={"technical-only": "bullish", "core": "mixed", "all": "mixed"},
                comparison_notes=["Disagreement"],
            ),
            _journal_row(
                ticker="NVDA",
                generated_at="2026-06-21T11:00:00",
                run_dir=tmp_path / "runs" / "nvda1",
                data_mode="live",
                offline_demo_data=False,
                action_by_preset={"core": "sell", "all": "sell", "no-news": "short"},
                consensus_by_preset={"core": "bearish", "all": "bearish", "no-news": "bearish"},
                comparison_notes=[],
            ),
            _journal_row(
                ticker="NVDA",
                generated_at="2026-06-22T11:00:00",
                run_dir=tmp_path / "runs" / "nvda2",
                data_mode="live",
                offline_demo_data=False,
                action_by_preset={"core": "sell", "all": "sell", "no-news": "short"},
                consensus_by_preset={"core": "bearish", "all": "bearish", "no-news": "bearish"},
                comparison_notes=[],
            ),
            _journal_row(
                ticker="AAPL",
                generated_at="2026-06-21T12:00:00",
                run_dir=tmp_path / "runs" / "aapl1",
                data_mode="live",
                offline_demo_data=False,
                action_by_preset={"core": "buy"},
                consensus_by_preset={"core": "bullish"},
                comparison_notes=[],
            ),
        ],
    )

    artifacts = build_research_watchlist(journal_path=journal_path, output_path=output_path)
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert "BB" in payload["categories"]["Strong Follow-Up Candidates"]
    assert "GME" in payload["categories"]["Mixed / Disagreement Candidates"]
    assert "NVDA" in payload["categories"]["Bearish / Avoid For Now Candidates"]
    assert "AAPL" in payload["categories"]["Insufficient History / Needs More Runs"]
    _assert_repo_watchlist_outputs_unchanged(snapshot)


def test_research_watchlist_custom_journal_path(tmp_path: Path) -> None:
    journal_path = tmp_path / "nested" / "custom_journal.csv"
    output_path = tmp_path / "reports" / "watchlist.md"
    snapshot = _snapshot_repo_watchlist_outputs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    artifacts = build_research_watchlist(journal_path=journal_path, output_path=output_path)

    assert artifacts.journal_path == journal_path
    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    _assert_repo_watchlist_outputs_unchanged(snapshot)


def test_research_watchlist_custom_output_path(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "custom" / "my_watchlist.md"
    snapshot = _snapshot_repo_watchlist_outputs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    artifacts = build_research_watchlist(journal_path=journal_path, output_path=output_path)

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    _assert_repo_watchlist_outputs_unchanged(snapshot)


def test_research_watchlist_malformed_json_warning(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "watchlist.md"
    snapshot = _snapshot_repo_watchlist_outputs()
    row = _journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")
    row["consensus_by_preset"] = '{"core":"bullish"'
    _write_journal_csv(journal_path, [row])

    artifacts = build_research_watchlist(journal_path=journal_path, output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.warnings
    assert "Malformed JSON in `consensus_by_preset`" in markdown
    assert payload["warnings"]
    _assert_repo_watchlist_outputs_unchanged(snapshot)


def test_research_watchlist_mode_does_not_run_analyst_or_data_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "watchlist.md"
    snapshot = _snapshot_repo_watchlist_outputs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-20T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    monkeypatch.setattr(
        "src.basket_runner.run_basket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_basket should not run in watchlist mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preset comparison should not run in watchlist mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.resolve_analysts_for_preset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysts should not resolve in watchlist mode")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--research-watchlist",
            "--research-journal-path",
            str(journal_path),
            "--watchlist-output",
            str(output_path),
        ],
    )

    assert main() == 0
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    _assert_repo_watchlist_outputs_unchanged(snapshot)


def test_research_watchlist_missing_journal_file_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(SystemExit, match="Research journal file not found:"):
        build_research_watchlist(journal_path=missing_path, output_path=tmp_path / "reports" / "watchlist.md")


def test_validation_checklist_generation_for_ticker_from_sample_journal(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "validation_checklist_BB.md"
    snapshot = _snapshot_repo_validation_outputs()
    _write_journal_csv(
        journal_path,
        [
            _journal_row(
                ticker="BB",
                generated_at="2026-06-20T09:00:00",
                run_dir=tmp_path / "runs" / "bb1",
                action_by_preset={"technical-only": "buy", "core": "buy", "all": "hold"},
                consensus_by_preset={"technical-only": "bullish", "core": "mixed", "all": "mixed"},
                comparison_notes=["Preset conflict requires review."],
            ),
            _journal_row(
                ticker="BB",
                generated_at="2026-06-21T09:00:00",
                run_dir=tmp_path / "runs" / "bb2",
                action_by_preset={"technical-only": "buy", "core": "buy", "all": "hold"},
                consensus_by_preset={"technical-only": "bullish", "core": "mixed", "all": "mixed"},
                comparison_notes=["Preset conflict requires review."],
            ),
        ],
    )

    artifacts = build_validation_checklist(ticker="BB", journal_path=journal_path, output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert DECISION_SUMMARY_DISCLAIMER in markdown
    assert "- Ticker: `BB`" in markdown
    assert f"- Source journal path: `{journal_path.as_posix()}`" in markdown
    assert "## Current Price / Chart Validation" in markdown
    assert "- [ ] Confirm current price and date of quote." in markdown
    assert "## Final Status" in markdown
    assert payload["ticker"] == "BB"
    assert payload["latest_actions"]["technical-only"] == "buy"
    assert payload["checklist_sections"]
    _assert_repo_validation_outputs_unchanged(snapshot)


def test_validation_checklist_uses_watchlist_json_when_available(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    watchlist_path = tmp_path / "research_watchlist.json"
    output_path = tmp_path / "reports" / "validation_checklist_BB.md"
    snapshot = _snapshot_repo_validation_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )
    _write_watchlist_json(
        watchlist_path,
        {
            "generated_at": "2026-06-24T10:00:00",
            "source_journal_path": journal_path.as_posix(),
            "rows_reviewed": 1,
            "ticker_count": 1,
            "scoring_rules": [],
            "per_ticker": [
                {
                    "ticker": "BB",
                    "score": 3,
                    "category": "Mixed / Disagreement Candidates",
                    "disagreement_flags": ["comparison notes present", "needs validation"],
                }
            ],
            "categories": {},
            "warnings": [],
        },
    )

    artifacts = build_validation_checklist(
        ticker="BB",
        journal_path=journal_path,
        watchlist_path=watchlist_path,
        output_path=output_path,
    )
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert f"- Source watchlist path: `{watchlist_path.as_posix()}`" in markdown
    assert "- Latest watchlist category: `Mixed / Disagreement Candidates`" in markdown
    assert payload["latest_watchlist_score"] == 3
    assert "needs validation" in payload["disagreement_flags"]
    _assert_repo_validation_outputs_unchanged(snapshot)


def test_validation_checklist_missing_watchlist_file_warning(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    missing_watchlist = tmp_path / "missing_watchlist.json"
    output_path = tmp_path / "reports" / "validation_checklist_BB.md"
    snapshot = _snapshot_repo_validation_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    artifacts = build_validation_checklist(
        ticker="BB",
        journal_path=journal_path,
        watchlist_path=missing_watchlist,
        output_path=output_path,
    )
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert artifacts.warnings
    assert "generating checklist from journal only" in markdown.lower()
    _assert_repo_validation_outputs_unchanged(snapshot)


def test_validation_checklist_missing_ticker_error(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    with pytest.raises(SystemExit, match="Validation checklist mode requires --ticker."):
        build_validation_checklist(ticker=None, journal_path=journal_path, output_path=tmp_path / "reports" / "validation.md")


def test_validation_checklist_ticker_not_found_error(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="GME", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "gme")],
    )

    with pytest.raises(SystemExit, match="Ticker `BB` was not found in research journal:"):
        build_validation_checklist(ticker="BB", journal_path=journal_path, output_path=tmp_path / "reports" / "validation.md")


def test_validation_checklist_missing_journal_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(SystemExit, match="Research journal file not found:"):
        build_validation_checklist(ticker="BB", journal_path=missing_path, output_path=tmp_path / "reports" / "validation.md")


def test_validation_checklist_malformed_json_warning(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    watchlist_path = tmp_path / "bad_watchlist.json"
    output_path = tmp_path / "reports" / "validation_checklist_BB.md"
    snapshot = _snapshot_repo_validation_outputs()
    row = _journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")
    row["action_by_preset"] = '{"core":"buy"'
    _write_journal_csv(journal_path, [row])
    watchlist_path.write_text('{"per_ticker":[', encoding="utf-8")

    artifacts = build_validation_checklist(
        ticker="BB",
        journal_path=journal_path,
        watchlist_path=watchlist_path,
        output_path=output_path,
    )
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert any("Malformed JSON in `action_by_preset`" in warning for warning in artifacts.warnings)
    assert any("Malformed watchlist JSON" in warning for warning in artifacts.warnings)
    assert "Malformed watchlist JSON" in markdown
    assert payload["warnings"]
    _assert_repo_validation_outputs_unchanged(snapshot)


def test_validation_checklist_custom_output_path(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "custom" / "bb_checklist.md"
    snapshot = _snapshot_repo_validation_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    artifacts = build_validation_checklist(ticker="BB", journal_path=journal_path, output_path=output_path)

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    _assert_repo_validation_outputs_unchanged(snapshot)


def test_validation_checklist_mode_does_not_run_analyst_or_data_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal_path = tmp_path / "journal.csv"
    output_path = tmp_path / "reports" / "validation_checklist_BB.md"
    snapshot = _snapshot_repo_validation_outputs()
    _write_journal_csv(
        journal_path,
        [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")],
    )

    monkeypatch.setattr(
        "src.basket_runner.run_basket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_basket should not run in validation mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preset comparison should not run in validation mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.resolve_analysts_for_preset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysts should not resolve in validation mode")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--validation-checklist",
            "--ticker",
            "BB",
            "--research-journal-path",
            str(journal_path),
            "--validation-output",
            str(output_path),
        ],
    )

    assert main() == 0
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    _assert_repo_validation_outputs_unchanged(snapshot)


def test_record_human_review_creates_log_with_headers(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    checklist_path = tmp_path / "validation_checklist_BB.json"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])
    checklist_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-25T09:00:00",
                "ticker": "BB",
                "source_journal_path": journal_path.as_posix(),
                "source_watchlist_path": None,
                "latest_research_packet_path": (tmp_path / "runs" / "bb" / "research_packet.md").as_posix(),
                "latest_watchlist_category": "Watchlist",
                "latest_watchlist_score": 1,
                "latest_actions": {"core": "buy"},
                "latest_consensus": {"core": "mixed"},
                "disagreement_flags": ["needs current-data validation"],
                "checklist_sections": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    artifacts = record_human_review(
        ticker="BB",
        human_status="Watchlist",
        review_notes="Needs live-data check before deeper work.",
        validation_checklist_path=checklist_path,
        journal_path=journal_path,
        human_review_log_path=log_path,
    )
    rows = list(csv.DictReader(log_path.open(encoding="utf-8", newline="")))
    header_line = log_path.read_text(encoding="utf-8").splitlines()[0]

    assert artifacts.log_path == log_path
    assert artifacts.rows_written == 1
    assert header_line.split(",") == HUMAN_REVIEW_LOG_FIELDNAMES
    assert len(rows) == 1
    assert rows[0]["ticker"] == "BB"
    assert rows[0]["human_status"] == "Watchlist"
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_record_human_review_second_append_does_not_duplicate_headers(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(
        journal_path,
        [
            _journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb"),
            _journal_row(ticker="GME", generated_at="2026-06-21T10:00:00", run_dir=tmp_path / "runs" / "gme"),
        ],
    )

    record_human_review(ticker="BB", human_status="Watchlist", journal_path=journal_path, human_review_log_path=log_path)
    record_human_review(ticker="GME", human_status="Reject", journal_path=journal_path, human_review_log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(log_path.open(encoding="utf-8", newline="")))

    assert sum(1 for line in lines if line.startswith("reviewed_at,")) == 1
    assert [row["ticker"] for row in rows] == ["BB", "GME"]
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_record_human_review_custom_log_path_works(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    log_path = tmp_path / "custom" / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    artifacts = record_human_review(ticker="BB", human_status="Deep Research", journal_path=journal_path, human_review_log_path=log_path)

    assert artifacts.log_path == log_path
    assert log_path.exists()
    _assert_repo_human_review_logs_unchanged(snapshot)


@pytest.mark.parametrize("status", ["Watchlist", "Reject", "Deep Research", "Paper Trade Candidate", "Trade Candidate"])
def test_record_human_review_allowed_statuses_work(tmp_path: Path, status: str) -> None:
    journal_path = tmp_path / "journal.csv"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    artifacts = record_human_review(ticker="BB", human_status=status, journal_path=journal_path, human_review_log_path=log_path)
    rows = list(csv.DictReader(log_path.open(encoding="utf-8", newline="")))

    assert artifacts.human_status == status
    assert rows[0]["human_status"] == status
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_record_human_review_invalid_status_fails_clearly(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    with pytest.raises(SystemExit, match="Invalid --human-status. Allowed values:"):
        record_human_review(ticker="BB", human_status="Maybe", journal_path=journal_path, human_review_log_path=tmp_path / "human_review_log.csv")


def test_record_human_review_missing_validation_checklist_warns_but_still_writes(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    missing_checklist = tmp_path / "missing_validation_checklist.json"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    artifacts = record_human_review(
        ticker="BB",
        human_status="Watchlist",
        validation_checklist_path=missing_checklist,
        journal_path=journal_path,
        human_review_log_path=log_path,
    )
    rows = list(csv.DictReader(log_path.open(encoding="utf-8", newline="")))

    assert artifacts.warnings
    assert any("Validation checklist JSON not found" in warning for warning in artifacts.warnings)
    assert len(rows) == 1
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_record_human_review_missing_watchlist_path_warns_but_still_writes(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    missing_watchlist = tmp_path / "missing_watchlist.json"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    artifacts = record_human_review(
        ticker="BB",
        human_status="Watchlist",
        journal_path=journal_path,
        watchlist_path=missing_watchlist,
        human_review_log_path=log_path,
    )
    rows = list(csv.DictReader(log_path.open(encoding="utf-8", newline="")))

    assert any("Research watchlist JSON not found" in warning for warning in artifacts.warnings)
    assert len(rows) == 1
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_record_human_review_context_is_pulled_from_validation_checklist_json_when_available(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    checklist_path = tmp_path / "validation_checklist_BB.json"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])
    checklist_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-25T09:00:00",
                "ticker": "BB",
                "source_journal_path": journal_path.as_posix(),
                "source_watchlist_path": "outputs/research_watchlist.json",
                "latest_research_packet_path": "outputs/ras_ollama_basket_runs/demo/research_packet.md",
                "latest_watchlist_category": "Mixed / Disagreement Candidates",
                "latest_watchlist_score": 2,
                "latest_actions": {"technical-only": "buy", "core": "buy"},
                "latest_consensus": {"technical-only": "bullish", "core": "mixed"},
                "disagreement_flags": ["comparison notes present"],
                "checklist_sections": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    record_human_review(
        ticker="BB",
        human_status="Deep Research",
        validation_checklist_path=checklist_path,
        journal_path=journal_path,
        human_review_log_path=log_path,
    )
    row = next(csv.DictReader(log_path.open(encoding="utf-8", newline="")))

    assert row["latest_research_packet_path"] == "outputs/ras_ollama_basket_runs/demo/research_packet.md"
    assert row["latest_watchlist_category"] == "Mixed / Disagreement Candidates"
    assert row["latest_watchlist_score"] == "2"
    assert json.loads(row["latest_actions"])["technical-only"] == "buy"
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_record_human_review_json_list_fields_are_valid_json_strings(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.csv"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])

    record_human_review(ticker="BB", human_status="Watchlist", journal_path=journal_path, human_review_log_path=log_path)
    row = next(csv.DictReader(log_path.open(encoding="utf-8", newline="")))

    assert isinstance(json.loads(row["latest_actions"]), dict)
    assert isinstance(json.loads(row["latest_consensus"]), dict)
    assert isinstance(json.loads(row["disagreement_flags"]), list)
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_record_human_review_mode_does_not_run_analyst_or_data_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal_path = tmp_path / "journal.csv"
    checklist_path = tmp_path / "validation_checklist_BB.json"
    log_path = tmp_path / "human_review_log.csv"
    snapshot = _snapshot_repo_human_review_logs()
    _write_journal_csv(journal_path, [_journal_row(ticker="BB", generated_at="2026-06-21T09:00:00", run_dir=tmp_path / "runs" / "bb")])
    checklist_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-25T09:00:00",
                "ticker": "BB",
                "source_journal_path": journal_path.as_posix(),
                "source_watchlist_path": None,
                "latest_research_packet_path": "",
                "latest_watchlist_category": None,
                "latest_watchlist_score": None,
                "latest_actions": {"core": "buy"},
                "latest_consensus": {"core": "mixed"},
                "disagreement_flags": [],
                "checklist_sections": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.basket_runner.run_basket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_basket should not run in record-human-review mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preset comparison should not run in record-human-review mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.resolve_analysts_for_preset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysts should not resolve in record-human-review mode")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--record-human-review",
            "--ticker",
            "BB",
            "--human-status",
            "Watchlist",
            "--validation-checklist-path",
            str(checklist_path),
            "--research-journal-path",
            str(journal_path),
            "--human-review-log-path",
            str(log_path),
        ],
    )

    assert main() == 0
    assert log_path.exists()
    _assert_repo_human_review_logs_unchanged(snapshot)


def test_review_human_reviews_summary_for_all_tickers(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "reports" / "human_review_summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "reviewed_at": "2026-06-25T09:00:00",
                    "ticker": "BB",
                    "human_status": "Watchlist",
                    "review_notes": "Needs more work.",
                    "validation_checklist_path": "outputs/validation_checklist_BB.json",
                    "latest_research_packet_path": "outputs/bb_packet.md",
                    "latest_watchlist_category": "Mixed / Disagreement Candidates",
                    "latest_watchlist_score": "2",
                    "latest_actions": json.dumps({"core": "buy"}),
                    "latest_consensus": json.dumps({"core": "mixed"}),
                    "disagreement_flags": json.dumps(["needs validation"]),
                    "data_mode": "offline_demo",
                    "offline_demo_warning": "Latest journal context reflects offline_demo data; current-data validation is still required.",
                    "source_journal_path": "outputs/research_journal.csv",
                    "source_watchlist_path": "outputs/research_watchlist.json",
                },
                {
                    "reviewed_at": "2026-06-25T10:00:00",
                    "ticker": "GME",
                    "human_status": "Reject",
                    "review_notes": "Current news invalidated setup.",
                    "validation_checklist_path": "outputs/validation_checklist_GME.json",
                    "latest_research_packet_path": "outputs/gme_packet.md",
                    "latest_watchlist_category": "Bearish / Avoid For Now Candidates",
                    "latest_watchlist_score": "-3",
                    "latest_actions": json.dumps({"all": "sell"}),
                    "latest_consensus": json.dumps({"all": "bearish"}),
                    "disagreement_flags": json.dumps([]),
                    "data_mode": "live",
                    "offline_demo_warning": "",
                    "source_journal_path": "outputs/research_journal.csv",
                    "source_watchlist_path": "outputs/research_watchlist.json",
                },
            ]
        )

    artifacts = review_human_reviews(log_path=log_path, output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert DECISION_SUMMARY_DISCLAIMER in markdown
    assert f"- Source human review log path: `{log_path.as_posix()}`" in markdown
    assert "- Scope: `All tickers`" in markdown
    assert "## Latest Review Per Ticker" in markdown
    assert "## Chronological Review History" in markdown
    assert "## Follow-Up Buckets" in markdown
    assert payload["total_review_rows"] == 2
    assert payload["reviewed_ticker_count"] == 2
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_summary_filtered_to_one_ticker(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "reports" / "human_review_summary_BB.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "reviewed_at": "2026-06-25T09:00:00",
                    "ticker": "BB",
                    "human_status": "Watchlist",
                    "review_notes": "Needs more work.",
                    "validation_checklist_path": "outputs/validation_checklist_BB.json",
                    "latest_research_packet_path": "outputs/bb_packet.md",
                    "latest_watchlist_category": "Mixed / Disagreement Candidates",
                    "latest_watchlist_score": "2",
                    "latest_actions": json.dumps({"core": "buy"}),
                    "latest_consensus": json.dumps({"core": "mixed"}),
                    "disagreement_flags": json.dumps(["needs validation"]),
                    "data_mode": "offline_demo",
                    "offline_demo_warning": "",
                    "source_journal_path": "outputs/research_journal.csv",
                    "source_watchlist_path": "outputs/research_watchlist.json",
                },
                {
                    "reviewed_at": "2026-06-25T10:00:00",
                    "ticker": "GME",
                    "human_status": "Reject",
                    "review_notes": "Invalidated.",
                    "validation_checklist_path": "outputs/validation_checklist_GME.json",
                    "latest_research_packet_path": "outputs/gme_packet.md",
                    "latest_watchlist_category": "Bearish / Avoid For Now Candidates",
                    "latest_watchlist_score": "-3",
                    "latest_actions": json.dumps({"all": "sell"}),
                    "latest_consensus": json.dumps({"all": "bearish"}),
                    "disagreement_flags": json.dumps([]),
                    "data_mode": "live",
                    "offline_demo_warning": "",
                    "source_journal_path": "outputs/research_journal.csv",
                    "source_watchlist_path": "outputs/research_watchlist.json",
                },
            ]
        )

    artifacts = review_human_reviews(log_path=log_path, ticker="BB", output_path=output_path)
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.ticker == "BB"
    assert payload["total_review_rows"] == 1
    assert payload["reviewed_ticker_count"] == 1
    assert payload["latest_review_per_ticker"][0]["ticker"] == "BB"
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_custom_log_path(tmp_path: Path) -> None:
    log_path = tmp_path / "custom" / "human_review_log.csv"
    output_path = tmp_path / "reports" / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()

    artifacts = review_human_reviews(log_path=log_path, output_path=output_path)

    assert artifacts.log_path == log_path
    assert artifacts.markdown_path == output_path
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_custom_output_path(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "custom" / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()

    artifacts = review_human_reviews(log_path=log_path, output_path=output_path)

    assert artifacts.markdown_path == output_path
    assert artifacts.json_path == output_path.with_suffix(".json")
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_counts_by_human_status(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()
        for idx, status in enumerate(["Watchlist", "Reject", "Deep Research", "Paper Trade Candidate", "Trade Candidate"], start=1):
            writer.writerow(
                {
                    "reviewed_at": f"2026-06-25T0{idx}:00:00",
                    "ticker": f"T{idx}",
                    "human_status": status,
                    "review_notes": "",
                    "validation_checklist_path": "",
                    "latest_research_packet_path": "",
                    "latest_watchlist_category": "",
                    "latest_watchlist_score": "",
                    "latest_actions": json.dumps({}),
                    "latest_consensus": json.dumps({}),
                    "disagreement_flags": json.dumps([]),
                    "data_mode": "",
                    "offline_demo_warning": "",
                    "source_journal_path": "",
                    "source_watchlist_path": "",
                }
            )

    payload = json.loads(review_human_reviews(log_path=log_path, output_path=output_path).json_path.read_text(encoding="utf-8"))

    assert payload["counts_by_human_status"]["Watchlist"] == 1
    assert payload["counts_by_human_status"]["Reject"] == 1
    assert payload["counts_by_human_status"]["Deep Research"] == 1
    assert payload["counts_by_human_status"]["Paper Trade Candidate"] == 1
    assert payload["counts_by_human_status"]["Trade Candidate"] == 1
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_latest_review_per_ticker_picks_most_recent_row(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "reviewed_at": "2026-06-25T09:00:00",
                    "ticker": "BB",
                    "human_status": "Watchlist",
                    "review_notes": "Older note",
                    "validation_checklist_path": "older.json",
                    "latest_research_packet_path": "",
                    "latest_watchlist_category": "",
                    "latest_watchlist_score": "",
                    "latest_actions": json.dumps({"core": "buy"}),
                    "latest_consensus": json.dumps({"core": "mixed"}),
                    "disagreement_flags": json.dumps([]),
                    "data_mode": "",
                    "offline_demo_warning": "",
                    "source_journal_path": "",
                    "source_watchlist_path": "",
                },
                {
                    "reviewed_at": "2026-06-25T10:00:00",
                    "ticker": "BB",
                    "human_status": "Deep Research",
                    "review_notes": "Newer note",
                    "validation_checklist_path": "newer.json",
                    "latest_research_packet_path": "",
                    "latest_watchlist_category": "",
                    "latest_watchlist_score": "",
                    "latest_actions": json.dumps({"core": "buy"}),
                    "latest_consensus": json.dumps({"core": "bullish"}),
                    "disagreement_flags": json.dumps([]),
                    "data_mode": "",
                    "offline_demo_warning": "",
                    "source_journal_path": "",
                    "source_watchlist_path": "",
                },
            ]
        )

    payload = json.loads(review_human_reviews(log_path=log_path, output_path=output_path).json_path.read_text(encoding="utf-8"))
    latest = payload["latest_review_per_ticker"][0]

    assert latest["latest_human_status"] == "Deep Research"
    assert latest["latest_review_notes"] == "Newer note"
    assert latest["validation_checklist_path"] == "newer.json"
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_follow_up_buckets_group_latest_statuses_correctly(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()
        rows = [
            ("BB", "Watchlist"),
            ("GME", "Reject"),
            ("AAPL", "Deep Research"),
            ("MSFT", "Paper Trade Candidate"),
            ("NVDA", "Trade Candidate"),
        ]
        for idx, (ticker_value, status) in enumerate(rows, start=1):
            writer.writerow(
                {
                    "reviewed_at": f"2026-06-25T0{idx}:00:00",
                    "ticker": ticker_value,
                    "human_status": status,
                    "review_notes": "",
                    "validation_checklist_path": "",
                    "latest_research_packet_path": "",
                    "latest_watchlist_category": "",
                    "latest_watchlist_score": "",
                    "latest_actions": json.dumps({}),
                    "latest_consensus": json.dumps({}),
                    "disagreement_flags": json.dumps([]),
                    "data_mode": "",
                    "offline_demo_warning": "",
                    "source_journal_path": "",
                    "source_watchlist_path": "",
                }
            )

    payload = json.loads(review_human_reviews(log_path=log_path, output_path=output_path).json_path.read_text(encoding="utf-8"))

    assert payload["follow_up_buckets"]["Active Watchlist"] == ["BB"]
    assert payload["follow_up_buckets"]["Rejected"] == ["GME"]
    assert payload["follow_up_buckets"]["Deep Research Queue"] == ["AAPL"]
    assert payload["follow_up_buckets"]["Paper Trade Candidates"] == ["MSFT"]
    assert payload["follow_up_buckets"]["Trade Candidates"] == ["NVDA"]
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_malformed_json_warning(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "reviewed_at": "2026-06-25T09:00:00",
                "ticker": "BB",
                "human_status": "Watchlist",
                "review_notes": "",
                "validation_checklist_path": "",
                "latest_research_packet_path": "",
                "latest_watchlist_category": "",
                "latest_watchlist_score": "",
                "latest_actions": '{"core":"buy"',
                "latest_consensus": json.dumps({"core": "mixed"}),
                "disagreement_flags": json.dumps([]),
                "data_mode": "",
                "offline_demo_warning": "",
                "source_journal_path": "",
                "source_watchlist_path": "",
            }
        )

    artifacts = review_human_reviews(log_path=log_path, output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert artifacts.warnings
    assert "Malformed JSON in `latest_actions`" in markdown
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_missing_log_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(SystemExit, match="Human review log file not found:"):
        review_human_reviews(log_path=missing_path, output_path=tmp_path / "summary.md")


def test_review_human_reviews_empty_log_report(tmp_path: Path) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()

    artifacts = review_human_reviews(log_path=log_path, output_path=output_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.rows_reviewed == 0
    assert artifacts.reviewed_ticker_count == 0
    assert "- Total review rows: `0`" in markdown
    assert payload["total_review_rows"] == 0
    assert payload["reviewed_ticker_count"] == 0
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_review_human_reviews_mode_does_not_run_analyst_data_workflow_or_append_anything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "human_review_log.csv"
    output_path = tmp_path / "summary.md"
    snapshot = _snapshot_repo_human_review_summary_outputs()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "reviewed_at": "2026-06-25T09:00:00",
                "ticker": "BB",
                "human_status": "Watchlist",
                "review_notes": "",
                "validation_checklist_path": "",
                "latest_research_packet_path": "",
                "latest_watchlist_category": "",
                "latest_watchlist_score": "",
                "latest_actions": json.dumps({}),
                "latest_consensus": json.dumps({}),
                "disagreement_flags": json.dumps([]),
                "data_mode": "",
                "offline_demo_warning": "",
                "source_journal_path": "",
                "source_watchlist_path": "",
            }
        )
    original_log_bytes = log_path.read_bytes()

    monkeypatch.setattr(
        "src.basket_runner.run_basket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_basket should not run in review-human-reviews mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preset comparison should not run in review-human-reviews mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.resolve_analysts_for_preset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysts should not resolve in review-human-reviews mode")),
    )
    monkeypatch.setattr(
        "src.basket_runner.record_human_review",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("record_human_review should not run in review-human-reviews mode")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--review-human-reviews",
            "--human-review-log-path",
            str(log_path),
            "--human-review-summary-output",
            str(output_path),
        ],
    )

    assert main() == 0
    assert output_path.exists()
    assert log_path.read_bytes() == original_log_bytes
    _assert_repo_human_review_summary_outputs_unchanged(snapshot)


def test_full_research_workflow_creates_and_returns_expected_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "workflow-run"
    comparison_csv = run_dir / "preset_comparison.csv"
    comparison_md = run_dir / "preset_comparison.md"
    comparison_rows = _comparison_rows_for_tickers(run_dir, ["BB"])

    def fake_run_preset_comparison(config: BasketRunConfig, presets: list[str]) -> SimpleNamespace:
        run_dir.mkdir(parents=True, exist_ok=True)
        comparison_csv.write_text("ticker\nBB\n", encoding="utf-8")
        comparison_md.write_text("# Preset Comparison\n", encoding="utf-8")
        return SimpleNamespace(
            run_dir=run_dir,
            csv_path=comparison_csv,
            markdown_path=comparison_md,
            rows=comparison_rows,
            preset_run_dirs={preset: (run_dir / "preset_runs" / preset).as_posix() for preset in presets},
        )

    monkeypatch.setattr("src.basket_runner.run_preset_comparison", fake_run_preset_comparison)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB",
            "--compare-presets",
            "technical-only,core,no-news,all",
            "--output-dir",
            str(tmp_path),
            "--research-journal-path",
            str(tmp_path / "custom_journal.csv"),
            "--watchlist-output",
            str(tmp_path / "reports" / "watchlist.md"),
            "--validation-output",
            str(tmp_path / "reports" / "validation_checklist_BB.md"),
            "--offline-demo-data",
            "--continue-on-error",
            "--fast-data-mode",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["run_dir"] == run_dir.as_posix()
    assert payload["workflow_status"] == "success"
    assert payload["tickers"] == ["BB"]
    assert payload["requested_presets"] == ["technical-only", "core", "no-news", "all"]
    assert payload["comparison_row_count"] == 4
    assert payload["successful_comparison_row_count"] == 4
    assert payload["failed_comparison_row_count"] == 0
    assert payload["successful_tickers"] == ["BB"]
    assert payload["partial_failure_tickers"] == []
    assert payload["failed_tickers"] == []
    assert payload["missing_tickers"] == []
    assert payload["ticker_results"]["BB"]["status"] == "success"
    assert payload["preset_comparison_csv"] == comparison_csv.as_posix()
    assert payload["preset_comparison_md"] == comparison_md.as_posix()
    assert Path(payload["research_packet_md"]).exists()
    assert Path(payload["research_packet_json"]).exists()
    assert payload["research_journal_csv"] == (tmp_path / "custom_journal.csv").as_posix()
    assert Path(payload["research_journal_csv"]).exists()
    assert payload["research_journal_rows_written"] == 1
    assert payload["research_watchlist_md"] == (tmp_path / "reports" / "watchlist.md").as_posix()
    assert payload["research_watchlist_json"] == (tmp_path / "reports" / "watchlist.json").as_posix()
    assert Path(payload["research_watchlist_md"]).exists()
    assert Path(payload["research_watchlist_json"]).exists()
    assert payload["research_watchlist_rows_reviewed"] == 1
    assert payload["research_watchlist_ticker_count"] == 1
    assert set(payload["validation_checklists"]) == {"BB"}
    assert payload["validation_checklist_count"] == 1
    assert Path(payload["validation_checklists"]["BB"]["markdown"]).exists()
    assert Path(payload["validation_checklists"]["BB"]["json"]).exists()
    assert any("offline_demo data" in warning for warning in payload["warnings"])

    packet_markdown = Path(payload["research_packet_md"]).read_text(encoding="utf-8")
    watchlist_markdown = Path(payload["research_watchlist_md"]).read_text(encoding="utf-8")
    validation_markdown = Path(payload["validation_checklists"]["BB"]["markdown"]).read_text(encoding="utf-8")

    assert DECISION_SUMMARY_DISCLAIMER in packet_markdown
    assert OFFLINE_DEMO_DISCLAIMER in packet_markdown
    assert DECISION_SUMMARY_DISCLAIMER in watchlist_markdown
    assert DECISION_SUMMARY_DISCLAIMER in validation_markdown
    assert "current-data validation is required" in validation_markdown


def test_full_research_workflow_defaults_compare_presets_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}

    def fake_run_preset_comparison(config: BasketRunConfig, presets: list[str]) -> SimpleNamespace:
        observed["presets"] = list(presets)
        observed["continue_on_error"] = config.continue_on_error
        run_dir = tmp_path / "workflow-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            run_dir=run_dir,
            csv_path=run_dir / "preset_comparison.csv",
            markdown_path=run_dir / "preset_comparison.md",
            rows=[],
            preset_run_dirs={},
        )

    monkeypatch.setattr("src.basket_runner.run_preset_comparison", fake_run_preset_comparison)
    monkeypatch.setattr(
        "src.basket_runner.write_research_packet",
        lambda *args, **kwargs: SimpleNamespace(
            markdown_path=tmp_path / "research_packet.md",
            json_path=tmp_path / "research_packet.json",
            payload={},
        ),
    )
    monkeypatch.setattr(
        "src.basket_runner.append_research_journal",
        lambda *args, **kwargs: SimpleNamespace(journal_path=tmp_path / "research_journal.csv", rows_written=0),
    )
    monkeypatch.setattr(
        "src.basket_runner.build_research_watchlist",
        lambda *args, **kwargs: SimpleNamespace(
            journal_path=tmp_path / "research_journal.csv",
            markdown_path=tmp_path / "research_watchlist.md",
            json_path=tmp_path / "research_watchlist.json",
            rows_reviewed=0,
            ticker_count=0,
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        "src.basket_runner.build_validation_checklist",
        lambda *args, **kwargs: SimpleNamespace(
            journal_path=tmp_path / "research_journal.csv",
            watchlist_path=tmp_path / "research_watchlist.json",
            markdown_path=tmp_path / "validation_checklist_BB.md",
            json_path=tmp_path / "validation_checklist_BB.json",
            ticker="BB",
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB",
            "--continue-on-error",
        ],
    )

    assert main() == 0
    json.loads(capsys.readouterr().out)
    assert observed["presets"] == ["technical-only", "core", "no-news", "all"]
    assert observed["continue_on_error"] is True


def test_full_research_workflow_multiple_tickers_create_multiple_validation_checklists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "workflow-run"
    comparison_rows = _comparison_rows_for_tickers(run_dir, ["BB", "GME"])

    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda config, presets: SimpleNamespace(
            run_dir=run_dir,
            csv_path=run_dir / "preset_comparison.csv",
            markdown_path=run_dir / "preset_comparison.md",
            rows=comparison_rows,
            preset_run_dirs={preset: (run_dir / "preset_runs" / preset).as_posix() for preset in presets},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB,GME",
            "--compare-presets",
            "technical-only,core",
            "--output-dir",
            str(tmp_path),
            "--offline-demo-data",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert set(payload["validation_checklists"]) == {"BB", "GME"}
    assert Path(payload["validation_checklists"]["BB"]["markdown"]).exists()
    assert Path(payload["validation_checklists"]["GME"]["markdown"]).exists()


def test_full_research_workflow_uses_output_dir_for_default_secondary_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow_output_dir = tmp_path / "owner_smoke"
    run_dir = workflow_output_dir / "workflow-run"
    comparison_rows = _comparison_rows_for_tickers(run_dir, ["BB"])
    journal_snapshot = _snapshot_repo_journal_outputs()
    watchlist_snapshot = _snapshot_repo_watchlist_outputs()
    validation_snapshot = _snapshot_repo_validation_outputs()

    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda config, presets: SimpleNamespace(
            run_dir=run_dir,
            csv_path=run_dir / "preset_comparison.csv",
            markdown_path=run_dir / "preset_comparison.md",
            rows=comparison_rows,
            preset_run_dirs={preset: (run_dir / "preset_runs" / preset).as_posix() for preset in presets},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB",
            "--output-dir",
            str(workflow_output_dir),
            "--offline-demo-data",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["workflow_status"] == "success"
    assert payload["research_journal_csv"] == (workflow_output_dir / "research_journal.csv").as_posix()
    assert payload["research_watchlist_md"] == (workflow_output_dir / "research_watchlist.md").as_posix()
    assert payload["research_watchlist_json"] == (workflow_output_dir / "research_watchlist.json").as_posix()
    assert payload["validation_checklists"]["BB"]["markdown"] == (
        workflow_output_dir / "validation_checklist_BB.md"
    ).as_posix()
    assert payload["validation_checklists"]["BB"]["json"] == (
        workflow_output_dir / "validation_checklist_BB.json"
    ).as_posix()
    assert payload["run_dir"].startswith(workflow_output_dir.as_posix())
    assert payload["preset_comparison_csv"].startswith(workflow_output_dir.as_posix())
    assert payload["research_packet_md"].startswith(workflow_output_dir.as_posix())
    assert payload["research_journal_csv"].startswith(workflow_output_dir.as_posix())
    assert payload["research_watchlist_md"].startswith(workflow_output_dir.as_posix())
    assert payload["validation_checklists"]["BB"]["markdown"].startswith(workflow_output_dir.as_posix())
    _assert_repo_journal_outputs_unchanged(journal_snapshot)
    _assert_repo_watchlist_outputs_unchanged(watchlist_snapshot)
    _assert_repo_validation_outputs_unchanged(validation_snapshot)


def test_full_research_workflow_reports_partial_failure_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "workflow-run"
    comparison_rows = _comparison_rows_for_tickers(run_dir, ["BB"])
    comparison_rows[1]["action"] = "FAILED"
    comparison_rows[1]["confidence"] = "-"
    comparison_rows[1]["reasoning"] = "Run failed after passing data checks: upstream timeout."
    comparison_rows[1]["run_status"] = "failed"
    comparison_rows[1]["failure_classification"] = "timeout"

    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda config, presets: SimpleNamespace(
            run_dir=run_dir,
            csv_path=run_dir / "preset_comparison.csv",
            markdown_path=run_dir / "preset_comparison.md",
            rows=comparison_rows,
            preset_run_dirs={preset: (run_dir / "preset_runs" / preset).as_posix() for preset in presets},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB",
            "--output-dir",
            str(tmp_path / "owner_smoke"),
            "--offline-demo-data",
            "--continue-on-error",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["workflow_status"] == "partial_success"
    assert payload["successful_comparison_row_count"] == 3
    assert payload["failed_comparison_row_count"] == 1
    assert payload["successful_tickers"] == []
    assert payload["partial_failure_tickers"] == ["BB"]
    assert payload["failed_tickers"] == []
    assert payload["missing_tickers"] == []
    assert payload["ticker_results"]["BB"]["status"] == "partial_failure"
    assert payload["ticker_results"]["BB"]["successful_preset_count"] == 3
    assert payload["ticker_results"]["BB"]["failed_preset_count"] == 1
    assert any(warning.startswith("Partial success:") for warning in payload["warnings"])


def test_full_research_workflow_validation_output_with_multiple_tickers_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow should fail before comparison starts")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB,GME",
            "--validation-output",
            str(tmp_path / "reports" / "validation.md"),
        ],
    )

    with pytest.raises(SystemExit, match="--validation-output can only be used with one ticker"):
        main()


def test_full_research_workflow_requires_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow should fail before comparison starts")),
    )
    monkeypatch.setattr(sys, "argv", ["basket_runner.py", "--full-research-workflow"])

    with pytest.raises(SystemExit, match="--full-research-workflow requires --tickers"):
        main()


def test_full_research_workflow_rejects_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow should fail before comparison starts")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["basket_runner.py", "--full-research-workflow", "--tickers", "BB", "--dry-run"],
    )

    with pytest.raises(SystemExit, match="--dry-run is incompatible with --full-research-workflow"):
        main()


def test_full_research_workflow_rejects_data_check_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow should fail before comparison starts")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["basket_runner.py", "--full-research-workflow", "--tickers", "BB", "--data-check-only"],
    )

    with pytest.raises(SystemExit, match="--data-check-only is incompatible with --full-research-workflow"):
        main()


def test_full_research_workflow_rejects_mismatched_watchlist_output_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow should fail before comparison starts")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB",
            "--watchlist-output",
            str(tmp_path / "watchlist.md"),
            "--watchlist-path",
            str(tmp_path / "other_watchlist.json"),
        ],
    )

    with pytest.raises(SystemExit, match="--watchlist-output and --watchlist-path must refer to the same companion JSON path"):
        main()


def test_full_research_workflow_custom_watchlist_output_is_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}
    run_dir = tmp_path / "workflow-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.basket_runner.run_preset_comparison",
        lambda config, presets: SimpleNamespace(
            run_dir=run_dir,
            csv_path=run_dir / "preset_comparison.csv",
            markdown_path=run_dir / "preset_comparison.md",
            rows=[],
            preset_run_dirs={},
        ),
    )
    monkeypatch.setattr(
        "src.basket_runner.write_research_packet",
        lambda *args, **kwargs: SimpleNamespace(
            markdown_path=tmp_path / "research_packet.md",
            json_path=tmp_path / "research_packet.json",
            payload={},
        ),
    )
    monkeypatch.setattr(
        "src.basket_runner.append_research_journal",
        lambda *args, **kwargs: SimpleNamespace(journal_path=tmp_path / "research_journal.csv", rows_written=0),
    )

    def fake_build_research_watchlist(*, journal_path, output_path):
        observed["output_path"] = output_path
        return SimpleNamespace(
            journal_path=Path(journal_path),
            markdown_path=Path(output_path),
            json_path=Path(output_path).with_suffix(".json"),
            rows_reviewed=0,
            ticker_count=0,
            warnings=[],
        )

    monkeypatch.setattr("src.basket_runner.build_research_watchlist", fake_build_research_watchlist)
    monkeypatch.setattr(
        "src.basket_runner.build_validation_checklist",
        lambda *args, **kwargs: SimpleNamespace(
            journal_path=tmp_path / "research_journal.csv",
            watchlist_path=tmp_path / "custom_watchlist.json",
            markdown_path=tmp_path / "validation_checklist_BB.md",
            json_path=tmp_path / "validation_checklist_BB.json",
            ticker="BB",
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basket_runner.py",
            "--full-research-workflow",
            "--tickers",
            "BB",
            "--watchlist-output",
            str(tmp_path / "custom_watchlist.md"),
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert observed["output_path"] == str(tmp_path / "custom_watchlist.md")
    assert payload["research_watchlist_md"] == (tmp_path / "custom_watchlist.md").as_posix()
    assert payload["research_watchlist_json"] == (tmp_path / "custom_watchlist.json").as_posix()


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
