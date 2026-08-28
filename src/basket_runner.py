from __future__ import annotations

import argparse
import csv
import io
import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.cli.input import resolve_dates
from src.data_diagnostics import format_ticker_data_check, run_ticker_data_check, ticker_data_check_to_dict
from src.main import run_hedge_fund
from src.offline_demo_data import OFFLINE_DEMO_DATA_STATUS, OFFLINE_DEMO_DISCLAIMER
from src.research_workflows import (
    DECISION_SUMMARY_DISCLAIMER,
    HUMAN_REVIEW_LOG_FIELDNAMES,
    HUMAN_REVIEW_STATUSES,
    RESEARCH_JOURNAL_FIELDNAMES,
    HumanReviewLogArtifacts,
    HumanReviewSummaryArtifacts,
    ResearchJournalArtifacts,
    ResearchJournalReviewArtifacts,
    ResearchPacketArtifacts,
    ResearchWatchlistArtifacts,
    ValidationChecklistArtifacts,
    _csv_row_list,
    _json_cell,
    _stringify_reasoning,
    _write_json,
    append_research_journal,
    build_research_watchlist,
    build_validation_checklist,
    record_human_review,
    review_human_reviews,
    review_research_journal,
    write_research_packet,
)
from src.research_workflows.common import _default_research_journal_path
from src.signal_artifacts import (
    DEFAULT_SIGNAL_ARTIFACT_VERSION,
    SIGNAL_CAPTURE_FILENAME,
    export_signal_ledger_bundle,
)
from src.utils.analysts import ANALYST_ORDER
from src.utils.display import print_trading_output
from src.utils.ollama import ensure_ollama_and_model
from src.tools.api import financial_data_request_settings, get_financial_data_request_settings, offline_demo_data_mode


BASKETS = {
    "speculative": ["BB", "GME", "KULR", "OPTT", "F"],
    "large-cap": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH", "COST"],
    "sector-etfs": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
    "full-research": [
        "BB",
        "GME",
        "KULR",
        "OPTT",
        "F",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "JPM",
        "XOM",
        "UNH",
        "COST",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLY",
        "XLP",
        "XLI",
        "XLB",
        "XLU",
        "XLRE",
        "XLC",
    ],
}

ANALYST_PRESETS = {
    "all": None,
    "core": ["fundamentals_analyst", "technical_analyst", "valuation_analyst"],
    "no-news": None,
    "technical-only": ["technical_analyst"],
}
NEWS_DISABLED_ANALYSTS = {"sentiment_analyst", "news_sentiment_analyst"}
PRESET_COMPARISON_CSV_FIELDNAMES = [
    "ticker",
    "analyst_preset",
    "action",
    "quantity",
    "confidence",
    "analyst_vote_summary",
    "analyst_consensus",
    "comparison_note",
    "reasoning",
    "data_status",
    "run_status",
    "failure_classification",
    "run_dir",
    "log_path",
]
DECISION_CSV_FIELDNAMES = [
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
]
def _parse_tickers(raw_tickers: str | None) -> list[str]:
    if not raw_tickers:
        return []
    return [ticker.strip().upper() for ticker in raw_tickers.split(",") if ticker.strip()]


def _select_tickers(tickers: str | None, basket_name: str | None, max_symbols: int | None) -> list[str]:
    selected = _parse_tickers(tickers)
    if not selected:
        selected = list(BASKETS.get((basket_name or "full-research").lower(), BASKETS["full-research"]))
    if max_symbols is not None:
        selected = selected[:max_symbols]
    return selected


def _default_output_root() -> Path:
    return Path("outputs") / "ras_ollama_basket_runs"


def _build_run_dir(output_dir: str | None) -> Path:
    base_dir = Path(output_dir) if output_dir else _default_output_root()
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _portfolio_for_ticker(ticker: str) -> dict[str, Any]:
    return {
        "cash": 100000.0,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            }
        },
    }


@dataclass
class BasketRunConfig:
    tickers: list[str]
    basket_name: str
    model: str
    output_dir: str | None
    max_symbols: int | None
    continue_on_error: bool
    show_reasoning: bool
    start_date: str | None
    end_date: str | None
    dry_run: bool
    data_check_only: bool
    analyst_preset: str
    analysts: list[str]
    request_timeout_seconds: int
    max_data_retries: int
    fast_data_mode: bool
    offline_demo_data: bool = False
    model_provider: str = "Ollama"


@dataclass
class PresetComparisonArtifacts:
    run_dir: Path
    csv_path: Path
    markdown_path: Path
    rows: list[dict[str, Any]]
    preset_run_dirs: dict[str, str]


def resolve_analysts_for_preset(preset: str) -> list[str]:
    all_analysts = [analyst_key for _, analyst_key in ANALYST_ORDER]
    if preset == "all":
        return all_analysts
    if preset == "core":
        return list(ANALYST_PRESETS["core"] or [])
    if preset == "no-news":
        return [analyst for analyst in all_analysts if analyst not in NEWS_DISABLED_ANALYSTS]
    if preset == "technical-only":
        return list(ANALYST_PRESETS["technical-only"] or [])
    raise ValueError(f"Unknown analyst preset: {preset}")


def parse_compare_presets(raw_presets: str | None) -> list[str]:
    if not raw_presets:
        return []

    presets: list[str] = []
    seen: set[str] = set()
    valid_presets = set(ANALYST_PRESETS.keys())
    for raw_preset in raw_presets.split(","):
        preset = raw_preset.strip()
        if not preset:
            continue
        if preset not in valid_presets:
            raise ValueError(f"Unknown analyst preset: {preset}")
        if preset in seen:
            continue
        presets.append(preset)
        seen.add(preset)
    return presets


def resolve_data_request_options(
    *,
    request_timeout_seconds: int,
    max_data_retries: int,
    fast_data_mode: bool,
) -> dict[str, Any]:
    if fast_data_mode:
        return {
            "request_timeout_seconds": min(request_timeout_seconds, 5),
            "max_data_retries": min(max_data_retries, 1),
            "skip_optional_slow_data": True,
        }
    return {
        "request_timeout_seconds": request_timeout_seconds,
        "max_data_retries": max_data_retries,
        "skip_optional_slow_data": False,
    }


def _ticker_result_record(
    *,
    ticker: str,
    run_status: str,
    data_check: dict[str, Any] | None,
    failure: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "run_status": run_status,
        "data_check": data_check,
        "failure": failure,
        "result": result,
    }


def _write_ticker_result_records(path: Path, *, run_dir: Path, config: BasketRunConfig, records: list[dict[str, Any]]) -> None:
    _write_json(
        path,
        {
            "run_id": run_dir.name,
            "basket_name": config.basket_name,
            "analyst_preset": config.analyst_preset,
            "tickers": list(config.tickers),
            "analysts": list(config.analysts),
            "records": records,
        },
    )


def _signal_ledger_payload(artifacts) -> dict[str, Any]:
    return {
        "signal_ledger_csv": artifacts.csv_path.as_posix(),
        "signal_ledger_json": artifacts.json_path.as_posix(),
        "signal_ledger_manifest_json": artifacts.manifest_path.as_posix(),
        "trading_foundation_trades_csv": artifacts.trading_foundation_trades_path.as_posix(),
        "trading_foundation_handoff_manifest_json": artifacts.trading_foundation_manifest_path.as_posix(),
        "signal_ledger_run_id": artifacts.run_id,
        "signal_ledger_workflow_status": artifacts.workflow_status,
        "signal_ledger_record_count": artifacts.record_count,
        "signal_ledger_eligible_record_count": artifacts.eligible_record_count,
        "trading_foundation_prepared_trade_row_count": artifacts.prepared_trade_row_count,
    }


def _write_summary(
    path: Path,
    config: BasketRunConfig,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    run_dir: Path,
    data_checks: list[dict[str, Any]] | None = None,
    research_journal_path: Path | None = None,
) -> None:
    data_checks = data_checks or []
    data_check_failures = [check for check in data_checks if not check.get("ok")]
    effective_request_options = resolve_data_request_options(
        request_timeout_seconds=config.request_timeout_seconds,
        max_data_retries=config.max_data_retries,
        fast_data_mode=config.fast_data_mode,
    )
    lines = [
        "# RAS Ollama Basket Run",
        "",
        f"- Basket: `{config.basket_name}`",
        f"- Model: `{config.model}`",
        f"- Provider: `{config.model_provider}`",
        f"- Mode: `{'data-check-only' if config.data_check_only else 'full-run'}`",
        f"- Analyst preset: `{config.analyst_preset}`",
        f"- Analysts: `{', '.join(config.analysts)}`",
        f"- Request timeout seconds: `{effective_request_options['request_timeout_seconds']}`",
        f"- Max data retries: `{effective_request_options['max_data_retries']}`",
        f"- Fast data mode: `{config.fast_data_mode}`",
        f"- Offline demo data: `{config.offline_demo_data}`",
        f"- Tickers requested: `{', '.join(config.tickers)}`",
        f"- Successes: `{len(successes)}`",
        f"- Failures: `{len(failures)}`",
        f"- Data checks passed: `{len(data_checks) - len(data_check_failures)}`",
        f"- Data checks failed: `{len(data_check_failures)}`",
        f"- Output directory: `{run_dir.as_posix()}`",
    ]
    if research_journal_path is not None:
        lines.append(f"- Research journal appended: `{research_journal_path.as_posix()}`")
    if config.offline_demo_data:
        lines.append(f"- Demo data note: {OFFLINE_DEMO_DISCLAIMER}")
    if data_checks:
        lines.extend(["", "## Data Check"])
        for check in data_checks:
            if check.get("ok"):
                status = "passed"
            elif check.get("classification") == "partial_data":
                status = "partial_data"
            else:
                status = check.get("classification", "unknown_error")
            lines.append(f"- `{check['ticker']}`: {status} - {check.get('diagnosis', 'No diagnosis provided.')}")
    if failures:
        lines.extend(["", "## Failures"])
        for failure in failures:
            detail = failure.get("diagnosis") or failure.get("error") or "Unknown failure"
            classification = failure.get("classification")
            suffix = f" ({classification})" if classification else ""
            lines.append(f"- `{failure['ticker']}`: {detail}{suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_compare_summary(
    path: Path,
    *,
    config: BasketRunConfig,
    run_dir: Path,
    comparison_rows: list[dict[str, Any]],
    research_journal_path: Path | None = None,
) -> None:
    tickers = sorted({str(row.get("ticker") or "").strip() for row in comparison_rows if str(row.get("ticker") or "").strip()})
    presets = []
    seen_presets: set[str] = set()
    for row in comparison_rows:
        preset = str(row.get("analyst_preset") or "").strip()
        if preset and preset not in seen_presets:
            presets.append(preset)
            seen_presets.add(preset)
    failures = [row for row in comparison_rows if str(row.get("run_status") or "").strip() == "failed"]
    uses_offline_demo = any(
        str(row.get("data_status") or "").strip() in {OFFLINE_DEMO_DATA_STATUS, "fixture_data"} for row in comparison_rows
    )

    lines = [
        "# RAS Ollama Basket Run",
        "",
        f"- Basket: `{config.basket_name}`",
        f"- Model: `{config.model}`",
        f"- Provider: `{config.model_provider}`",
        "- Mode: `preset-comparison`",
        f"- Presets analyzed: `{', '.join(presets)}`",
        f"- Tickers requested: `{', '.join(tickers)}`",
        f"- Comparison rows: `{len(comparison_rows)}`",
        f"- Failed comparison rows: `{len(failures)}`",
        f"- Output directory: `{run_dir.as_posix()}`",
    ]
    if research_journal_path is not None:
        lines.append(f"- Research journal appended: `{research_journal_path.as_posix()}`")
    if uses_offline_demo:
        lines.append(f"- Demo data note: {OFFLINE_DEMO_DISCLAIMER}")
    if failures:
        lines.extend(["", "## Failures"])
        for row in failures:
            detail = str(row.get("reasoning") or "").strip() or "Unknown failure"
            classification = str(row.get("failure_classification") or "").strip()
            suffix = f" ({classification})" if classification else ""
            lines.append(f"- `{row.get('ticker', '')}` / `{row.get('analyst_preset', '')}`: {detail}{suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _signal_counts(result: dict[str, Any], ticker: str) -> tuple[int, int, int]:
    analyst_signals = result.get("analyst_signals", {})
    if not isinstance(analyst_signals, dict):
        return (0, 0, 0)

    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for signals in analyst_signals.values():
        if not isinstance(signals, dict):
            continue
        signal_payload = signals.get(ticker)
        if not isinstance(signal_payload, dict):
            continue
        signal = str(signal_payload.get("signal") or "").upper()
        if signal == "BULLISH":
            bullish_count += 1
        elif signal == "BEARISH":
            bearish_count += 1
        elif signal == "NEUTRAL":
            neutral_count += 1

    return (bullish_count, bearish_count, neutral_count)


def _analyst_vote_summary(bullish_count: int, bearish_count: int, neutral_count: int) -> str:
    return f"bullish={bullish_count}, bearish={bearish_count}, neutral={neutral_count}"


def _analyst_consensus_label(bullish_count: int, bearish_count: int, neutral_count: int) -> str:
    counts = {
        "bullish": bullish_count,
        "bearish": bearish_count,
        "neutral": neutral_count,
    }
    max_count = max(counts.values())
    if max_count <= 0:
        return "none"
    leaders = [label for label, count in counts.items() if count == max_count]
    if len(leaders) > 1:
        return "mixed"
    return leaders[0]


def _report_note(
    config: BasketRunConfig,
    action: Any,
    bullish_count: int,
    bearish_count: int,
    neutral_count: int,
) -> str:
    notes: list[str] = []
    if config.offline_demo_data:
        notes.append(OFFLINE_DEMO_DISCLAIMER)

    normalized_action = str(action or "").strip().lower()
    consensus = _analyst_consensus_label(bullish_count, bearish_count, neutral_count)

    if not normalized_action:
        return " ".join(notes).strip()
    if consensus == "none":
        notes.append("No analyst votes were captured; action reflects the final portfolio-manager output.")
        return " ".join(notes).strip()
    if normalized_action == "buy" and consensus != "bullish":
        notes.append("Action is the final portfolio-manager output. Analyst votes shown here do not indicate a bullish consensus.")
        return " ".join(notes).strip()
    if normalized_action in {"sell", "short"} and consensus != "bearish":
        notes.append("Action is the final portfolio-manager output. Analyst votes shown here do not indicate a bearish consensus.")
        return " ".join(notes).strip()
    if normalized_action in {"hold", "cover"} and consensus == "mixed":
        notes.append("Analyst votes are mixed; no single analyst consensus is shown in this report row.")
        return " ".join(notes).strip()
    return " ".join(notes).strip()


def _is_directional_action(action: str) -> bool:
    return action in {"buy", "sell", "short"}


def _comparison_note(
    action: Any,
    consensus: Any,
    bullish_count: Any,
    bearish_count: Any,
    neutral_count: Any,
    reasoning: Any,
) -> str:
    notes: list[str] = []
    normalized_action = str(action or "").strip().lower()
    normalized_consensus = str(consensus or "").strip().lower()
    normalized_reasoning = _stringify_reasoning(reasoning).lower()

    if not normalized_action or normalized_action in {"hold", "cover"} or normalized_consensus in {"", "none"}:
        return ""

    if _is_directional_action(normalized_action) and normalized_consensus != "mixed":
        if (
            (normalized_action == "buy" and normalized_consensus != "bullish")
            or (normalized_action in {"sell", "short"} and normalized_consensus != "bearish")
        ):
            notes.append("Final portfolio-manager action differs from analyst vote consensus.")

    if normalized_action == "buy" and normalized_consensus == "bearish":
        notes.append("Buy action is portfolio-manager output despite bearish analyst vote consensus.")
    elif normalized_action in {"sell", "short"} and normalized_consensus == "bullish":
        notes.append(f"{normalized_action.capitalize()} action is portfolio-manager output despite bullish analyst vote consensus.")
    elif _is_directional_action(normalized_action) and normalized_consensus == "mixed":
        notes.append("Action is directional despite mixed analyst votes.")

    try:
        bullish = int(bullish_count)
        bearish = int(bearish_count)
        neutral = int(neutral_count)
    except (TypeError, ValueError):
        bullish = bearish = neutral = 0

    if "majority bullish" in normalized_reasoning and bullish < max(bearish, neutral):
        notes.append("Reasoning should be read as portfolio-manager rationale, not analyst vote majority.")
    if "majority bearish" in normalized_reasoning and bearish < max(bullish, neutral):
        notes.append("Reasoning should be read as portfolio-manager rationale, not analyst vote majority.")

    return " ".join(notes).strip()


def _data_status(check: dict[str, Any] | None) -> str:
    if not check:
        return ""
    classification = check.get("classification")
    if classification in {OFFLINE_DEMO_DATA_STATUS, "fixture_data"}:
        return str(classification)
    if check.get("ok"):
        return "ok"
    return str(classification) if classification else "failed"


def _flatten_decision(
    ticker: str,
    result: dict[str, Any],
    *,
    config: BasketRunConfig,
    data_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = (result.get("decisions") or {}).get(ticker) or {}
    bullish_count, bearish_count, neutral_count = _signal_counts(result, ticker)
    action = decision.get("action", "UNKNOWN")
    return {
        "ticker": ticker,
        "action": action,
        "quantity": decision.get("quantity", ""),
        "confidence": decision.get("confidence", ""),
        "reasoning": _stringify_reasoning(decision.get("reasoning", "")),
        "analyst_vote_summary": _analyst_vote_summary(bullish_count, bearish_count, neutral_count),
        "analyst_consensus": _analyst_consensus_label(bullish_count, bearish_count, neutral_count),
        "report_note": _report_note(config, action, bullish_count, bearish_count, neutral_count),
        "analyst_preset": config.analyst_preset,
        "model": config.model,
        "provider": config.model_provider,
        "data_status": _data_status(data_check),
        "failure_classification": "",
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "run_status": "success",
    }


def _failure_row(
    ticker: str,
    *,
    config: BasketRunConfig,
    failure: dict[str, Any] | None = None,
    data_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classification = ""
    if failure:
        classification = str(failure.get("classification") or "")
    reasoning = ""
    if failure:
        reasoning = _stringify_reasoning(failure.get("diagnosis") or failure.get("error") or "")
    return {
        "ticker": ticker,
        "action": "",
        "quantity": "",
        "confidence": "",
        "reasoning": reasoning,
        "analyst_vote_summary": _analyst_vote_summary(0, 0, 0),
        "analyst_consensus": "none",
        "report_note": OFFLINE_DEMO_DISCLAIMER if config.offline_demo_data else "",
        "analyst_preset": config.analyst_preset,
        "model": config.model,
        "provider": config.model_provider,
        "data_status": _data_status(data_check),
        "failure_classification": classification,
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
        "run_status": "failed",
    }


def _write_decisions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_decision_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    uses_offline_demo = any(
        str(row.get("data_status") or "").strip() in {OFFLINE_DEMO_DATA_STATUS, "fixture_data"} for row in rows
    )
    lines = [
        "# Decision Summary",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
    ]
    if uses_offline_demo:
        lines.extend(
            [
                f"Offline/demo data note: {OFFLINE_DEMO_DISCLAIMER}",
                "",
            ]
        )
    lines.extend(
        [
            "Action is the final portfolio-manager output. Analyst votes are shown separately and are not simple majority-vote trading logic.",
            "",
            "| Ticker | Action | Confidence | Analyst Votes | Analyst Consensus | Reasoning | Run Status | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        action = str(row.get("action") or "").strip() or "FAILED"
        confidence = str(row.get("confidence") or "").strip()
        reasoning = str(row.get("reasoning") or "").replace("\n", " ").replace("|", "\\|").strip()
        analyst_votes = str(row.get("analyst_vote_summary") or "").replace("|", "\\|").strip()
        analyst_consensus = str(row.get("analyst_consensus") or "").replace("|", "\\|").strip()
        run_status = str(row.get("run_status") or "").strip() or "unknown"
        notes = str(row.get("report_note") or "").replace("\n", " ").replace("|", "\\|").strip()
        lines.append(
            f"| {row.get('ticker', '')} | {action} | {confidence} | {analyst_votes} | "
            f"{analyst_consensus} | {reasoning} | {run_status} | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_row_from_decision_row(row: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    ticker = str(row.get("ticker") or "").strip()
    return {
        "ticker": ticker,
        "analyst_preset": row.get("analyst_preset", ""),
        "action": row.get("action", ""),
        "quantity": row.get("quantity", ""),
        "confidence": row.get("confidence", ""),
        "analyst_vote_summary": row.get("analyst_vote_summary", ""),
        "analyst_consensus": row.get("analyst_consensus", ""),
        "comparison_note": _comparison_note(
            row.get("action", ""),
            row.get("analyst_consensus", ""),
            row.get("bullish_count", 0),
            row.get("bearish_count", 0),
            row.get("neutral_count", 0),
            row.get("reasoning", ""),
        ),
        "reasoning": row.get("reasoning", ""),
        "data_status": row.get("data_status", ""),
        "run_status": row.get("run_status", ""),
        "failure_classification": row.get("failure_classification", ""),
        "run_dir": run_dir.as_posix(),
        "log_path": (run_dir / "logs" / f"{ticker}.log").as_posix() if ticker else "",
    }


def _failed_comparison_row(
    ticker: str,
    *,
    preset: str,
    run_dir: Path,
    classification: str,
    reasoning: str,
    offline_demo_data: bool,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "analyst_preset": preset,
        "action": "",
        "quantity": "",
        "confidence": "",
        "analyst_vote_summary": "bullish=0, bearish=0, neutral=0",
        "analyst_consensus": "none",
        "comparison_note": "",
        "reasoning": reasoning,
        "data_status": OFFLINE_DEMO_DATA_STATUS if offline_demo_data else "",
        "run_status": "failed",
        "failure_classification": classification,
        "run_dir": run_dir.as_posix(),
        "log_path": (run_dir / "logs" / f"{ticker}.log").as_posix(),
    }


def _write_preset_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRESET_COMPARISON_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_preset_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    uses_offline_demo = any(
        str(row.get("data_status") or "").strip() in {OFFLINE_DEMO_DATA_STATUS, "fixture_data"} for row in rows
    )
    lines = [
        "# Preset Comparison",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
    ]
    if uses_offline_demo:
        lines.extend(
            [
                f"Offline/demo data note: {OFFLINE_DEMO_DISCLAIMER}",
                "",
            ]
        )

    tickers = sorted({str(row.get("ticker") or "").strip() for row in rows if str(row.get("ticker") or "").strip()})
    for ticker in tickers:
        ticker_rows = [row for row in rows if str(row.get("ticker") or "").strip() == ticker]
        lines.extend([f"## {ticker}", ""])
        for row in ticker_rows:
            action = str(row.get("action") or "").strip() or "FAILED"
            quantity = str(row.get("quantity") or "").strip() or "-"
            consensus = str(row.get("analyst_consensus") or "").strip() or "none"
            comparison_note = str(row.get("comparison_note") or "").strip()
            if comparison_note:
                lines.append(f"- {row.get('analyst_preset', '')}: {action} / {quantity} / {consensus} / {comparison_note}")
            else:
                lines.append(f"- {row.get('analyst_preset', '')}: {action} / {quantity} / {consensus}")
        lines.extend(
            [
                "",
                "| Preset | Action | Quantity | Confidence | Analyst Votes | Consensus | Comparison Note | Data Status | Run Status | Failure | Reasoning | Log |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in ticker_rows:
            action = str(row.get("action") or "").strip() or "FAILED"
            quantity = str(row.get("quantity") or "").strip() or "-"
            confidence = str(row.get("confidence") or "").strip() or "-"
            analyst_votes = str(row.get("analyst_vote_summary") or "").replace("|", "\\|").strip()
            consensus = str(row.get("analyst_consensus") or "").replace("|", "\\|").strip() or "none"
            comparison_note = str(row.get("comparison_note") or "").replace("\n", " ").replace("|", "\\|").strip() or "-"
            data_status = str(row.get("data_status") or "").replace("|", "\\|").strip() or "-"
            run_status = str(row.get("run_status") or "").replace("|", "\\|").strip() or "-"
            failure = str(row.get("failure_classification") or "").replace("|", "\\|").strip() or "-"
            reasoning = str(row.get("reasoning") or "").replace("\n", " ").replace("|", "\\|").strip() or "-"
            log_path = str(row.get("log_path") or "").replace("|", "\\|").strip() or "-"
            lines.append(
                f"| {row.get('analyst_preset', '')} | {action} | {quantity} | {confidence} | {analyst_votes} | "
                f"{consensus} | {comparison_note} | {data_status} | {run_status} | {failure} | {reasoning} | {log_path} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preset_comparison(config: BasketRunConfig, presets: list[str]) -> PresetComparisonArtifacts:
    comparison_run_dir = _build_run_dir(config.output_dir)
    preset_base_dir = comparison_run_dir / "preset_runs"
    preset_base_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, Any]] = []
    preset_run_dirs: dict[str, str] = {}

    for preset in presets:
        preset_config = BasketRunConfig(
            tickers=list(config.tickers),
            basket_name=config.basket_name,
            model=config.model,
            output_dir=str(preset_base_dir / preset),
            max_symbols=config.max_symbols,
            continue_on_error=config.continue_on_error,
            show_reasoning=config.show_reasoning,
            start_date=config.start_date,
            end_date=config.end_date,
            dry_run=config.dry_run,
            data_check_only=config.data_check_only,
            analyst_preset=preset,
            analysts=resolve_analysts_for_preset(preset),
            request_timeout_seconds=config.request_timeout_seconds,
            max_data_retries=config.max_data_retries,
            fast_data_mode=config.fast_data_mode,
            offline_demo_data=config.offline_demo_data,
            model_provider=config.model_provider,
        )
        try:
            preset_run_dir = run_basket(preset_config)
            preset_run_dirs[preset] = preset_run_dir.as_posix()
            csv_path = preset_run_dir / "combined_decisions.csv"
            if csv_path.exists():
                with csv_path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        comparison_rows.append(_comparison_row_from_decision_row(row, run_dir=preset_run_dir))
        except Exception as exc:
            preset_run_dir = preset_base_dir / preset
            preset_run_dir.mkdir(parents=True, exist_ok=True)
            preset_run_dirs[preset] = preset_run_dir.as_posix()
            classification, diagnosis = _classify_run_exception(exc)
            for ticker in config.tickers:
                comparison_rows.append(
                    _failed_comparison_row(
                        ticker,
                        preset=preset,
                        run_dir=preset_run_dir,
                        classification=classification,
                        reasoning=diagnosis,
                        offline_demo_data=config.offline_demo_data,
                    )
                )
            if not config.continue_on_error:
                raise

    csv_path = comparison_run_dir / "preset_comparison.csv"
    markdown_path = comparison_run_dir / "preset_comparison.md"
    _write_preset_comparison_csv(csv_path, comparison_rows)
    _write_preset_comparison_markdown(markdown_path, comparison_rows)
    return PresetComparisonArtifacts(
        run_dir=comparison_run_dir,
        csv_path=csv_path,
        markdown_path=markdown_path,
        rows=comparison_rows,
        preset_run_dirs=preset_run_dirs,
    )


def _classify_run_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lowered = message.lower()
    if "401" in message or "unauthorized" in lowered:
        return ("unauthorized_401", "Run failed because the provider rejected the financial data request with HTTP 401. The key may be invalid or expired.")
    if "ssl" in lowered:
        return ("ssl_error", "Run failed because of an SSL/TLS error while contacting the financial data provider.")
    if "10054" in message or "forcibly closed" in lowered or "connectionreseterror" in lowered:
        return ("connection_reset", "Run failed because the remote host reset the financial data connection.")
    if "timeout" in lowered or "timed out" in lowered:
        return ("timeout", "Run failed because the financial data request timed out.")
    return ("unknown_error", f"Run failed after passing data checks: {message}")


def run_basket(config: BasketRunConfig) -> Path:
    run_dir = _build_run_dir(config.output_dir)
    start_date, end_date = resolve_dates(config.start_date, config.end_date, default_months_back=3)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    request_options = resolve_data_request_options(
        request_timeout_seconds=config.request_timeout_seconds,
        max_data_retries=config.max_data_retries,
        fast_data_mode=config.fast_data_mode,
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "basket_name": config.basket_name,
        "tickers": config.tickers,
        "model": config.model,
        "model_provider": config.model_provider,
        "analysts": config.analysts,
        "continue_on_error": config.continue_on_error,
        "show_reasoning": config.show_reasoning,
        "start_date": start_date,
        "end_date": end_date,
        "dry_run": config.dry_run,
        "data_check_only": config.data_check_only,
        "analyst_preset": config.analyst_preset,
        "request_timeout_seconds": request_options["request_timeout_seconds"],
        "max_data_retries": request_options["max_data_retries"],
        "fast_data_mode": config.fast_data_mode,
        "offline_demo_data": config.offline_demo_data,
        "skip_optional_slow_data": request_options["skip_optional_slow_data"],
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    combined_logs: list[str] = []
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    data_checks: list[dict[str, Any]] = []
    ticker_result_records: list[dict[str, Any]] = []

    if config.dry_run:
        _write_json(run_dir / "failures.json", failures)
        _write_json(run_dir / "data_check.json", data_checks)
        _write_ticker_result_records(run_dir / SIGNAL_CAPTURE_FILENAME, run_dir=run_dir, config=config, records=ticker_result_records)
        (run_dir / "raw_console_output.txt").write_text("", encoding="utf-8")
        _write_decision_summary(run_dir / "decision_summary.md", decision_rows)
        _write_summary(run_dir / "run_summary.md", config, successes, failures, run_dir, data_checks)
        return run_dir

    if not config.data_check_only and not ensure_ollama_and_model(config.model, interactive=False):
        raise RuntimeError(f"Ollama model '{config.model}' is not available locally.")

    print(
        f"Writing basket logs to {logs_dir.as_posix()} | "
        f"timeout={request_options['request_timeout_seconds']}s | "
        f"max_data_retries={request_options['max_data_retries']} | "
        f"fast_data_mode={config.fast_data_mode}"
    )

    try:
        with offline_demo_data_mode(enabled=config.offline_demo_data):
            with financial_data_request_settings(
                timeout_seconds=request_options["request_timeout_seconds"],
                max_attempts=request_options["max_data_retries"] + 1,
                skip_optional_slow_data=request_options["skip_optional_slow_data"],
            ):
                active_settings = get_financial_data_request_settings()
                for ticker in config.tickers:
                    print(f"[ticker:{ticker}] starting data check | log={logs_dir / f'{ticker}.data_check.log'}")
                    data_check_result = run_ticker_data_check(ticker, start_date, end_date)
                    data_check_payload = ticker_data_check_to_dict(data_check_result)
                    data_checks.append(data_check_payload)
                    (logs_dir / f"{ticker}.data_check.log").write_text(
                        format_ticker_data_check(data_check_result) + "\n",
                        encoding="utf-8",
                    )

                    if config.data_check_only:
                        print(f"[ticker:{ticker}] data-check-only complete with classification={data_check_result.classification}")
                        (logs_dir / f"{ticker}.log").write_text(format_ticker_data_check(data_check_result) + "\n", encoding="utf-8")
                        if data_check_result.ok:
                            successes.append({"ticker": ticker, "mode": "data-check-only"})
                            ticker_result_records.append(
                                _ticker_result_record(
                                    ticker=ticker,
                                    run_status="data_check_only",
                                    data_check=data_check_payload,
                                )
                            )
                        else:
                            failure_payload = {
                                "ticker": ticker,
                                "classification": data_check_result.classification,
                                "diagnosis": data_check_result.diagnosis,
                                "error": data_check_result.diagnosis,
                            }
                            failures.append(failure_payload)
                            ticker_result_records.append(
                                _ticker_result_record(
                                    ticker=ticker,
                                    run_status="failed",
                                    data_check=data_check_payload,
                                    failure=failure_payload,
                                )
                            )
                            decision_rows.append(
                                _failure_row(
                                    ticker,
                                    config=config,
                                    failure=failure_payload,
                                    data_check=data_check_payload,
                                )
                            )
                        continue

                    if not data_check_result.ok:
                        message = f"Data check failed before LLM run: {data_check_result.diagnosis}"
                        print(f"[ticker:{ticker}] stopping before live run: {message}")
                        combined_logs.append(
                            f"===== {ticker} =====\n{format_ticker_data_check(data_check_result)}\nERROR: {message}\n"
                        )
                        (logs_dir / f"{ticker}.log").write_text(
                            f"{format_ticker_data_check(data_check_result)}\nERROR: {message}\n",
                            encoding="utf-8",
                        )
                        failure_payload = {
                            "ticker": ticker,
                            "classification": data_check_result.classification,
                            "diagnosis": data_check_result.diagnosis,
                            "error": message,
                        }
                        failures.append(failure_payload)
                        ticker_result_records.append(
                            _ticker_result_record(
                                ticker=ticker,
                                run_status="failed",
                                data_check=data_check_payload,
                                failure=failure_payload,
                            )
                        )
                        decision_rows.append(
                            _failure_row(
                                ticker,
                                config=config,
                                failure=failure_payload,
                                data_check=data_check_payload,
                            )
                        )
                        if not config.continue_on_error:
                            break
                        continue

                    print(
                        f"[ticker:{ticker}] starting live run | analysts={','.join(config.analysts)} | "
                        f"timeout={active_settings.timeout_seconds}s | attempts={active_settings.max_attempts} | "
                        f"skip_optional_slow_data={active_settings.skip_optional_slow_data}"
                    )
                    buffer = io.StringIO()
                    try:
                        with redirect_stdout(buffer):
                            result = run_hedge_fund(
                                tickers=[ticker],
                                start_date=start_date,
                                end_date=end_date,
                                portfolio=_portfolio_for_ticker(ticker),
                                show_reasoning=config.show_reasoning,
                                selected_analysts=config.analysts,
                                model_name=config.model,
                                model_provider=config.model_provider,
                            )
                            print_trading_output(result)
                        log_output = buffer.getvalue()
                        full_log_output = f"{format_ticker_data_check(data_check_result)}\n\n{log_output}"
                        combined_logs.append(f"===== {ticker} =====\n{full_log_output}")
                        (logs_dir / f"{ticker}.log").write_text(full_log_output, encoding="utf-8")
                        print(f"[ticker:{ticker}] completed successfully | log={logs_dir / f'{ticker}.log'}")
                        successes.append({"ticker": ticker})
                        ticker_result_records.append(
                            _ticker_result_record(
                                ticker=ticker,
                                run_status="success",
                                data_check=data_check_payload,
                                result=result,
                            )
                        )
                        decision_rows.append(_flatten_decision(ticker, result, config=config, data_check=data_check_payload))
                    except Exception as exc:
                        log_output = buffer.getvalue()
                        full_log_output = f"{format_ticker_data_check(data_check_result)}\n\n{log_output}\nERROR: {exc}\n"
                        combined_logs.append(f"===== {ticker} =====\n{full_log_output}")
                        (logs_dir / f"{ticker}.log").write_text(full_log_output, encoding="utf-8")
                        classification, diagnosis = _classify_run_exception(exc)
                        print(f"[ticker:{ticker}] failed with {classification}: {diagnosis}")
                        failure_payload = {
                            "ticker": ticker,
                            "classification": classification,
                            "diagnosis": diagnosis,
                            "error": str(exc),
                        }
                        failures.append(failure_payload)
                        ticker_result_records.append(
                            _ticker_result_record(
                                ticker=ticker,
                                run_status="failed",
                                data_check=data_check_payload,
                                failure=failure_payload,
                            )
                        )
                        decision_rows.append(
                            _failure_row(
                                ticker,
                                config=config,
                                failure=failure_payload,
                                data_check=data_check_payload,
                            )
                        )
                        if not config.continue_on_error:
                            break
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Writing partial basket artifacts before exit.")
        failures.append(
            {
                "ticker": "RUN_ABORTED",
                "classification": "unknown_error",
                "diagnosis": "Run interrupted by user via Ctrl+C.",
                "error": "KeyboardInterrupt",
            }
        )
    finally:
        (run_dir / "raw_console_output.txt").write_text("\n".join(combined_logs), encoding="utf-8")
        _write_json(run_dir / "failures.json", failures)
        _write_json(run_dir / "data_check.json", data_checks)
        _write_ticker_result_records(run_dir / SIGNAL_CAPTURE_FILENAME, run_dir=run_dir, config=config, records=ticker_result_records)
        _write_decisions_csv(run_dir / "combined_decisions.csv", decision_rows)
        _write_decision_summary(run_dir / "decision_summary.md", decision_rows)
        _write_summary(run_dir / "run_summary.md", config, successes, failures, run_dir, data_checks)
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Ollama basket with all analysts by default.")
    parser.add_argument("--tickers", type=str, help="Comma-separated ticker list")
    parser.add_argument("--ticker", type=str, help="Single ticker filter used by research-journal review mode.")
    parser.add_argument("--basket-name", type=str, default="full-research", choices=sorted(BASKETS.keys()))
    parser.add_argument("--model", type=str, default="llama3.1:latest")
    parser.add_argument("--output-dir", type=str, help="Base directory for run outputs")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--analyst-preset", type=str, default="all", choices=sorted(ANALYST_PRESETS.keys()))
    parser.add_argument(
        "--full-research-workflow",
        action="store_true",
        help="Run preset comparison, research packet generation, journal append, watchlist refresh, and per-ticker validation checklists in one research-only workflow.",
    )
    parser.add_argument(
        "--compare-presets",
        type=str,
        help="Comma-separated analyst presets to run for side-by-side comparison reports.",
    )
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-check-only", action="store_true", help="Check data reachability per ticker without calling any LLM")
    parser.add_argument("--request-timeout-seconds", type=int, default=15)
    parser.add_argument("--max-data-retries", type=int, default=3)
    parser.add_argument("--fast-data-mode", "--low-latency-data-mode", dest="fast_data_mode", action="store_true")
    parser.add_argument(
        "--offline-demo-data",
        "--demo-data-mode",
        dest="offline_demo_data",
        action="store_true",
        help="Use static local fixture data for educational/demo runs without live financial-data API access.",
    )
    parser.add_argument(
        "--research-packet",
        action="store_true",
        help="Write a research_packet.md and research_packet.json summary into the run output directory.",
    )
    parser.add_argument(
        "--export-signal-ledger",
        action="store_true",
        help="Write a canonical signal ledger and Trading Foundation handoff bundle for the run or for an existing run directory.",
    )
    parser.add_argument(
        "--signal-ledger-output",
        type=str,
        help="Custom directory for signal ledger and Trading Foundation handoff artifacts. Defaults to the source run directory.",
    )
    parser.add_argument(
        "--signal-artifact-version",
        type=str,
        default=DEFAULT_SIGNAL_ARTIFACT_VERSION,
        help="Version label to stamp into exported signal-ledger records and manifests.",
    )
    parser.add_argument(
        "--signal-ledger-source-run-dir",
        type=str,
        help="Existing basket run directory to export without rerunning analysts. Requires --export-signal-ledger.",
    )
    parser.add_argument(
        "--append-research-journal",
        action="store_true",
        help="Append per-ticker research packet rows to a durable local CSV journal.",
    )
    parser.add_argument(
        "--research-journal-path",
        type=str,
        help="Custom CSV path for the durable research journal append.",
    )
    parser.add_argument(
        "--review-research-journal",
        action="store_true",
        help="Read the durable research journal CSV and write a markdown review report without running analysts or data workflows.",
    )
    parser.add_argument(
        "--journal-review-output",
        type=str,
        help="Custom markdown output path for research-journal review mode. A JSON companion is written beside it.",
    )
    parser.add_argument(
        "--research-watchlist",
        action="store_true",
        help="Read the durable research journal CSV and write a watchlist summary report without running analysts or data workflows.",
    )
    parser.add_argument(
        "--watchlist-output",
        type=str,
        help="Custom markdown output path for research watchlist mode. A JSON companion is written beside it.",
    )
    parser.add_argument(
        "--watchlist-path",
        type=str,
        help="Optional JSON path for research watchlist input used by validation-checklist mode.",
    )
    parser.add_argument(
        "--validation-checklist",
        action="store_true",
        help="Read the research journal and optional watchlist JSON to create a manual validation checklist for one ticker.",
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        help="Custom markdown output path for validation-checklist mode. A JSON companion is written beside it.",
    )
    parser.add_argument(
        "--validation-checklist-path",
        type=str,
        help="Optional JSON path for validation checklist input used by record-human-review mode.",
    )
    parser.add_argument(
        "--record-human-review",
        action="store_true",
        help="Append a human validation outcome row to the local human review log without running analysts or data workflows.",
    )
    parser.add_argument(
        "--human-status",
        type=str,
        help="Human review outcome. Allowed values: Watchlist, Reject, Deep Research, Paper Trade Candidate, Trade Candidate.",
    )
    parser.add_argument(
        "--review-notes",
        type=str,
        help="Optional free-form notes for record-human-review mode.",
    )
    parser.add_argument(
        "--human-review-log-path",
        type=str,
        help="Custom CSV path for the append-only human review log.",
    )
    parser.add_argument(
        "--review-human-reviews",
        action="store_true",
        help="Read the human review log CSV and write a markdown/json summary report without running analysts or data workflows.",
    )
    parser.add_argument(
        "--human-review-summary-output",
        type=str,
        help="Custom markdown output path for review-human-reviews mode. A JSON companion is written beside it.",
    )
    return parser


def _full_workflow_watchlist_output_path(args: argparse.Namespace) -> str | None:
    if args.watchlist_output:
        if args.watchlist_path and Path(args.watchlist_output).with_suffix(".json") != Path(args.watchlist_path):
            raise SystemExit(
                "--watchlist-output and --watchlist-path must refer to the same companion JSON path in --full-research-workflow mode."
            )
        return args.watchlist_output
    if args.watchlist_path:
        return str(Path(args.watchlist_path).with_suffix(".md"))
    return None


def _full_workflow_output_root(config: BasketRunConfig) -> Path | None:
    if not config.output_dir:
        return None
    return Path(config.output_dir)


def _full_workflow_research_journal_path(
    args: argparse.Namespace,
    config: BasketRunConfig,
) -> str | None:
    if args.research_journal_path:
        return args.research_journal_path
    output_root = _full_workflow_output_root(config)
    if output_root is None:
        return None
    return str(output_root / "research_journal.csv")


def _full_workflow_watchlist_markdown_path(
    args: argparse.Namespace,
    config: BasketRunConfig,
) -> str | None:
    explicit_path = _full_workflow_watchlist_output_path(args)
    if explicit_path:
        return explicit_path
    output_root = _full_workflow_output_root(config)
    if output_root is None:
        return None
    return str(output_root / "research_watchlist.md")


def _full_workflow_validation_output_path(
    args: argparse.Namespace,
    config: BasketRunConfig,
    *,
    ticker: str,
) -> str | None:
    if args.validation_output:
        return args.validation_output
    output_root = _full_workflow_output_root(config)
    if output_root is None:
        return None
    return str(output_root / f"validation_checklist_{ticker.upper()}.md")


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        normalized = str(warning).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _full_workflow_ticker_results(
    *,
    tickers: list[str],
    comparison_rows: list[dict[str, Any]],
    presets_requested: list[str],
) -> dict[str, dict[str, Any]]:
    ticker_results: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        rows = [row for row in comparison_rows if str(row.get("ticker") or "").strip().upper() == ticker.upper()]
        successful_rows = [row for row in rows if str(row.get("run_status") or "").strip().lower() == "success"]
        failed_rows = [row for row in rows if str(row.get("run_status") or "").strip().lower() != "success"]
        seen_presets: list[str] = []
        seen_presets_set: set[str] = set()
        for row in rows:
            preset = str(row.get("analyst_preset") or "").strip()
            if preset and preset not in seen_presets_set:
                seen_presets.append(preset)
                seen_presets_set.add(preset)

        if not rows:
            status = "missing"
        elif failed_rows and successful_rows:
            status = "partial_failure"
        elif failed_rows:
            status = "failed"
        else:
            status = "success"

        ticker_results[ticker] = {
            "status": status,
            "successful_preset_count": len(successful_rows),
            "failed_preset_count": len(failed_rows),
            "presets_requested": list(presets_requested),
            "presets_observed": seen_presets,
        }
    return ticker_results


def _run_full_research_workflow(
    *,
    args: argparse.Namespace,
    config: BasketRunConfig,
    compare_presets: list[str],
) -> dict[str, Any]:
    if not args.tickers:
        raise SystemExit("--full-research-workflow requires --tickers.")
    if len(config.tickers) > 1 and args.validation_output:
        raise SystemExit("--validation-output can only be used with one ticker in --full-research-workflow mode.")
    if config.dry_run:
        raise SystemExit("--dry-run is incompatible with --full-research-workflow because no research artifacts would be generated.")
    if config.data_check_only:
        raise SystemExit(
            "--data-check-only is incompatible with --full-research-workflow because current behavior only validates data reachability and does not produce research packets."
        )
    journal_path = _full_workflow_research_journal_path(args, config)
    watchlist_output_path = _full_workflow_watchlist_markdown_path(args, config)

    comparison_artifacts = run_preset_comparison(config, compare_presets)
    research_artifacts = write_research_packet(
        comparison_artifacts.run_dir,
        config=config,
        comparison_rows=comparison_artifacts.rows,
        preset_run_dirs=comparison_artifacts.preset_run_dirs,
    )
    journal_artifacts = append_research_journal(
        research_artifacts,
        journal_path=journal_path,
    )
    _write_compare_summary(
        comparison_artifacts.run_dir / "run_summary.md",
        config=config,
        run_dir=comparison_artifacts.run_dir,
        comparison_rows=comparison_artifacts.rows,
        research_journal_path=journal_artifacts.journal_path,
    )

    watchlist_artifacts = build_research_watchlist(
        journal_path=journal_artifacts.journal_path,
        output_path=watchlist_output_path,
    )

    ticker_results = _full_workflow_ticker_results(
        tickers=list(config.tickers),
        comparison_rows=comparison_artifacts.rows,
        presets_requested=compare_presets,
    )
    successful_tickers = [ticker for ticker, result in ticker_results.items() if result["status"] == "success"]
    partial_failure_tickers = [ticker for ticker, result in ticker_results.items() if result["status"] == "partial_failure"]
    failed_tickers = [ticker for ticker, result in ticker_results.items() if result["status"] == "failed"]
    missing_tickers = [ticker for ticker, result in ticker_results.items() if result["status"] == "missing"]

    validation_checklists: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    if partial_failure_tickers:
        warnings.append(
            "Partial success: one or more presets failed for "
            + ", ".join(f"`{ticker}`" for ticker in partial_failure_tickers)
            + ", so downstream artifacts summarize incomplete comparisons."
        )
    if failed_tickers:
        warnings.append(
            "Failed tickers: "
            + ", ".join(f"`{ticker}`" for ticker in failed_tickers)
            + " had no successful preset comparisons."
        )
    if missing_tickers:
        warnings.append(
            "Missing comparison rows: "
            + ", ".join(f"`{ticker}`" for ticker in missing_tickers)
            + " did not appear in preset comparison output."
        )
    warnings.extend(watchlist_artifacts.warnings)
    for ticker in config.tickers:
        validation_artifacts = build_validation_checklist(
            ticker=ticker,
            journal_path=journal_artifacts.journal_path,
            watchlist_path=watchlist_artifacts.json_path,
            output_path=_full_workflow_validation_output_path(args, config, ticker=ticker),
        )
        validation_checklists[ticker] = {
            "markdown": validation_artifacts.markdown_path.as_posix(),
            "json": validation_artifacts.json_path.as_posix(),
        }
        warnings.extend(validation_artifacts.warnings)

    if failed_tickers or missing_tickers:
        workflow_status = "failed" if len(failed_tickers) + len(missing_tickers) == len(config.tickers) else "partial_success"
    elif partial_failure_tickers:
        workflow_status = "partial_success"
    else:
        workflow_status = "success"

    successful_comparison_rows = [
        row for row in comparison_artifacts.rows if str(row.get("run_status") or "").strip().lower() == "success"
    ]
    failed_comparison_rows = [
        row for row in comparison_artifacts.rows if str(row.get("run_status") or "").strip().lower() != "success"
    ]

    return {
        "run_dir": comparison_artifacts.run_dir.as_posix(),
        "workflow_status": workflow_status,
        "tickers": list(config.tickers),
        "requested_presets": list(compare_presets),
        "preset_run_dirs": comparison_artifacts.preset_run_dirs,
        "comparison_row_count": len(comparison_artifacts.rows),
        "successful_comparison_row_count": len(successful_comparison_rows),
        "failed_comparison_row_count": len(failed_comparison_rows),
        "successful_tickers": successful_tickers,
        "partial_failure_tickers": partial_failure_tickers,
        "failed_tickers": failed_tickers,
        "missing_tickers": missing_tickers,
        "ticker_results": ticker_results,
        "preset_comparison_csv": comparison_artifacts.csv_path.as_posix(),
        "preset_comparison_md": comparison_artifacts.markdown_path.as_posix(),
        "research_packet_md": research_artifacts.markdown_path.as_posix(),
        "research_packet_json": research_artifacts.json_path.as_posix(),
        "research_journal_csv": journal_artifacts.journal_path.as_posix(),
        "research_journal_rows_written": journal_artifacts.rows_written,
        "research_watchlist_md": watchlist_artifacts.markdown_path.as_posix(),
        "research_watchlist_json": watchlist_artifacts.json_path.as_posix(),
        "research_watchlist_rows_reviewed": watchlist_artifacts.rows_reviewed,
        "research_watchlist_ticker_count": watchlist_artifacts.ticker_count,
        "validation_checklists": validation_checklists,
        "validation_checklist_count": len(validation_checklists),
        "warnings": _dedupe_warnings(warnings),
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.review_human_reviews:
        summary_ticker = str(args.ticker or "").strip().upper() or None
        artifacts = review_human_reviews(
            log_path=args.human_review_log_path,
            ticker=summary_ticker,
            output_path=args.human_review_summary_output,
        )
        print(
            json.dumps(
                {
                    "review_human_reviews": True,
                    "source_human_review_log_path": artifacts.log_path.as_posix(),
                    "human_review_summary_markdown": artifacts.markdown_path.as_posix(),
                    "human_review_summary_json": artifacts.json_path.as_posix(),
                    "ticker": artifacts.ticker,
                    "rows_reviewed": artifacts.rows_reviewed,
                    "reviewed_ticker_count": artifacts.reviewed_ticker_count,
                    "warnings": artifacts.warnings,
                },
                indent=2,
            )
        )
        return 0

    if args.record_human_review:
        artifacts = record_human_review(
            ticker=args.ticker,
            human_status=args.human_status,
            review_notes=args.review_notes,
            validation_checklist_path=args.validation_checklist_path,
            journal_path=args.research_journal_path,
            watchlist_path=args.watchlist_path,
            human_review_log_path=args.human_review_log_path,
        )
        print(
            json.dumps(
                {
                    "human_review_log_path": artifacts.log_path.as_posix(),
                    "rows_written": artifacts.rows_written,
                    "ticker": artifacts.ticker,
                    "human_status": artifacts.human_status,
                    "warnings": artifacts.warnings,
                },
                indent=2,
            )
        )
        return 0

    if args.validation_checklist:
        checklist_ticker = str(args.ticker or "").strip().upper() or None
        artifacts = build_validation_checklist(
            ticker=checklist_ticker,
            journal_path=args.research_journal_path,
            watchlist_path=args.watchlist_path,
            output_path=args.validation_output,
        )
        print(
            json.dumps(
                {
                    "validation_checklist": True,
                    "ticker": artifacts.ticker,
                    "source_journal_path": artifacts.journal_path.as_posix(),
                    "source_watchlist_path": artifacts.watchlist_path.as_posix() if artifacts.watchlist_path else None,
                    "validation_markdown": artifacts.markdown_path.as_posix(),
                    "validation_json": artifacts.json_path.as_posix(),
                    "warnings": artifacts.warnings,
                },
                indent=2,
            )
        )
        return 0

    if args.research_watchlist:
        artifacts = build_research_watchlist(
            journal_path=args.research_journal_path,
            output_path=args.watchlist_output,
        )
        print(
            json.dumps(
                {
                    "research_watchlist": True,
                    "source_journal_path": artifacts.journal_path.as_posix(),
                    "watchlist_markdown": artifacts.markdown_path.as_posix(),
                    "watchlist_json": artifacts.json_path.as_posix(),
                    "rows_reviewed": artifacts.rows_reviewed,
                    "ticker_count": artifacts.ticker_count,
                    "warnings": artifacts.warnings,
                },
                indent=2,
            )
        )
        return 0

    if args.review_research_journal:
        review_ticker = str(args.ticker or "").strip().upper() or None
        artifacts = review_research_journal(
            journal_path=args.research_journal_path,
            ticker=review_ticker,
            output_path=args.journal_review_output,
        )
        print(
            json.dumps(
                {
                    "review_research_journal": True,
                    "source_journal_path": artifacts.journal_path.as_posix(),
                    "journal_review_markdown": artifacts.markdown_path.as_posix(),
                    "journal_review_json": artifacts.json_path.as_posix(),
                    "ticker": artifacts.ticker,
                    "entries_reviewed": artifacts.entries_reviewed,
                    "warnings": artifacts.warnings,
                },
                indent=2,
            )
        )
        return 0

    if args.signal_ledger_source_run_dir:
        if not args.export_signal_ledger:
            raise SystemExit("--signal-ledger-source-run-dir requires --export-signal-ledger.")
        if args.tickers:
            raise SystemExit("--signal-ledger-source-run-dir is export-only and cannot be combined with --tickers.")
        if args.compare_presets:
            raise SystemExit("--signal-ledger-source-run-dir cannot be combined with --compare-presets.")
        if args.full_research_workflow:
            raise SystemExit("--signal-ledger-source-run-dir cannot be combined with --full-research-workflow.")
        export_artifacts = export_signal_ledger_bundle(
            run_dir=args.signal_ledger_source_run_dir,
            output_dir=args.signal_ledger_output,
            artifact_version=args.signal_artifact_version,
        )
        print(json.dumps(_signal_ledger_payload(export_artifacts), indent=2))
        return 0

    tickers = _select_tickers(args.tickers, args.basket_name, args.max_symbols)
    compare_presets = parse_compare_presets(args.compare_presets)
    analysts = resolve_analysts_for_preset(args.analyst_preset)

    if not tickers:
        raise SystemExit("No tickers selected.")

    config = BasketRunConfig(
        tickers=tickers,
        basket_name=args.basket_name,
        model=args.model,
        output_dir=args.output_dir,
        max_symbols=args.max_symbols,
        continue_on_error=args.continue_on_error,
        show_reasoning=args.show_reasoning,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=args.dry_run,
        data_check_only=args.data_check_only,
        analyst_preset=args.analyst_preset,
        analysts=analysts,
        request_timeout_seconds=args.request_timeout_seconds,
        max_data_retries=args.max_data_retries,
        fast_data_mode=args.fast_data_mode,
        offline_demo_data=args.offline_demo_data,
    )

    if args.full_research_workflow:
        if args.export_signal_ledger:
            raise SystemExit("--export-signal-ledger is currently limited to single basket runs or --signal-ledger-source-run-dir.")
        workflow_presets = compare_presets or ["technical-only", "core", "no-news", "all"]
        print(
            json.dumps(
                _run_full_research_workflow(
                    args=args,
                    config=config,
                    compare_presets=workflow_presets,
                ),
                indent=2,
            )
        )
        return 0

    if compare_presets:
        if args.export_signal_ledger:
            raise SystemExit("--export-signal-ledger is currently limited to single basket runs or --signal-ledger-source-run-dir.")
        artifacts = run_preset_comparison(config, compare_presets)
        research_packet_path = None
        research_packet_json = None
        research_journal_path = None
        if args.research_packet:
            research_artifacts = write_research_packet(
                artifacts.run_dir,
                config=config,
                comparison_rows=artifacts.rows,
                preset_run_dirs=artifacts.preset_run_dirs,
            )
            research_packet_path = research_artifacts.markdown_path.as_posix()
            research_packet_json = research_artifacts.json_path.as_posix()
            if args.append_research_journal:
                journal_artifacts = append_research_journal(
                    research_artifacts,
                    journal_path=args.research_journal_path,
                )
                research_journal_path = journal_artifacts.journal_path.as_posix()
                _write_compare_summary(
                    artifacts.run_dir / "run_summary.md",
                    config=config,
                    run_dir=artifacts.run_dir,
                    comparison_rows=artifacts.rows,
                    research_journal_path=journal_artifacts.journal_path,
                )
        print(
            json.dumps(
                {
                    "run_dir": artifacts.run_dir.as_posix(),
                    "preset_comparison_csv": artifacts.csv_path.as_posix(),
                    "preset_comparison_md": artifacts.markdown_path.as_posix(),
                    "research_packet_md": research_packet_path,
                    "research_packet_json": research_packet_json,
                    "research_journal_csv": research_journal_path,
                    "preset_run_dirs": artifacts.preset_run_dirs,
                    "dry_run": config.dry_run,
                    "data_check_only": config.data_check_only,
                    "offline_demo_data": config.offline_demo_data,
                },
                indent=2,
            )
        )
        return 0

    if args.export_signal_ledger and config.dry_run:
        raise SystemExit("--export-signal-ledger is incompatible with --dry-run because no signal decisions are produced.")
    if args.export_signal_ledger and config.data_check_only:
        raise SystemExit("--export-signal-ledger is incompatible with --data-check-only because no signal decisions are produced.")

    run_dir = run_basket(config)
    research_packet_path = None
    research_packet_json = None
    research_journal_path = None
    signal_ledger_payload: dict[str, Any] = {}
    if args.export_signal_ledger:
        export_artifacts = export_signal_ledger_bundle(
            run_dir=run_dir,
            output_dir=args.signal_ledger_output,
            artifact_version=args.signal_artifact_version,
        )
        signal_ledger_payload = _signal_ledger_payload(export_artifacts)
    if args.research_packet:
        research_artifacts = write_research_packet(run_dir, config=config)
        research_packet_path = research_artifacts.markdown_path.as_posix()
        research_packet_json = research_artifacts.json_path.as_posix()
        if args.append_research_journal:
            journal_artifacts = append_research_journal(
                research_artifacts,
                journal_path=args.research_journal_path,
            )
            research_journal_path = journal_artifacts.journal_path.as_posix()
            failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
            data_checks = json.loads((run_dir / "data_check.json").read_text(encoding="utf-8"))
            successes = [
                {"ticker": row["ticker"]}
                for row in _csv_row_list(run_dir / "combined_decisions.csv")
                if str(row.get("run_status") or "").strip() == "success"
            ]
            _write_summary(
                run_dir / "run_summary.md",
                config,
                successes,
                failures,
                run_dir,
                data_checks,
                research_journal_path=journal_artifacts.journal_path,
            )
    print(
        json.dumps(
            {
                "run_dir": run_dir.as_posix(),
                "research_packet_md": research_packet_path,
                "research_packet_json": research_packet_json,
                "research_journal_csv": research_journal_path,
                "dry_run": config.dry_run,
                "data_check_only": config.data_check_only,
                "offline_demo_data": config.offline_demo_data,
                **signal_ledger_payload,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

