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
DECISION_SUMMARY_DISCLAIMER = (
    "Educational use only. This report is not financial advice, not an offer to buy or sell securities, "
    "and not a recommendation to take any trading action."
)


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_summary(
    path: Path,
    config: BasketRunConfig,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    run_dir: Path,
    data_checks: list[dict[str, Any]] | None = None,
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


def _stringify_reasoning(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


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

    if config.dry_run:
        _write_json(run_dir / "failures.json", failures)
        _write_json(run_dir / "data_check.json", data_checks)
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
                        else:
                            failure_payload = {
                                "ticker": ticker,
                                "classification": data_check_result.classification,
                                "diagnosis": data_check_result.diagnosis,
                                "error": data_check_result.diagnosis,
                            }
                            failures.append(failure_payload)
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
        _write_decisions_csv(run_dir / "combined_decisions.csv", decision_rows)
        _write_decision_summary(run_dir / "decision_summary.md", decision_rows)
        _write_summary(run_dir / "run_summary.md", config, successes, failures, run_dir, data_checks)
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Ollama basket with all analysts by default.")
    parser.add_argument("--tickers", type=str, help="Comma-separated ticker list")
    parser.add_argument("--basket-name", type=str, default="full-research", choices=sorted(BASKETS.keys()))
    parser.add_argument("--model", type=str, default="llama3.1:latest")
    parser.add_argument("--output-dir", type=str, help="Base directory for run outputs")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--analyst-preset", type=str, default="all", choices=sorted(ANALYST_PRESETS.keys()))
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
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
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

    if compare_presets:
        artifacts = run_preset_comparison(config, compare_presets)
        print(
            json.dumps(
                {
                    "run_dir": artifacts.run_dir.as_posix(),
                    "preset_comparison_csv": artifacts.csv_path.as_posix(),
                    "preset_comparison_md": artifacts.markdown_path.as_posix(),
                    "preset_run_dirs": artifacts.preset_run_dirs,
                    "dry_run": config.dry_run,
                    "data_check_only": config.data_check_only,
                    "offline_demo_data": config.offline_demo_data,
                },
                indent=2,
            )
        )
        return 0

    run_dir = run_basket(config)
    print(
        json.dumps(
            {
                "run_dir": run_dir.as_posix(),
                "dry_run": config.dry_run,
                "data_check_only": config.data_check_only,
                "offline_demo_data": config.offline_demo_data,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
