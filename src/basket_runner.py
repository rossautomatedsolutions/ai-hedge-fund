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
from src.utils.analysts import ANALYST_ORDER
from src.utils.display import print_trading_output
from src.utils.ollama import ensure_ollama_and_model
from src.tools.api import financial_data_request_settings, get_financial_data_request_settings


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
    model_provider: str = "Ollama"


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
        f"- Tickers requested: `{', '.join(config.tickers)}`",
        f"- Successes: `{len(successes)}`",
        f"- Failures: `{len(failures)}`",
        f"- Data checks passed: `{len(data_checks) - len(data_check_failures)}`",
        f"- Data checks failed: `{len(data_check_failures)}`",
        f"- Output directory: `{run_dir.as_posix()}`",
    ]
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
    action: Any,
    bullish_count: int,
    bearish_count: int,
    neutral_count: int,
) -> str:
    normalized_action = str(action or "").strip().lower()
    consensus = _analyst_consensus_label(bullish_count, bearish_count, neutral_count)

    if not normalized_action:
        return ""
    if consensus == "none":
        return "No analyst votes were captured; action reflects the final portfolio-manager output."
    if normalized_action == "buy" and consensus != "bullish":
        return (
            "Action is the final portfolio-manager output. Analyst votes shown here do not indicate a bullish consensus."
        )
    if normalized_action in {"sell", "short"} and consensus != "bearish":
        return (
            "Action is the final portfolio-manager output. Analyst votes shown here do not indicate a bearish consensus."
        )
    if normalized_action in {"hold", "cover"} and consensus == "mixed":
        return "Analyst votes are mixed; no single analyst consensus is shown in this report row."
    return ""


def _data_status(check: dict[str, Any] | None) -> str:
    if not check:
        return ""
    if check.get("ok"):
        return "ok"
    classification = check.get("classification")
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
        "report_note": _report_note(action, bullish_count, bearish_count, neutral_count),
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
        "report_note": "",
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
    lines = [
        "# Decision Summary",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
        "Action is the final portfolio-manager output. Analyst votes are shown separately and are not simple majority-vote trading logic.",
        "",
        "| Ticker | Action | Confidence | Analyst Votes | Analyst Consensus | Reasoning | Run Status | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
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
                (logs_dir / f"{ticker}.data_check.log").write_text(format_ticker_data_check(data_check_result) + "\n", encoding="utf-8")

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
                    combined_logs.append(f"===== {ticker} =====\n{format_ticker_data_check(data_check_result)}\nERROR: {message}\n")
                    (logs_dir / f"{ticker}.log").write_text(f"{format_ticker_data_check(data_check_result)}\nERROR: {message}\n", encoding="utf-8")
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
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-check-only", action="store_true", help="Check data reachability per ticker without calling any LLM")
    parser.add_argument("--request-timeout-seconds", type=int, default=15)
    parser.add_argument("--max-data-retries", type=int, default=3)
    parser.add_argument("--fast-data-mode", "--low-latency-data-mode", dest="fast_data_mode", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    analysts = resolve_analysts_for_preset(args.analyst_preset)
    tickers = _select_tickers(args.tickers, args.basket_name, args.max_symbols)

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
    )

    run_dir = run_basket(config)
    print(json.dumps({"run_dir": run_dir.as_posix(), "dry_run": config.dry_run, "data_check_only": config.data_check_only}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
