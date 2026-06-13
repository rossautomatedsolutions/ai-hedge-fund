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
from src.main import run_hedge_fund
from src.utils.analysts import ANALYST_ORDER
from src.utils.display import print_trading_output
from src.utils.ollama import ensure_ollama_and_model


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
    analysts: list[str]
    model_provider: str = "Ollama"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_summary(path: Path, config: BasketRunConfig, successes: list[dict[str, Any]], failures: list[dict[str, Any]], run_dir: Path) -> None:
    lines = [
        "# RAS Ollama Basket Run",
        "",
        f"- Basket: `{config.basket_name}`",
        f"- Model: `{config.model}`",
        f"- Provider: `{config.model_provider}`",
        f"- Tickers requested: `{', '.join(config.tickers)}`",
        f"- Successes: `{len(successes)}`",
        f"- Failures: `{len(failures)}`",
        f"- Output directory: `{run_dir.as_posix()}`",
    ]
    if failures:
        lines.extend(["", "## Failures"])
        for failure in failures:
            lines.append(f"- `{failure['ticker']}`: {failure['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flatten_decision(ticker: str, result: dict[str, Any]) -> dict[str, Any]:
    decision = (result.get("decisions") or {}).get(ticker) or {}
    return {
        "ticker": ticker,
        "action": decision.get("action", "UNKNOWN"),
        "quantity": decision.get("quantity", ""),
        "confidence": decision.get("confidence", ""),
        "reasoning": decision.get("reasoning", ""),
    }


def _write_decisions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "action", "quantity", "confidence", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)


def run_basket(config: BasketRunConfig) -> Path:
    run_dir = _build_run_dir(config.output_dir)
    start_date, end_date = resolve_dates(config.start_date, config.end_date, default_months_back=3)

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
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    if config.dry_run:
        _write_summary(run_dir / "run_summary.md", config, [], [], run_dir)
        return run_dir

    if not ensure_ollama_and_model(config.model, interactive=False):
        raise RuntimeError(f"Ollama model '{config.model}' is not available locally.")

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    combined_logs: list[str] = []
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for ticker in config.tickers:
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
            combined_logs.append(f"===== {ticker} =====\n{log_output}")
            (logs_dir / f"{ticker}.log").write_text(log_output, encoding="utf-8")
            successes.append({"ticker": ticker})
            decision_rows.append(_flatten_decision(ticker, result))
        except Exception as exc:
            log_output = buffer.getvalue()
            combined_logs.append(f"===== {ticker} =====\n{log_output}\nERROR: {exc}\n")
            (logs_dir / f"{ticker}.log").write_text(f"{log_output}\nERROR: {exc}\n", encoding="utf-8")
            failures.append({"ticker": ticker, "error": str(exc)})
            if not config.continue_on_error:
                break

    (run_dir / "raw_console_output.txt").write_text("\n".join(combined_logs), encoding="utf-8")
    _write_json(run_dir / "failures.json", failures)
    _write_decisions_csv(run_dir / "combined_decisions.csv", decision_rows)
    _write_summary(run_dir / "run_summary.md", config, successes, failures, run_dir)
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
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    analysts = [analyst_key for _, analyst_key in ANALYST_ORDER]
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
        analysts=analysts,
    )

    run_dir = run_basket(config)
    print(json.dumps({"run_dir": run_dir.as_posix(), "dry_run": config.dry_run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
