from __future__ import annotations

import csv
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.offline_demo_data import OFFLINE_DEMO_DATA_STATUS
from src.research_workflows.common import _write_json
from src.utils.analysts import ANALYST_CONFIG


DEFAULT_SIGNAL_ARTIFACT_VERSION = "1.0"
SIGNAL_CAPTURE_FILENAME = "ticker_results.json"
SIGNAL_LEDGER_FILENAME = "signal_ledger.json"
SIGNAL_LEDGER_CSV_FILENAME = "signal_ledger.csv"
SIGNAL_LEDGER_MANIFEST_FILENAME = "signal_ledger_manifest.json"
TRADING_FOUNDATION_TRADES_FILENAME = "trading_foundation_trades.csv"
TRADING_FOUNDATION_HANDOFF_MANIFEST_FILENAME = "trading_foundation_handoff_manifest.json"
TRADING_FOUNDATION_ENGINE_PATH = "C:/GitHub/Trading_Foundation/backtesting/run_backtest_from_trades.py"
TRADING_FOUNDATION_ENGINE_ENTRYPOINT = "backtesting.run_backtest_from_trades.run"
CONFIDENCE_SEMANTICS = "ordinal_0_to_100_uncalibrated_non_probability"
SIGNAL_LEDGER_FIELDNAMES = [
    "artifact_version",
    "decision_id",
    "run_id",
    "as_of_timestamp",
    "decision_date",
    "ticker",
    "analyst_name",
    "analyst_family",
    "analyst_preset",
    "signal",
    "signal_score",
    "confidence_score",
    "confidence_semantics",
    "intended_horizon",
    "model_provider",
    "model_name",
    "model_version",
    "prompt_or_strategy_version",
    "data_mode",
    "data_cutoff",
    "price_data_cutoff",
    "fundamental_data_cutoff",
    "generated_at",
    "workflow_status",
    "ticker_status",
    "evidence_available",
    "partial_failure",
    "failure_reason",
    "source_artifact",
    "source_run_directory",
    "is_backtest_eligible",
    "backtest_ineligibility_reason",
    "source_signal_label",
]
TRADING_FOUNDATION_TRADE_FIELDNAMES = [
    "symbol",
    "entry_date",
    "side",
    "strategy",
    "confidence_score",
    "run_id",
    "decision_id",
    "ticker",
    "analyst_name",
    "analyst_family",
    "analyst_preset",
    "signal",
    "source_signal_label",
]


@dataclass(frozen=True)
class AnalystDescriptor:
    family: str
    display_name: str
    signal_keys: tuple[str, ...]
    supports_price_cutoff: bool
    supports_fundamental_cutoff: bool


@dataclass(frozen=True)
class SignalLedgerRecord:
    artifact_version: str
    decision_id: str
    run_id: str
    as_of_timestamp: str | None
    decision_date: str | None
    ticker: str
    analyst_name: str
    analyst_family: str
    analyst_preset: str | None
    signal: str
    signal_score: int | None
    confidence_score: float | int | None
    confidence_semantics: str | None
    intended_horizon: str | None
    model_provider: str | None
    model_name: str | None
    model_version: str | None
    prompt_or_strategy_version: str | None
    data_mode: str | None
    data_cutoff: str | None
    price_data_cutoff: str | None
    fundamental_data_cutoff: str | None
    generated_at: str | None
    workflow_status: str
    ticker_status: str
    evidence_available: bool
    partial_failure: bool
    failure_reason: str | None
    source_artifact: str
    source_run_directory: str
    is_backtest_eligible: bool
    backtest_ineligibility_reason: str | None
    source_signal_label: str | None


@dataclass(frozen=True)
class SignalLedgerArtifacts:
    source_run_dir: Path
    output_dir: Path
    csv_path: Path
    json_path: Path
    manifest_path: Path
    trading_foundation_trades_path: Path
    trading_foundation_manifest_path: Path
    run_id: str
    workflow_status: str
    record_count: int
    eligible_record_count: int
    prepared_trade_row_count: int


@dataclass(frozen=True)
class CapturedCutoffs:
    data_cutoff: str | None
    price_data_cutoff: str | None
    fundamental_data_cutoff: str | None


def export_signal_ledger_bundle(
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    artifact_version: str = DEFAULT_SIGNAL_ARTIFACT_VERSION,
) -> SignalLedgerArtifacts:
    source_run_dir = Path(run_dir).resolve()
    output_root = Path(output_dir).resolve() if output_dir else source_run_dir
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest_path = source_run_dir / "run_manifest.json"
    signal_capture_path = source_run_dir / SIGNAL_CAPTURE_FILENAME
    if not run_manifest_path.exists():
        raise ValueError(f"Run directory is missing run_manifest.json: {source_run_dir.as_posix()}")
    if not signal_capture_path.exists():
        raise ValueError(
            "Run directory is missing ticker-level signal capture metadata required for signal-ledger export: "
            f"{signal_capture_path.as_posix()}"
        )

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    capture = json.loads(signal_capture_path.read_text(encoding="utf-8"))
    records = _build_signal_records(
        manifest=manifest,
        capture=capture,
        artifact_version=artifact_version,
        source_run_dir=source_run_dir,
    )
    trade_rows = _build_trading_foundation_trade_rows(records)

    csv_path = output_root / SIGNAL_LEDGER_CSV_FILENAME
    json_path = output_root / SIGNAL_LEDGER_FILENAME
    manifest_path = output_root / SIGNAL_LEDGER_MANIFEST_FILENAME
    trades_path = output_root / TRADING_FOUNDATION_TRADES_FILENAME
    handoff_manifest_path = output_root / TRADING_FOUNDATION_HANDOFF_MANIFEST_FILENAME

    _write_signal_ledger_csv(csv_path, records)
    _write_json(json_path, [asdict(record) for record in records])
    _write_trading_foundation_trades_csv(trades_path, trade_rows)

    run_id = str(capture.get("run_id") or source_run_dir.name)
    workflow_status = _workflow_status(manifest=manifest, capture=capture)
    exported_at = datetime.now().isoformat()
    _write_json(
        manifest_path,
        {
            "artifact_version": artifact_version,
            "generated_at": exported_at,
            "run_id": run_id,
            "workflow_status": workflow_status,
            "source_run_directory": source_run_dir.as_posix(),
            "source_run_manifest": run_manifest_path.as_posix(),
            "source_signal_capture": signal_capture_path.as_posix(),
            "record_count": len(records),
            "eligible_record_count": sum(1 for record in records if record.is_backtest_eligible),
            "prepared_trade_row_count": len(trade_rows),
            "schema_fields": list(SIGNAL_LEDGER_FIELDNAMES),
            "records_by_analyst_family": _count_by(records, "analyst_family"),
            "records_by_data_mode": _count_by(records, "data_mode"),
            "records_by_signal": _count_by(records, "signal"),
            "backtest_ineligibility_reasons": _count_by(records, "backtest_ineligibility_reason"),
            "selected_integration_path": {
                "selected_engine_path": TRADING_FOUNDATION_ENGINE_PATH,
                "selected_engine_entrypoint": TRADING_FOUNDATION_ENGINE_ENTRYPOINT,
                "selected_engine_input_contract": {
                    "required_columns": ["symbol", "entry_date", "side"],
                    "optional_columns": ["strategy", "confidence_score", "run_id", "decision_id"],
                },
                "why_selected": (
                    "Trading Foundation's mapped-trade backtest entrypoint is the narrowest existing engine that can "
                    "consume timestamped external trade rows without changing Trading Foundation source."
                ),
                "why_not_auto_invoked": (
                    "Trading Foundation currently writes outputs under backtesting/backtests and core/outputs and its "
                    "historical-data helper writes cache/diagnostic files under core/cache and core/outputs. "
                    "AI Hedge Fund therefore stops at a prepared handoff bundle to preserve the read-only upstream boundary."
                ),
            },
            "exports": {
                "signal_ledger_csv": csv_path.as_posix(),
                "signal_ledger_json": json_path.as_posix(),
                "signal_ledger_manifest_json": manifest_path.as_posix(),
                "trading_foundation_trades_csv": trades_path.as_posix(),
                "trading_foundation_handoff_manifest_json": handoff_manifest_path.as_posix(),
            },
        },
    )
    _write_json(
        handoff_manifest_path,
        {
            "artifact_version": artifact_version,
            "generated_at": exported_at,
            "run_id": run_id,
            "workflow_status": workflow_status,
            "source_run_directory": source_run_dir.as_posix(),
            "prepared_trades_csv": trades_path.as_posix(),
            "prepared_trade_row_count": len(trade_rows),
            "eligible_signal_record_count": sum(1 for record in records if record.is_backtest_eligible),
            "selected_engine_path": TRADING_FOUNDATION_ENGINE_PATH,
            "selected_engine_entrypoint": TRADING_FOUNDATION_ENGINE_ENTRYPOINT,
            "engine_input_contract": {
                "required_columns": ["symbol", "entry_date", "side"],
                "optional_columns": ["strategy", "confidence_score", "run_id", "decision_id"],
            },
            "auto_invocation_blocked": True,
            "auto_invocation_blocker": (
                "AI Hedge Fund does not invoke Trading Foundation automatically because the current Trading Foundation "
                "engine writes outputs and cache/diagnostic files inside its own repository tree."
            ),
            "notes": [
                "Only rows marked is_backtest_eligible=true are converted into Trading Foundation trade rows.",
                "Current AI Hedge Fund live/offline research exports fail closed unless point-in-time eligibility can be demonstrated.",
            ],
        },
    )

    return SignalLedgerArtifacts(
        source_run_dir=source_run_dir,
        output_dir=output_root,
        csv_path=csv_path,
        json_path=json_path,
        manifest_path=manifest_path,
        trading_foundation_trades_path=trades_path,
        trading_foundation_manifest_path=handoff_manifest_path,
        run_id=run_id,
        workflow_status=workflow_status,
        record_count=len(records),
        eligible_record_count=sum(1 for record in records if record.is_backtest_eligible),
        prepared_trade_row_count=len(trade_rows),
    )


def _build_signal_records(
    *,
    manifest: dict[str, Any],
    capture: dict[str, Any],
    artifact_version: str,
    source_run_dir: Path,
) -> list[SignalLedgerRecord]:
    requested_tickers = [str(ticker).strip().upper() for ticker in manifest.get("tickers", []) if str(ticker).strip()]
    analysts = [str(analyst).strip() for analyst in manifest.get("analysts", []) if str(analyst).strip()]
    capture_by_ticker = {
        str(entry.get("ticker") or "").strip().upper(): entry
        for entry in capture.get("records", [])
        if str(entry.get("ticker") or "").strip()
    }
    workflow_status = _workflow_status(manifest=manifest, capture=capture)
    run_id = str(capture.get("run_id") or source_run_dir.name)
    decision_date = _date_or_none(manifest.get("end_date"))
    generated_at = _string_or_none(manifest.get("created_at"))
    artifact_records: list[SignalLedgerRecord] = []

    for ticker in requested_tickers:
        ticker_capture = capture_by_ticker.get(
            ticker,
            {
                "ticker": ticker,
                "run_status": "not_executed",
                "result": None,
                "failure": {
                    "classification": "not_executed",
                    "diagnosis": "Ticker was not processed because the run stopped before reaching it.",
                },
                "data_check": None,
            },
        )
        ticker_run_status = str(ticker_capture.get("run_status") or "").strip().lower() or "failed"
        data_check = ticker_capture.get("data_check") if isinstance(ticker_capture.get("data_check"), dict) else None
        result = ticker_capture.get("result") if isinstance(ticker_capture.get("result"), dict) else {}
        analyst_signals = result.get("analyst_signals") if isinstance(result.get("analyst_signals"), dict) else {}
        decisions = result.get("decisions") if isinstance(result.get("decisions"), dict) else {}
        failure = ticker_capture.get("failure") if isinstance(ticker_capture.get("failure"), dict) else None
        partial_data = str((data_check or {}).get("classification") or "").strip().lower() == "partial_data"
        cutoffs = _captured_cutoffs(
            manifest=manifest,
            data_check=data_check,
            ticker=ticker,
        )

        analyst_payloads: dict[str, tuple[AnalystDescriptor, dict[str, Any] | None, str | None]] = {}
        missing_analysts: list[str] = []
        present_analysts: list[str] = []
        for analyst_family in analysts:
            descriptor = _analyst_descriptor(analyst_family)
            payload, signal_key = _find_analyst_payload(analyst_signals, descriptor, ticker)
            analyst_payloads[analyst_family] = (descriptor, payload, signal_key)
            if payload is None:
                missing_analysts.append(analyst_family)
            else:
                present_analysts.append(analyst_family)

        for analyst_family in analysts:
            descriptor, payload, signal_key = analyst_payloads[analyst_family]
            normalized_signal, signal_score, raw_signal = _normalize_signal((payload or {}).get("signal"))
            evidence_available = payload is not None
            row_partial_failure = _row_partial_failure(
                ticker_run_status=ticker_run_status,
                evidence_available=evidence_available,
                partial_data=partial_data,
                missing_analysts=missing_analysts,
                present_analysts=present_analysts,
                portfolio_decision=False,
            )
            ticker_status = _ticker_status(
                ticker_run_status=ticker_run_status,
                evidence_available=evidence_available,
                partial_data=partial_data,
            )
            data_mode = _data_mode(
                offline_demo_data=bool(manifest.get("offline_demo_data")),
                decision_date=decision_date,
                evidence_available=evidence_available,
            )
            record = SignalLedgerRecord(
                artifact_version=artifact_version,
                decision_id=_decision_id(
                    artifact_version=artifact_version,
                    run_id=run_id,
                    ticker=ticker,
                    analyst_family=descriptor.family,
                    analyst_preset=_string_or_none(manifest.get("analyst_preset")),
                    decision_date=decision_date,
                    as_of_timestamp=None,
                    source_signal_label=raw_signal,
                    source_artifact=_source_artifact_label(
                        ticker=ticker,
                        analyst_family=descriptor.family,
                        signal_key=signal_key,
                        decision=False,
                    ),
                ),
                run_id=run_id,
                as_of_timestamp=None,
                decision_date=decision_date,
                ticker=ticker,
                analyst_name=descriptor.display_name,
                analyst_family=descriptor.family,
                analyst_preset=_string_or_none(manifest.get("analyst_preset")),
                signal=normalized_signal,
                signal_score=signal_score,
                confidence_score=_numeric_or_none((payload or {}).get("confidence")),
                confidence_semantics=CONFIDENCE_SEMANTICS if evidence_available else None,
                intended_horizon=None,
                model_provider=_string_or_none(manifest.get("model_provider")),
                model_name=_string_or_none(manifest.get("model")),
                model_version=None,
                prompt_or_strategy_version=None,
                data_mode=data_mode,
                data_cutoff=cutoffs.data_cutoff,
                price_data_cutoff=cutoffs.price_data_cutoff if descriptor.supports_price_cutoff else None,
                fundamental_data_cutoff=cutoffs.fundamental_data_cutoff if descriptor.supports_fundamental_cutoff else None,
                generated_at=generated_at,
                workflow_status=workflow_status,
                ticker_status=ticker_status,
                evidence_available=evidence_available,
                partial_failure=row_partial_failure,
                failure_reason=_failure_reason(
                    ticker_run_status=ticker_run_status,
                    ticker=ticker,
                    analyst_family=descriptor.family,
                    evidence_available=evidence_available,
                    partial_data=partial_data,
                    missing_analysts=missing_analysts,
                    present_analysts=present_analysts,
                    data_check=data_check,
                    failure=failure,
                    portfolio_decision=False,
                ),
                source_artifact=_source_artifact_label(
                    ticker=ticker,
                    analyst_family=descriptor.family,
                    signal_key=signal_key,
                    decision=False,
                ),
                source_run_directory=source_run_dir.as_posix(),
                is_backtest_eligible=_is_backtest_eligible(
                    data_mode=data_mode,
                    decision_date=decision_date,
                    as_of_timestamp=None,
                    evidence_available=evidence_available,
                    signal=normalized_signal,
                    partial_failure=row_partial_failure,
                ),
                backtest_ineligibility_reason=_backtest_ineligibility_reason(
                    data_mode=data_mode,
                    decision_date=decision_date,
                    as_of_timestamp=None,
                    evidence_available=evidence_available,
                    signal=normalized_signal,
                    partial_failure=row_partial_failure,
                ),
                source_signal_label=raw_signal,
            )
            artifact_records.append(record)

        portfolio_decision = decisions.get(ticker) if isinstance(decisions.get(ticker), dict) else None
        portfolio_signal, portfolio_signal_score, raw_portfolio_signal = _normalize_signal((portfolio_decision or {}).get("action"))
        portfolio_evidence = portfolio_decision is not None
        portfolio_partial_failure = _row_partial_failure(
            ticker_run_status=ticker_run_status,
            evidence_available=portfolio_evidence,
            partial_data=partial_data,
            missing_analysts=missing_analysts,
            present_analysts=present_analysts,
            portfolio_decision=True,
        )
        portfolio_data_mode = _data_mode(
            offline_demo_data=bool(manifest.get("offline_demo_data")),
            decision_date=decision_date,
            evidence_available=portfolio_evidence,
        )
        artifact_records.append(
            SignalLedgerRecord(
                artifact_version=artifact_version,
                decision_id=_decision_id(
                    artifact_version=artifact_version,
                    run_id=run_id,
                    ticker=ticker,
                    analyst_family="portfolio_manager",
                    analyst_preset=_string_or_none(manifest.get("analyst_preset")),
                    decision_date=decision_date,
                    as_of_timestamp=None,
                    source_signal_label=raw_portfolio_signal,
                    source_artifact=_source_artifact_label(
                        ticker=ticker,
                        analyst_family="portfolio_manager",
                        signal_key=None,
                        decision=True,
                    ),
                ),
                run_id=run_id,
                as_of_timestamp=None,
                decision_date=decision_date,
                ticker=ticker,
                analyst_name="Portfolio Manager",
                analyst_family="portfolio_manager",
                analyst_preset=_string_or_none(manifest.get("analyst_preset")),
                signal=portfolio_signal,
                signal_score=portfolio_signal_score,
                confidence_score=_numeric_or_none((portfolio_decision or {}).get("confidence")),
                confidence_semantics=CONFIDENCE_SEMANTICS if portfolio_evidence else None,
                intended_horizon=None,
                model_provider=_string_or_none(manifest.get("model_provider")),
                model_name=_string_or_none(manifest.get("model")),
                model_version=None,
                prompt_or_strategy_version=None,
                data_mode=portfolio_data_mode,
                data_cutoff=cutoffs.data_cutoff,
                price_data_cutoff=None,
                fundamental_data_cutoff=None,
                generated_at=generated_at,
                workflow_status=workflow_status,
                ticker_status=_ticker_status(
                    ticker_run_status=ticker_run_status,
                    evidence_available=portfolio_evidence,
                    partial_data=partial_data,
                ),
                evidence_available=portfolio_evidence,
                partial_failure=portfolio_partial_failure,
                failure_reason=_failure_reason(
                    ticker_run_status=ticker_run_status,
                    ticker=ticker,
                    analyst_family="portfolio_manager",
                    evidence_available=portfolio_evidence,
                    partial_data=partial_data,
                    missing_analysts=missing_analysts,
                    present_analysts=present_analysts,
                    data_check=data_check,
                    failure=failure,
                    portfolio_decision=True,
                ),
                source_artifact=_source_artifact_label(
                    ticker=ticker,
                    analyst_family="portfolio_manager",
                    signal_key=None,
                    decision=True,
                ),
                source_run_directory=source_run_dir.as_posix(),
                is_backtest_eligible=_is_backtest_eligible(
                    data_mode=portfolio_data_mode,
                    decision_date=decision_date,
                    as_of_timestamp=None,
                    evidence_available=portfolio_evidence,
                    signal=portfolio_signal,
                    partial_failure=portfolio_partial_failure,
                ),
                backtest_ineligibility_reason=_backtest_ineligibility_reason(
                    data_mode=portfolio_data_mode,
                    decision_date=decision_date,
                    as_of_timestamp=None,
                    evidence_available=portfolio_evidence,
                    signal=portfolio_signal,
                    partial_failure=portfolio_partial_failure,
                ),
                source_signal_label=raw_portfolio_signal,
            )
        )

    return artifact_records


def _captured_cutoffs(
    *,
    manifest: dict[str, Any],
    data_check: dict[str, Any] | None,
    ticker: str,
) -> CapturedCutoffs:
    _ = manifest
    _ = data_check
    _ = ticker
    return CapturedCutoffs(
        data_cutoff=None,
        price_data_cutoff=None,
        fundamental_data_cutoff=None,
    )


def _analyst_descriptor(analyst_family: str) -> AnalystDescriptor:
    config = ANALYST_CONFIG.get(analyst_family, {})
    display_name = str(config.get("display_name") or analyst_family)
    signal_keys = {f"{analyst_family}_agent"}
    agent_func = config.get("agent_func")
    if agent_func is not None:
        try:
            agent_id_param = inspect.signature(agent_func).parameters.get("agent_id")
            if agent_id_param is not None and agent_id_param.default is not inspect._empty:
                signal_keys.add(str(agent_id_param.default))
        except (TypeError, ValueError):
            pass
    return AnalystDescriptor(
        family=analyst_family,
        display_name=display_name,
        signal_keys=tuple(sorted(signal_keys)),
        supports_price_cutoff=analyst_family == "technical_analyst",
        supports_fundamental_cutoff=analyst_family in {"fundamentals_analyst", "valuation_analyst", "growth_analyst"},
    )


def _find_analyst_payload(
    analyst_signals: dict[str, Any],
    descriptor: AnalystDescriptor,
    ticker: str,
) -> tuple[dict[str, Any] | None, str | None]:
    for signal_key in descriptor.signal_keys:
        signal_block = analyst_signals.get(signal_key)
        if not isinstance(signal_block, dict):
            continue
        payload = signal_block.get(ticker)
        if isinstance(payload, dict):
            return payload, signal_key
    return None, None


def _normalize_signal(raw_signal: Any) -> tuple[str, int | None, str | None]:
    if raw_signal is None:
        return "abstain", None, None
    raw_label = str(raw_signal).strip().lower()
    if not raw_label:
        return "abstain", None, None
    if raw_label in {"bullish", "buy", "cover"}:
        return "bullish", 1, raw_label
    if raw_label in {"bearish", "sell", "short"}:
        return "bearish", -1, raw_label
    if raw_label in {"neutral", "hold"}:
        return "neutral", 0, raw_label
    return "unmapped", None, raw_label


def _workflow_status(*, manifest: dict[str, Any], capture: dict[str, Any]) -> str:
    if bool(manifest.get("dry_run")):
        return "dry_run"
    if bool(manifest.get("data_check_only")):
        return "data_check_only"
    requested_tickers = [str(ticker).strip().upper() for ticker in manifest.get("tickers", []) if str(ticker).strip()]
    capture_by_ticker = {
        str(entry.get("ticker") or "").strip().upper(): str(entry.get("run_status") or "").strip().lower()
        for entry in capture.get("records", [])
        if str(entry.get("ticker") or "").strip()
    }
    statuses = [capture_by_ticker.get(ticker, "not_executed") for ticker in requested_tickers]
    success_count = sum(1 for status in statuses if status == "success")
    if success_count == len(statuses) and statuses:
        return "success"
    if success_count > 0:
        return "partial_success"
    return "failed"


def _ticker_status(*, ticker_run_status: str, evidence_available: bool, partial_data: bool) -> str:
    if evidence_available:
        if partial_data:
            return "partial_data"
        return "success"
    if ticker_run_status == "success":
        return "missing_signal"
    if ticker_run_status == "data_check_only":
        return "data_check_only"
    if ticker_run_status == "not_executed":
        return "not_executed"
    return "failed"


def _row_partial_failure(
    *,
    ticker_run_status: str,
    evidence_available: bool,
    partial_data: bool,
    missing_analysts: list[str],
    present_analysts: list[str],
    portfolio_decision: bool,
) -> bool:
    if partial_data and evidence_available:
        return True
    if ticker_run_status == "success" and evidence_available and missing_analysts and portfolio_decision:
        return True
    if ticker_run_status == "success" and not evidence_available and present_analysts:
        return True
    return False


def _data_mode(*, offline_demo_data: bool, decision_date: str | None, evidence_available: bool) -> str:
    if offline_demo_data:
        return OFFLINE_DEMO_DATA_STATUS
    if not evidence_available or not decision_date:
        return "unknown_provenance"
    if decision_date < date.today().isoformat():
        return "historical_live_research_unverified_pit"
    return "current_live_research"


def _failure_reason(
    *,
    ticker_run_status: str,
    ticker: str,
    analyst_family: str,
    evidence_available: bool,
    partial_data: bool,
    missing_analysts: list[str],
    present_analysts: list[str],
    data_check: dict[str, Any] | None,
    failure: dict[str, Any] | None,
    portfolio_decision: bool,
) -> str | None:
    messages: list[str] = []
    if ticker_run_status == "data_check_only":
        messages.append("Source run was data-check-only, so no signal decision was generated.")
    elif ticker_run_status == "not_executed":
        messages.append("Ticker was not processed because the run stopped before reaching it.")
    elif ticker_run_status != "success":
        detail = _string_or_none((failure or {}).get("diagnosis")) or _string_or_none((failure or {}).get("error"))
        if detail:
            messages.append(detail)
        elif data_check:
            messages.append(_string_or_none(data_check.get("diagnosis")) or "Ticker failed before a signal could be emitted.")
    elif not evidence_available:
        if portfolio_decision:
            messages.append("Portfolio manager did not emit a final decision for this ticker.")
        else:
            messages.append(f"Requested analyst `{analyst_family}` did not emit a signal for `{ticker}`.")
    if partial_data:
        messages.append(_string_or_none((data_check or {}).get("diagnosis")) or "Data check reported partial_data for this ticker.")
    if portfolio_decision and missing_analysts and evidence_available:
        messages.append(
            "One or more requested analyst signals were missing: "
            + ", ".join(f"`{family}`" for family in missing_analysts)
            + "."
        )
    if missing_analysts and not evidence_available and present_analysts and analyst_family in missing_analysts:
        messages.append("This row is an explicit abstain placeholder for incomplete analyst output.")
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = str(message).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return "; ".join(deduped) if deduped else None


def _is_backtest_eligible(
    *,
    data_mode: str | None,
    decision_date: str | None,
    as_of_timestamp: str | None,
    evidence_available: bool,
    signal: str,
    partial_failure: bool,
) -> bool:
    if not evidence_available or signal in {"abstain", "unmapped"}:
        return False
    if partial_failure:
        return False
    if not as_of_timestamp and not decision_date:
        return False
    return data_mode == "historical_point_in_time"


def _backtest_ineligibility_reason(
    *,
    data_mode: str | None,
    decision_date: str | None,
    as_of_timestamp: str | None,
    evidence_available: bool,
    signal: str,
    partial_failure: bool,
) -> str | None:
    if not evidence_available or signal in {"abstain", "unmapped"}:
        return "no_signal_emitted"
    if partial_failure:
        return "partial_or_incomplete_signal_evidence"
    if data_mode == OFFLINE_DEMO_DATA_STATUS:
        return "offline_demo_fixture_not_historical_evidence"
    if not as_of_timestamp and not decision_date:
        return "missing_as_of_or_decision_date"
    if data_mode == "current_live_research":
        return "current_live_research_not_backtest_ready"
    if data_mode == "historical_live_research_unverified_pit":
        return "historical_window_not_proven_point_in_time_safe"
    if data_mode == "unknown_provenance":
        return "unknown_data_provenance"
    return None


def _decision_id(
    *,
    artifact_version: str,
    run_id: str,
    ticker: str,
    analyst_family: str,
    analyst_preset: str | None,
    decision_date: str | None,
    as_of_timestamp: str | None,
    source_signal_label: str | None,
    source_artifact: str,
) -> str:
    identity = {
        "artifact_version": artifact_version,
        "run_id": run_id,
        "ticker": ticker,
        "analyst_family": analyst_family,
        "analyst_preset": analyst_preset,
        "decision_date": decision_date,
        "as_of_timestamp": as_of_timestamp,
        "source_signal_label": source_signal_label,
        "source_artifact": source_artifact,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    return digest.hexdigest()


def _source_artifact_label(
    *,
    ticker: str,
    analyst_family: str,
    signal_key: str | None,
    decision: bool,
) -> str:
    if decision:
        return f"{SIGNAL_CAPTURE_FILENAME}:{ticker}:decisions:{analyst_family}"
    if signal_key:
        return f"{SIGNAL_CAPTURE_FILENAME}:{ticker}:analyst_signals:{signal_key}"
    return f"{SIGNAL_CAPTURE_FILENAME}:{ticker}:analyst_signals:{analyst_family}:missing"


def _build_trading_foundation_trade_rows(records: list[SignalLedgerRecord]) -> list[dict[str, Any]]:
    trade_rows: list[dict[str, Any]] = []
    for record in records:
        if not record.is_backtest_eligible:
            continue
        if record.signal_score is None or record.signal_score == 0:
            continue
        entry_date = record.as_of_timestamp or record.decision_date
        if not entry_date:
            continue
        trade_rows.append(
            {
                "symbol": record.ticker,
                "entry_date": entry_date,
                "side": "LONG" if record.signal_score > 0 else "SHORT",
                "strategy": _strategy_name(record),
                "confidence_score": record.confidence_score,
                "run_id": record.run_id,
                "decision_id": record.decision_id,
                "ticker": record.ticker,
                "analyst_name": record.analyst_name,
                "analyst_family": record.analyst_family,
                "analyst_preset": record.analyst_preset,
                "signal": record.signal,
                "source_signal_label": record.source_signal_label,
            }
        )
    return trade_rows


def _strategy_name(record: SignalLedgerRecord) -> str:
    preset = str(record.analyst_preset or "default").strip().replace(" ", "_")
    family = record.analyst_family.strip().replace(" ", "_")
    return f"aihf_{family}_{preset}"


def _write_signal_ledger_csv(path: Path, records: list[SignalLedgerRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIGNAL_LEDGER_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_ready_row(asdict(record)))


def _write_trading_foundation_trades_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRADING_FOUNDATION_TRADE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_ready_row(row))


def _csv_ready_row(row: dict[str, Any]) -> dict[str, str]:
    return {key: _csv_cell(value) for key, value in row.items()}


def _csv_cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _count_by(records: list[SignalLedgerRecord], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        raw_value = getattr(record, field_name)
        key = "null" if raw_value is None else str(raw_value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _date_or_none(value: Any) -> str | None:
    text = _string_or_none(value)
    if not text:
        return None
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return text


def _numeric_or_none(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric
