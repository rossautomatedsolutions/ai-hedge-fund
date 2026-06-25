from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    DECISION_SUMMARY_DISCLAIMER,
    HUMAN_REVIEW_LOG_FIELDNAMES,
    HUMAN_REVIEW_STATUSES,
    HumanReviewLogArtifacts,
    HumanReviewSummaryArtifacts,
    _as_dict_of_strings,
    _as_list_of_strings,
    _default_human_review_log_path,
    _human_review_summary_output_paths,
    _json_cell,
    _parse_json_cell,
    _safe_generated_sort_key,
    _summarize_mapping,
    _summarize_notes,
    _write_json,
)
from .research_journal import _parsed_research_journal_rows
from .research_watchlist import _has_disagreement, _load_watchlist_payload
from .validation_checklist import _load_validation_checklist_payload


def record_human_review(
    *,
    ticker: str | None,
    human_status: str | None,
    review_notes: str | None = None,
    validation_checklist_path: Path | str | None = None,
    journal_path: Path | str | None = None,
    watchlist_path: Path | str | None = None,
    human_review_log_path: Path | str | None = None,
) -> HumanReviewLogArtifacts:
    ticker_symbol = str(ticker or "").strip().upper()
    if not ticker_symbol:
        raise SystemExit("Human review logging mode requires --ticker.")

    status_value = str(human_status or "").strip()
    if not status_value:
        raise SystemExit("Human review logging mode requires --human-status.")
    if status_value not in HUMAN_REVIEW_STATUSES:
        raise SystemExit("Invalid --human-status. Allowed values: " + ", ".join(HUMAN_REVIEW_STATUSES))

    warnings: list[str] = []
    validation_path_used, validation_payload, validation_warnings = _load_validation_checklist_payload(
        ticker=ticker_symbol,
        validation_checklist_path=validation_checklist_path,
    )
    warnings.extend(validation_warnings)

    journal_path_used: Path | None = None
    journal_rows: list[dict[str, Any]] = []
    if journal_path is None:
        default_journal = Path("outputs") / "research_journal.csv"
        if default_journal.exists():
            journal_path_used, journal_rows, journal_warnings = _parsed_research_journal_rows(
                journal_path=default_journal,
                ticker=ticker_symbol,
            )
            warnings.extend(journal_warnings)
        else:
            warnings.append("Research journal CSV not found; journal-derived fields will be left blank.")
    else:
        explicit_journal = Path(journal_path)
        if explicit_journal.exists():
            journal_path_used, journal_rows, journal_warnings = _parsed_research_journal_rows(
                journal_path=explicit_journal,
                ticker=ticker_symbol,
            )
            warnings.extend(journal_warnings)
        else:
            journal_path_used = explicit_journal
            warnings.append(f"Research journal CSV not found: {explicit_journal.as_posix()}; journal-derived fields will be left blank.")

    latest_journal_row = journal_rows[-1] if journal_rows else None

    watchlist_path_used, watchlist_payload, watchlist_warnings = _load_watchlist_payload(watchlist_path=watchlist_path)
    warnings.extend(watchlist_warnings)
    watchlist_entry: dict[str, Any] | None = None
    if isinstance(watchlist_payload, dict):
        per_ticker = watchlist_payload.get("per_ticker")
        if isinstance(per_ticker, list):
            for item in per_ticker:
                if isinstance(item, dict) and str(item.get("ticker") or "").strip().upper() == ticker_symbol:
                    watchlist_entry = item
                    break
        else:
            warnings.append("Watchlist JSON missing `per_ticker` list; watchlist-derived fields will be left blank.")

    latest_actions = _as_dict_of_strings((validation_payload or {}).get("latest_actions"))
    if not latest_actions and latest_journal_row:
        latest_actions = _as_dict_of_strings(latest_journal_row.get("action_by_preset"))

    latest_consensus = _as_dict_of_strings((validation_payload or {}).get("latest_consensus"))
    if not latest_consensus and latest_journal_row:
        latest_consensus = _as_dict_of_strings(latest_journal_row.get("consensus_by_preset"))

    disagreement_flags = _as_list_of_strings((validation_payload or {}).get("disagreement_flags"))
    if not disagreement_flags and watchlist_entry and isinstance(watchlist_entry.get("disagreement_flags"), list):
        disagreement_flags = [str(item).strip() for item in watchlist_entry["disagreement_flags"] if str(item).strip()]
    if not disagreement_flags and latest_journal_row:
        _, derived_flags = _has_disagreement(latest_actions, latest_journal_row.get("comparison_notes"))
        disagreement_flags = derived_flags

    latest_research_packet_path = str((validation_payload or {}).get("latest_research_packet_path") or "")
    if not latest_research_packet_path and latest_journal_row:
        latest_research_packet_path = str(latest_journal_row.get("research_packet_md_path") or "")

    latest_watchlist_category = str((validation_payload or {}).get("latest_watchlist_category") or "")
    if not latest_watchlist_category:
        latest_watchlist_category = str((watchlist_entry or {}).get("category") or "")

    latest_watchlist_score = (validation_payload or {}).get("latest_watchlist_score")
    if latest_watchlist_score in (None, ""):
        latest_watchlist_score = (watchlist_entry or {}).get("score")

    data_mode = str((latest_journal_row or {}).get("data_mode") or "")
    offline_demo_warning = ""
    if str((latest_journal_row or {}).get("offline_demo_data") or "").strip().lower() == "true" or data_mode.strip().lower() == "offline_demo":
        offline_demo_warning = "Latest journal context reflects offline_demo data; current-data validation is still required."

    row = {
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker_symbol,
        "human_status": status_value,
        "review_notes": str(review_notes or "").strip(),
        "validation_checklist_path": validation_path_used.as_posix() if validation_path_used else "",
        "latest_research_packet_path": latest_research_packet_path,
        "latest_watchlist_category": latest_watchlist_category,
        "latest_watchlist_score": "" if latest_watchlist_score in (None, "") else str(latest_watchlist_score),
        "latest_actions": _json_cell(latest_actions or {}),
        "latest_consensus": _json_cell(latest_consensus or {}),
        "disagreement_flags": _json_cell(disagreement_flags or []),
        "data_mode": data_mode,
        "offline_demo_warning": offline_demo_warning,
        "source_journal_path": journal_path_used.as_posix() if journal_path_used else "",
        "source_watchlist_path": watchlist_path_used.as_posix() if watchlist_path_used else "",
    }

    target_log_path = Path(human_review_log_path) if human_review_log_path else _default_human_review_log_path()
    target_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target_log_path.exists() or target_log_path.stat().st_size == 0
    with target_log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_REVIEW_LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return HumanReviewLogArtifacts(
        log_path=target_log_path,
        rows_written=1,
        ticker=ticker_symbol,
        human_status=status_value,
        warnings=warnings,
    )


def _parsed_human_review_rows(
    *,
    log_path: Path | str | None = None,
    ticker: str | None = None,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    target_log_path = Path(log_path) if log_path else _default_human_review_log_path()
    if not target_log_path.exists():
        raise SystemExit(f"Human review log file not found: {target_log_path.as_posix()}")

    warnings: list[str] = []
    parsed_rows: list[dict[str, Any]] = []
    ticker_filter = str(ticker or "").strip().upper() or None

    with target_log_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row_ticker = str(raw_row.get("ticker") or "").strip().upper()
            if ticker_filter and row_ticker != ticker_filter:
                continue
            reviewed_at = str(raw_row.get("reviewed_at") or "").strip()
            parsed_rows.append(
                {
                    "reviewed_at": reviewed_at,
                    "ticker": row_ticker,
                    "human_status": str(raw_row.get("human_status") or "").strip(),
                    "review_notes": str(raw_row.get("review_notes") or "").strip(),
                    "validation_checklist_path": str(raw_row.get("validation_checklist_path") or "").strip(),
                    "latest_research_packet_path": str(raw_row.get("latest_research_packet_path") or "").strip(),
                    "latest_watchlist_category": str(raw_row.get("latest_watchlist_category") or "").strip(),
                    "latest_watchlist_score": str(raw_row.get("latest_watchlist_score") or "").strip(),
                    "latest_actions": _parse_json_cell(
                        str(raw_row.get("latest_actions") or ""),
                        cell_name="latest_actions",
                        ticker=row_ticker,
                        generated_at=reviewed_at,
                        warnings=warnings,
                    ),
                    "latest_consensus": _parse_json_cell(
                        str(raw_row.get("latest_consensus") or ""),
                        cell_name="latest_consensus",
                        ticker=row_ticker,
                        generated_at=reviewed_at,
                        warnings=warnings,
                    ),
                    "disagreement_flags": _parse_json_cell(
                        str(raw_row.get("disagreement_flags") or ""),
                        cell_name="disagreement_flags",
                        ticker=row_ticker,
                        generated_at=reviewed_at,
                        warnings=warnings,
                    ),
                    "data_mode": str(raw_row.get("data_mode") or "").strip(),
                    "offline_demo_warning": str(raw_row.get("offline_demo_warning") or "").strip(),
                    "source_journal_path": str(raw_row.get("source_journal_path") or "").strip(),
                    "source_watchlist_path": str(raw_row.get("source_watchlist_path") or "").strip(),
                }
            )

    parsed_rows.sort(key=lambda row: (_safe_generated_sort_key(str(row.get("reviewed_at") or "")), str(row.get("ticker") or "")))
    return target_log_path, parsed_rows, warnings


def _short_text(value: Any, *, limit: int = 80) -> str:
    text = _summarize_notes(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def review_human_reviews(
    *,
    log_path: Path | str | None = None,
    ticker: str | None = None,
    output_path: Path | str | None = None,
) -> HumanReviewSummaryArtifacts:
    ticker_filter = str(ticker or "").strip().upper() or None
    target_log_path, parsed_rows, warnings = _parsed_human_review_rows(log_path=log_path, ticker=ticker_filter)
    markdown_path, json_path = _human_review_summary_output_paths(output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    latest_by_ticker: dict[str, dict[str, Any]] = {}
    for row in parsed_rows:
        latest_by_ticker[str(row.get("ticker") or "UNKNOWN")] = row

    status_counts = {status: 0 for status in HUMAN_REVIEW_STATUSES}
    for row in parsed_rows:
        status = str(row.get("human_status") or "").strip()
        if status in status_counts:
            status_counts[status] += 1

    latest_rows = sorted(latest_by_ticker.values(), key=lambda row: str(row.get("ticker") or ""))
    follow_up_buckets = {
        "Active Watchlist": [],
        "Rejected": [],
        "Deep Research Queue": [],
        "Paper Trade Candidates": [],
        "Trade Candidates": [],
    }
    status_to_bucket = {
        "Watchlist": "Active Watchlist",
        "Reject": "Rejected",
        "Deep Research": "Deep Research Queue",
        "Paper Trade Candidate": "Paper Trade Candidates",
        "Trade Candidate": "Trade Candidates",
    }
    for row in latest_rows:
        bucket = status_to_bucket.get(str(row.get("human_status") or "").strip())
        if bucket:
            follow_up_buckets[bucket].append(str(row.get("ticker") or ""))

    process_warnings: list[str] = []
    for row in parsed_rows:
        row_label = f"{row.get('ticker') or 'UNKNOWN'} @ {row.get('reviewed_at') or 'unknown time'}"
        if str(row.get("offline_demo_warning") or "").strip():
            process_warnings.append(f"{row_label}: offline_demo_warning present.")
        if not str(row.get("validation_checklist_path") or "").strip():
            process_warnings.append(f"{row_label}: validation_checklist_path missing.")
        if not str(row.get("latest_research_packet_path") or "").strip():
            process_warnings.append(f"{row_label}: latest_research_packet_path missing.")
    process_warnings.extend(warnings)

    scope_text = ticker_filter if ticker_filter else "All tickers"
    lines = [
        "# Human Review Summary",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
        f"- Source human review log path: `{target_log_path.as_posix()}`",
        f"- Generated timestamp: `{generated_at}`",
        f"- Scope: `{scope_text}`",
        f"- Total review rows: `{len(parsed_rows)}`",
        f"- Reviewed ticker count: `{len(latest_rows)}`",
        f"- Watchlist count: `{status_counts['Watchlist']}`",
        f"- Reject count: `{status_counts['Reject']}`",
        f"- Deep Research count: `{status_counts['Deep Research']}`",
        f"- Paper Trade Candidate count: `{status_counts['Paper Trade Candidate']}`",
        f"- Trade Candidate count: `{status_counts['Trade Candidate']}`",
        "",
        "## Latest Review Per Ticker",
        "",
        "| Ticker | Latest Reviewed At | Latest Human Status | Latest Review Notes | Latest Watchlist Category | Latest Watchlist Score | Latest Actions | Latest Consensus | Disagreement Flags | Validation Checklist Path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not latest_rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
    for row in latest_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("ticker") or "-"),
                    str(row.get("reviewed_at") or "-"),
                    str(row.get("human_status") or "-"),
                    _short_text(row.get("review_notes"), limit=60).replace("|", "/"),
                    str(row.get("latest_watchlist_category") or "-"),
                    str(row.get("latest_watchlist_score") or "-"),
                    _summarize_mapping(row.get("latest_actions")).replace("|", "/"),
                    _summarize_mapping(row.get("latest_consensus")).replace("|", "/"),
                    _summarize_notes(row.get("disagreement_flags")).replace("|", "/"),
                    str(row.get("validation_checklist_path") or "-").replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Chronological Review History",
            "",
            "| Reviewed At | Ticker | Human Status | Review Notes | Latest Watchlist Category | Latest Watchlist Score |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if not parsed_rows:
        lines.append("| - | - | - | - | - | - |")
    for row in parsed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("reviewed_at") or "-"),
                    str(row.get("ticker") or "-"),
                    str(row.get("human_status") or "-"),
                    _short_text(row.get("review_notes"), limit=70).replace("|", "/"),
                    str(row.get("latest_watchlist_category") or "-"),
                    str(row.get("latest_watchlist_score") or "-"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Follow-Up Buckets", ""])
    for bucket, tickers in follow_up_buckets.items():
        lines.append(f"- {bucket}: {', '.join(tickers) if tickers else 'None'}")

    lines.extend(["", "## Process Warnings", ""])
    if process_warnings:
        for warning in process_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_payload = {
        "generated_at": generated_at,
        "source_human_review_log_path": target_log_path.as_posix(),
        "scope": {"ticker": ticker_filter, "all_tickers": ticker_filter is None},
        "total_review_rows": len(parsed_rows),
        "reviewed_ticker_count": len(latest_rows),
        "counts_by_human_status": status_counts,
        "latest_review_per_ticker": [
            {
                "ticker": str(row.get("ticker") or ""),
                "latest_reviewed_at": str(row.get("reviewed_at") or ""),
                "latest_human_status": str(row.get("human_status") or ""),
                "latest_review_notes": str(row.get("review_notes") or ""),
                "latest_watchlist_category": str(row.get("latest_watchlist_category") or ""),
                "latest_watchlist_score": str(row.get("latest_watchlist_score") or ""),
                "latest_actions": row.get("latest_actions"),
                "latest_consensus": row.get("latest_consensus"),
                "disagreement_flags": row.get("disagreement_flags"),
                "validation_checklist_path": str(row.get("validation_checklist_path") or ""),
            }
            for row in latest_rows
        ],
        "chronological_review_history": [
            {
                "reviewed_at": str(row.get("reviewed_at") or ""),
                "ticker": str(row.get("ticker") or ""),
                "human_status": str(row.get("human_status") or ""),
                "review_notes": str(row.get("review_notes") or ""),
                "latest_watchlist_category": str(row.get("latest_watchlist_category") or ""),
                "latest_watchlist_score": str(row.get("latest_watchlist_score") or ""),
            }
            for row in parsed_rows
        ],
        "follow_up_buckets": follow_up_buckets,
        "process_warnings": process_warnings,
        "warnings": warnings,
    }
    _write_json(json_path, json_payload)

    return HumanReviewSummaryArtifacts(
        log_path=target_log_path,
        markdown_path=markdown_path,
        json_path=json_path,
        ticker=ticker_filter,
        rows_reviewed=len(parsed_rows),
        reviewed_ticker_count=len(latest_rows),
        warnings=warnings,
    )
