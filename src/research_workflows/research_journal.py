from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    DECISION_SUMMARY_DISCLAIMER,
    RESEARCH_JOURNAL_FIELDNAMES,
    ResearchJournalArtifacts,
    ResearchJournalReviewArtifacts,
    ResearchPacketArtifacts,
    _as_dict_of_strings,
    _as_list_of_strings,
    _default_research_journal_path,
    _display_theme_counts,
    _journal_review_output_paths,
    _json_cell,
    _parse_json_cell,
    _safe_generated_sort_key,
    _summarize_mapping,
    _summarize_notes,
    _theme_counts,
    _write_json,
)


def _research_journal_rows_from_payload(
    payload: dict[str, Any],
    *,
    research_packet_md_path: Path,
    research_packet_json_path: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ticker_packet in payload.get("tickers", []):
        rows.append(
            {
                "generated_at": str(payload.get("generated_at") or ""),
                "ticker": str(ticker_packet.get("ticker") or ""),
                "model": str(ticker_packet.get("model") or payload.get("model") or ""),
                "data_mode": str(ticker_packet.get("data_mode") or payload.get("data_mode") or ""),
                "offline_demo_data": str(bool(payload.get("offline_demo_data"))),
                "run_dir": str(payload.get("run_dir") or ""),
                "presets_analyzed": _json_cell(ticker_packet.get("presets_analyzed") or []),
                "action_by_preset": _json_cell(ticker_packet.get("action_by_preset") or {}),
                "confidence_by_preset": _json_cell(ticker_packet.get("confidence_by_preset") or {}),
                "consensus_by_preset": _json_cell(ticker_packet.get("consensus_by_preset") or {}),
                "comparison_notes": _json_cell(ticker_packet.get("comparison_notes") or []),
                "bull_case": str(ticker_packet.get("bull_case") or ""),
                "bear_case": str(ticker_packet.get("bear_case") or ""),
                "key_disagreement_points": _json_cell(ticker_packet.get("key_disagreement_points") or []),
                "data_limitations": _json_cell(ticker_packet.get("data_limitations") or []),
                "what_to_check_next_manually": _json_cell(ticker_packet.get("what_to_check_next_manually") or []),
                "notable_risks_or_reasons_not_to_act": _json_cell(
                    ticker_packet.get("notable_risks_or_reasons_not_to_act") or []
                ),
                "research_packet_md_path": research_packet_md_path.as_posix(),
                "research_packet_json_path": research_packet_json_path.as_posix(),
            }
        )
    return rows


def append_research_journal(
    artifacts: ResearchPacketArtifacts,
    *,
    journal_path: Path | str | None = None,
) -> ResearchJournalArtifacts:
    target_path = Path(journal_path) if journal_path else _default_research_journal_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _research_journal_rows_from_payload(
        artifacts.payload,
        research_packet_md_path=artifacts.markdown_path,
        research_packet_json_path=artifacts.json_path,
    )
    write_header = not target_path.exists() or target_path.stat().st_size == 0
    with target_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESEARCH_JOURNAL_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return ResearchJournalArtifacts(journal_path=target_path, rows_written=len(rows))


def _parsed_research_journal_rows(
    *,
    journal_path: Path | str | None = None,
    ticker: str | None = None,
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    target_journal_path = Path(journal_path) if journal_path else _default_research_journal_path()
    if not target_journal_path.exists():
        raise SystemExit(f"Research journal file not found: {target_journal_path.as_posix()}")

    warnings: list[str] = []
    parsed_rows: list[dict[str, Any]] = []
    ticker_filter = ticker.upper() if ticker else None

    with target_journal_path.open(encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row_ticker = str(raw_row.get("ticker") or "").strip().upper()
            if ticker_filter and row_ticker != ticker_filter:
                continue
            generated_at = str(raw_row.get("generated_at") or "").strip()
            parsed_rows.append(
                {
                    "generated_at": generated_at,
                    "ticker": row_ticker,
                    "model": str(raw_row.get("model") or "").strip(),
                    "data_mode": str(raw_row.get("data_mode") or "").strip(),
                    "offline_demo_data": str(raw_row.get("offline_demo_data") or "").strip(),
                    "run_dir": str(raw_row.get("run_dir") or "").strip(),
                    "presets_analyzed": _parse_json_cell(
                        str(raw_row.get("presets_analyzed") or ""),
                        cell_name="presets_analyzed",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "action_by_preset": _parse_json_cell(
                        str(raw_row.get("action_by_preset") or ""),
                        cell_name="action_by_preset",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "confidence_by_preset": _parse_json_cell(
                        str(raw_row.get("confidence_by_preset") or ""),
                        cell_name="confidence_by_preset",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "consensus_by_preset": _parse_json_cell(
                        str(raw_row.get("consensus_by_preset") or ""),
                        cell_name="consensus_by_preset",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "comparison_notes": _parse_json_cell(
                        str(raw_row.get("comparison_notes") or ""),
                        cell_name="comparison_notes",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "bull_case": str(raw_row.get("bull_case") or "").strip(),
                    "bear_case": str(raw_row.get("bear_case") or "").strip(),
                    "key_disagreement_points": _parse_json_cell(
                        str(raw_row.get("key_disagreement_points") or ""),
                        cell_name="key_disagreement_points",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "data_limitations": _parse_json_cell(
                        str(raw_row.get("data_limitations") or ""),
                        cell_name="data_limitations",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "what_to_check_next_manually": _parse_json_cell(
                        str(raw_row.get("what_to_check_next_manually") or ""),
                        cell_name="what_to_check_next_manually",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "notable_risks_or_reasons_not_to_act": _parse_json_cell(
                        str(raw_row.get("notable_risks_or_reasons_not_to_act") or ""),
                        cell_name="notable_risks_or_reasons_not_to_act",
                        ticker=row_ticker,
                        generated_at=generated_at,
                        warnings=warnings,
                    ),
                    "research_packet_md_path": str(raw_row.get("research_packet_md_path") or "").strip(),
                    "research_packet_json_path": str(raw_row.get("research_packet_json_path") or "").strip(),
                }
            )

    parsed_rows.sort(key=lambda row: (_safe_generated_sort_key(str(row.get("generated_at") or "")), str(row.get("ticker") or "")))
    return target_journal_path, parsed_rows, warnings


def review_research_journal(
    *,
    journal_path: Path | str | None = None,
    ticker: str | None = None,
    output_path: Path | str | None = None,
) -> ResearchJournalReviewArtifacts:
    ticker_filter = ticker.upper() if ticker else None
    target_journal_path, parsed_rows, warnings = _parsed_research_journal_rows(journal_path=journal_path, ticker=ticker_filter)
    markdown_path, json_path = _journal_review_output_paths(ticker=ticker_filter, output_path=output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    latest_entry = parsed_rows[-1] if parsed_rows else None
    scope_text = ticker_filter if ticker_filter else "All tickers"
    generated_at = datetime.now().isoformat(timespec="seconds")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in parsed_rows:
        grouped.setdefault(str(row.get("ticker") or "UNKNOWN"), []).append(row)

    lines = [
        "# Research Journal Review",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
        f"- Source journal path: `{target_journal_path.as_posix()}`",
        f"- Generated timestamp: `{generated_at}`",
        f"- Scope: `{scope_text}`",
        f"- Journal entries reviewed: `{len(parsed_rows)}`",
        f"- Latest run directory: `{str((latest_entry or {}).get('run_dir') or 'N/A')}`",
        f"- Latest research_packet.md path: `{str((latest_entry or {}).get('research_packet_md_path') or 'N/A')}`",
        f"- Latest research_packet.json path: `{str((latest_entry or {}).get('research_packet_json_path') or 'N/A')}`",
    ]
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    for ticker_key in sorted(grouped):
        ticker_rows = grouped[ticker_key]
        latest_ticker_row = ticker_rows[-1]
        lines.extend(
            [
                "",
                f"## {ticker_key}",
                "",
                "| Generated At | Model | Data Mode | Action By Preset | Consensus By Preset | Comparison Notes | Research Packet |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in ticker_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("generated_at") or "-"),
                        str(row.get("model") or "-"),
                        str(row.get("data_mode") or "-"),
                        _summarize_mapping(row.get("action_by_preset")).replace("|", "/"),
                        _summarize_mapping(row.get("consensus_by_preset")).replace("|", "/"),
                        _summarize_notes(row.get("comparison_notes")).replace("|", "/"),
                        str(row.get("research_packet_md_path") or "-").replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"- Latest action_by_preset: `{_summarize_mapping(latest_ticker_row.get('action_by_preset'))}`",
                f"- Latest confidence_by_preset: `{_summarize_mapping(latest_ticker_row.get('confidence_by_preset'))}`",
                f"- Latest consensus_by_preset: `{_summarize_mapping(latest_ticker_row.get('consensus_by_preset'))}`",
                f"- Repeated disagreement themes: {'; '.join(_display_theme_counts([row.get('key_disagreement_points') for row in ticker_rows]))}",
                f"- Recurring bull-case themes: {'; '.join(_display_theme_counts([row.get('bull_case') for row in ticker_rows]))}",
                f"- Recurring bear-case themes: {'; '.join(_display_theme_counts([row.get('bear_case') for row in ticker_rows]))}",
                f"- Recurring what to check next manually: {'; '.join(_display_theme_counts([row.get('what_to_check_next_manually') for row in ticker_rows]))}",
                f"- Latest notable risks / reasons not to act: `{_summarize_notes(latest_ticker_row.get('notable_risks_or_reasons_not_to_act'))}`",
            ]
        )

    if not parsed_rows:
        lines.extend(["", "## No Matching Entries", "", "No journal entries matched the requested scope."])

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_payload = {
        "generated_at": generated_at,
        "source_journal_path": target_journal_path.as_posix(),
        "scope": {"ticker": ticker_filter, "all_tickers": ticker_filter is None},
        "entries_reviewed": len(parsed_rows),
        "latest_run_dir": str((latest_entry or {}).get("run_dir") or ""),
        "latest_research_packet_md_path": str((latest_entry or {}).get("research_packet_md_path") or ""),
        "latest_research_packet_json_path": str((latest_entry or {}).get("research_packet_json_path") or ""),
        "warnings": warnings,
        "tickers": [],
    }
    for ticker_key in sorted(grouped):
        ticker_rows = grouped[ticker_key]
        latest_ticker_row = ticker_rows[-1]
        json_payload["tickers"].append(
            {
                "ticker": ticker_key,
                "entry_count": len(ticker_rows),
                "latest_action_by_preset": _as_dict_of_strings(latest_ticker_row.get("action_by_preset")),
                "latest_confidence_by_preset": _as_dict_of_strings(latest_ticker_row.get("confidence_by_preset")),
                "latest_consensus_by_preset": _as_dict_of_strings(latest_ticker_row.get("consensus_by_preset")),
                "repeated_disagreement_themes": _theme_counts([row.get("key_disagreement_points") for row in ticker_rows]),
                "recurring_bull_case_themes": _theme_counts([row.get("bull_case") for row in ticker_rows]),
                "recurring_bear_case_themes": _theme_counts([row.get("bear_case") for row in ticker_rows]),
                "recurring_manual_checks": _theme_counts([row.get("what_to_check_next_manually") for row in ticker_rows]),
                "latest_notable_risks_or_reasons_not_to_act": _as_list_of_strings(
                    latest_ticker_row.get("notable_risks_or_reasons_not_to_act")
                ),
                "entries": [
                    {
                        "generated_at": str(row.get("generated_at") or ""),
                        "model": str(row.get("model") or ""),
                        "data_mode": str(row.get("data_mode") or ""),
                        "action_by_preset": row.get("action_by_preset"),
                        "confidence_by_preset": row.get("confidence_by_preset"),
                        "consensus_by_preset": row.get("consensus_by_preset"),
                        "comparison_notes": row.get("comparison_notes"),
                        "research_packet_md_path": str(row.get("research_packet_md_path") or ""),
                        "research_packet_json_path": str(row.get("research_packet_json_path") or ""),
                    }
                    for row in ticker_rows
                ],
            }
        )
    _write_json(json_path, json_payload)

    return ResearchJournalReviewArtifacts(
        journal_path=target_journal_path,
        markdown_path=markdown_path,
        json_path=json_path,
        ticker=ticker_filter,
        entries_reviewed=len(parsed_rows),
        warnings=warnings,
    )
