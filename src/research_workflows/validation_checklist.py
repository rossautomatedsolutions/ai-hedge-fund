from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    DECISION_SUMMARY_DISCLAIMER,
    ValidationChecklistArtifacts,
    _as_dict_of_strings,
    _as_list_of_strings,
    _default_validation_checklist_json_path,
    _summarize_mapping,
    _summarize_notes,
    _validation_output_paths,
    _write_json,
)
from .research_journal import _parsed_research_journal_rows
from .research_watchlist import _has_disagreement, _load_watchlist_payload


VALIDATION_CHECKLIST_SECTIONS = [
    (
        "Current Price / Chart Validation",
        [
            "Confirm current price and date of quote.",
            "Check whether the latest move is extended vs recent average range.",
            "Review recent support/resistance or breakout/breakdown context.",
        ],
    ),
    (
        "Volume / Liquidity Validation",
        [
            "Review latest volume relative to recent average volume.",
            "Check spread, liquidity, and whether position size would be realistic.",
            "Note whether trading conditions look abnormal today.",
        ],
    ),
    (
        "Recent News and Catalyst Check",
        [
            "Check latest company news and press releases.",
            "Check for sector, macro, regulatory, or idiosyncratic catalysts.",
            "Confirm whether the original research thesis is stale or still relevant.",
        ],
    ),
    (
        "SEC Filings / Earnings Check",
        [
            "Check latest 10-Q/10-K/8-K or earnings release.",
            "Confirm next earnings date and any upcoming investor events.",
            "Look for guidance changes, financing updates, or material disclosures.",
        ],
    ),
    (
        "Business Fundamentals Check",
        [
            "Review revenue, margin, cash-flow, and balance-sheet quality at a high level.",
            "Confirm whether the current setup aligns with the company fundamentals.",
            "Write down the main fundamental reason the idea could still work or fail.",
        ],
    ),
    (
        "Risk / Position Sizing Sanity Check",
        [
            "List the top risks that could invalidate or overwhelm the setup.",
            "Check whether volatility/liquidity would force a smaller size or no action.",
            "Decide whether this belongs on watchlist only, paper trade only, or deeper work.",
        ],
    ),
    (
        "Thesis / Invalidation Criteria",
        [
            "Write the bull thesis in one paragraph.",
            "Write the bear thesis in one paragraph.",
            "Define what would invalidate the thesis.",
        ],
    ),
    (
        "Final Human Review",
        [
            "Confirm this review used current data rather than only historical journal artifacts.",
            "Decide whether this remains watchlist-only, reject, or candidate for deeper work.",
            "Document any unanswered questions before considering the ticker actionable research.",
        ],
    ),
]


def _load_validation_checklist_payload(
    *,
    ticker: str,
    validation_checklist_path: Path | str | None = None,
) -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    if validation_checklist_path is None:
        default_path = _default_validation_checklist_json_path(ticker)
        if not default_path.exists():
            return None, None, ["Validation checklist JSON not found; using available journal/watchlist context only."]
        target_path = default_path
    else:
        target_path = Path(validation_checklist_path)
        if not target_path.exists():
            return target_path, None, [f"Validation checklist JSON not found: {target_path.as_posix()}; using available journal/watchlist context only."]

    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return target_path, None, [f"Malformed validation checklist JSON at {target_path.as_posix()}; using available journal/watchlist context only."]
    if not isinstance(payload, dict):
        return target_path, None, [f"Unexpected validation checklist JSON structure at {target_path.as_posix()}; using available journal/watchlist context only."]
    return target_path, payload, []


def build_validation_checklist(
    *,
    ticker: str | None,
    journal_path: Path | str | None = None,
    watchlist_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> ValidationChecklistArtifacts:
    ticker_symbol = str(ticker or "").strip().upper()
    if not ticker_symbol:
        raise SystemExit("Validation checklist mode requires --ticker.")

    target_journal_path, parsed_rows, warnings = _parsed_research_journal_rows(journal_path=journal_path, ticker=ticker_symbol)
    if not parsed_rows:
        raise SystemExit(f"Ticker `{ticker_symbol}` was not found in research journal: {target_journal_path.as_posix()}")

    latest_row = parsed_rows[-1]
    watchlist_used_path, watchlist_payload, watchlist_warnings = _load_watchlist_payload(watchlist_path=watchlist_path)
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
            warnings.append("Watchlist JSON missing `per_ticker` list; continuing with journal only.")

    latest_actions = _as_dict_of_strings(latest_row.get("action_by_preset"))
    latest_consensus = _as_dict_of_strings(latest_row.get("consensus_by_preset"))
    latest_comparison_notes = _as_list_of_strings(latest_row.get("comparison_notes"))
    disagreement_flags: list[str] = []
    _, derived_flags = _has_disagreement(latest_actions, latest_row.get("comparison_notes"))
    disagreement_flags.extend(derived_flags)
    if watchlist_entry and isinstance(watchlist_entry.get("disagreement_flags"), list):
        disagreement_flags.extend([str(item).strip() for item in watchlist_entry["disagreement_flags"] if str(item).strip()])
    if str(latest_row.get("data_mode") or "").strip().lower() == "offline_demo" or str(latest_row.get("offline_demo_data") or "").strip().lower() == "true":
        if "needs current-data validation" not in disagreement_flags:
            disagreement_flags.append("needs current-data validation")
        warnings.append("Latest journal entry used offline_demo data; current-data validation is required.")

    deduped_flags: list[str] = []
    seen_flags: set[str] = set()
    for flag in disagreement_flags:
        normalized = flag.lower()
        if normalized in seen_flags:
            continue
        seen_flags.add(normalized)
        deduped_flags.append(flag)

    markdown_path, json_path = _validation_output_paths(ticker=ticker_symbol, output_path=output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")

    latest_watchlist_category = str((watchlist_entry or {}).get("category") or "")
    latest_watchlist_score = (watchlist_entry or {}).get("score")
    latest_research_packet_path = str(latest_row.get("research_packet_md_path") or "")

    lines = [
        f"# Validation Checklist: {ticker_symbol}",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
        f"- Ticker: `{ticker_symbol}`",
        f"- Generated timestamp: `{generated_at}`",
        f"- Source journal path: `{target_journal_path.as_posix()}`",
        f"- Source watchlist path: `{watchlist_used_path.as_posix() if watchlist_used_path else 'N/A'}`",
        f"- Latest research packet path: `{latest_research_packet_path or 'N/A'}`",
        f"- Latest watchlist category: `{latest_watchlist_category or 'N/A'}`",
        f"- Latest watchlist score: `{latest_watchlist_score if latest_watchlist_score is not None else 'N/A'}`",
        f"- Latest actions by preset: `{_summarize_mapping(latest_actions)}`",
        f"- Latest consensus by preset: `{_summarize_mapping(latest_consensus)}`",
        f"- Comparison/disagreement flags: `{'; '.join(deduped_flags) if deduped_flags else 'None noted'}`",
        f"- Latest comparison notes: `{_summarize_notes(latest_comparison_notes)}`",
    ]
    if str(latest_row.get("data_mode") or "").strip().lower() == "offline_demo":
        lines.append("- Data warning: `Latest journal entry used offline_demo data. Validate all items against current market and company information.`")
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    for section_title, items in VALIDATION_CHECKLIST_SECTIONS:
        lines.extend(["", f"## {section_title}", ""])
        for item in items:
            lines.append(f"- [ ] {item}")

    lines.extend(
        [
            "",
            "## Final Status",
            "",
            "- Human status: Watchlist / Reject / Deep Research / Paper Trade Candidate / Trade Candidate",
            "- Reviewed by:",
            "- Reviewed at:",
            "- Notes:",
        ]
    )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_payload = {
        "generated_at": generated_at,
        "ticker": ticker_symbol,
        "source_journal_path": target_journal_path.as_posix(),
        "source_watchlist_path": watchlist_used_path.as_posix() if watchlist_used_path else None,
        "latest_research_packet_path": latest_research_packet_path,
        "latest_watchlist_category": latest_watchlist_category or None,
        "latest_watchlist_score": latest_watchlist_score,
        "latest_actions": latest_actions,
        "latest_consensus": latest_consensus,
        "disagreement_flags": deduped_flags,
        "checklist_sections": [{"title": title, "items": items} for title, items in VALIDATION_CHECKLIST_SECTIONS],
        "warnings": warnings,
    }
    _write_json(json_path, json_payload)

    return ValidationChecklistArtifacts(
        journal_path=target_journal_path,
        watchlist_path=watchlist_used_path,
        markdown_path=markdown_path,
        json_path=json_path,
        ticker=ticker_symbol,
        warnings=warnings,
    )
