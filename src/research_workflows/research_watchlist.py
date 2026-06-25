from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    DECISION_SUMMARY_DISCLAIMER,
    ResearchWatchlistArtifacts,
    _as_dict_of_strings,
    _as_list_of_strings,
    _default_research_watchlist_json_path,
    _summarize_mapping,
    _write_json,
    _watchlist_output_paths,
)
from .research_journal import _parsed_research_journal_rows


WATCHLIST_SCORING_RULES = [
    {"rule": "Latest technical-only action is buy", "points": 2},
    {"rule": "Latest core action is buy", "points": 2},
    {"rule": "Latest technical-only consensus is bullish", "points": 1},
    {"rule": "Latest core consensus is bullish", "points": 1},
    {"rule": "Latest all action is sell or short", "points": -3},
    {"rule": "Latest no-news action is sell or short", "points": -2},
    {"rule": "Latest all consensus is bearish", "points": -2},
    {"rule": "Latest no-news consensus is bearish", "points": -1},
    {"rule": "Disagreement across preset actions", "points": -1},
    {"rule": "Comparison notes present", "points": -1},
    {"rule": "Offline demo / needs validation", "points": -1},
    {"rule": "Only one journal entry", "points": -2},
]


def _short_text(value: Any, *, limit: int = 80) -> str:
    text = _summarize_mapping(value) if isinstance(value, dict) else "; ".join(_as_list_of_strings(value)) or str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _is_buy_like(value: str) -> bool:
    return value.strip().lower() == "buy"


def _is_bearish_action(value: str) -> bool:
    return value.strip().lower() in {"sell", "short"}


def _has_disagreement(action_by_preset: dict[str, str], comparison_notes: Any) -> tuple[bool, list[str]]:
    flags: list[str] = []
    unique_actions = {value.strip().lower() for value in action_by_preset.values() if value.strip()}
    if len(unique_actions) > 1:
        flags.append("preset action conflict")
    if _as_list_of_strings(comparison_notes):
        flags.append("comparison notes present")
    return (len(flags) > 0, flags)


def _watchlist_category(score: int, *, disagreement: bool, needs_validation: bool, entry_count: int, bearish_flag: bool) -> str:
    if entry_count <= 1:
        return "Insufficient History / Needs More Runs"
    if bearish_flag and score <= -2:
        return "Bearish / Avoid For Now Candidates"
    if disagreement or needs_validation or score <= 1:
        return "Mixed / Disagreement Candidates"
    if score >= 2:
        return "Strong Follow-Up Candidates"
    return "Mixed / Disagreement Candidates"


def _load_watchlist_payload(
    *,
    watchlist_path: Path | str | None = None,
) -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    if watchlist_path is None:
        default_path = _default_research_watchlist_json_path()
        if not default_path.exists():
            return None, None, ["Research watchlist JSON not found; generating checklist from journal only."]
        target_path = default_path
    else:
        target_path = Path(watchlist_path)
        if not target_path.exists():
            return target_path, None, [f"Research watchlist JSON not found: {target_path.as_posix()}; generating checklist from journal only."]

    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return target_path, None, [f"Malformed watchlist JSON at {target_path.as_posix()}; generating checklist from journal only."]
    if not isinstance(payload, dict):
        return target_path, None, [f"Unexpected watchlist JSON structure at {target_path.as_posix()}; generating checklist from journal only."]
    return target_path, payload, []


def build_research_watchlist(
    *,
    journal_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> ResearchWatchlistArtifacts:
    target_journal_path, parsed_rows, warnings = _parsed_research_journal_rows(journal_path=journal_path)
    markdown_path, json_path = _watchlist_output_paths(output_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in parsed_rows:
        grouped.setdefault(str(row.get("ticker") or "UNKNOWN"), []).append(row)

    generated_at = datetime.now().isoformat(timespec="seconds")
    scored_tickers: list[dict[str, Any]] = []

    for ticker_key in sorted(grouped):
        ticker_rows = grouped[ticker_key]
        latest = ticker_rows[-1]
        actions = _as_dict_of_strings(latest.get("action_by_preset"))
        consensus = _as_dict_of_strings(latest.get("consensus_by_preset"))
        confidence = _as_dict_of_strings(latest.get("confidence_by_preset"))
        comparison_notes = latest.get("comparison_notes")
        disagreement, disagreement_flags = _has_disagreement(actions, comparison_notes)
        needs_validation = str(latest.get("data_mode") or "").strip().lower() == "offline_demo"
        if str(latest.get("offline_demo_data") or "").strip().lower() == "true":
            needs_validation = True

        score = 0
        score_reasons: list[str] = []

        if _is_buy_like(actions.get("technical-only", "")):
            score += 2
            score_reasons.append("technical-only action buy (+2)")
        if _is_buy_like(actions.get("core", "")):
            score += 2
            score_reasons.append("core action buy (+2)")
        if consensus.get("technical-only", "").lower() == "bullish":
            score += 1
            score_reasons.append("technical-only consensus bullish (+1)")
        if consensus.get("core", "").lower() == "bullish":
            score += 1
            score_reasons.append("core consensus bullish (+1)")
        if _is_bearish_action(actions.get("all", "")):
            score -= 3
            score_reasons.append("all action bearish (-3)")
        if _is_bearish_action(actions.get("no-news", "")):
            score -= 2
            score_reasons.append("no-news action bearish (-2)")
        if consensus.get("all", "").lower() == "bearish":
            score -= 2
            score_reasons.append("all consensus bearish (-2)")
        if consensus.get("no-news", "").lower() == "bearish":
            score -= 1
            score_reasons.append("no-news consensus bearish (-1)")
        if disagreement:
            score -= 1
            score_reasons.append("disagreement flag (-1)")
        if needs_validation:
            score -= 1
            score_reasons.append("offline demo / needs validation (-1)")
        if len(ticker_rows) == 1:
            score -= 2
            score_reasons.append("single journal entry (-2)")

        bearish_flag = (
            _is_bearish_action(actions.get("all", ""))
            or _is_bearish_action(actions.get("no-news", ""))
            or consensus.get("all", "").lower() == "bearish"
            or consensus.get("no-news", "").lower() == "bearish"
        )
        category = _watchlist_category(
            score,
            disagreement=disagreement,
            needs_validation=needs_validation,
            entry_count=len(ticker_rows),
            bearish_flag=bearish_flag,
        )
        scored_tickers.append(
            {
                "ticker": ticker_key,
                "latest_generated_at": str(latest.get("generated_at") or ""),
                "score": score,
                "category": category,
                "entry_count": len(ticker_rows),
                "latest_actions": actions,
                "latest_consensus": consensus,
                "latest_confidence": confidence,
                "disagreement_flags": disagreement_flags + (["needs validation"] if needs_validation else []),
                "latest_bull_case_summary": _short_text(latest.get("bull_case")),
                "latest_bear_case_summary": _short_text(latest.get("bear_case")),
                "latest_packet_path": str(latest.get("research_packet_md_path") or ""),
                "score_reasons": score_reasons,
            }
        )

    categories = [
        "Strong Follow-Up Candidates",
        "Mixed / Disagreement Candidates",
        "Bearish / Avoid For Now Candidates",
        "Insufficient History / Needs More Runs",
    ]
    categorized: dict[str, list[dict[str, Any]]] = {category: [] for category in categories}
    for item in sorted(scored_tickers, key=lambda entry: (-entry["score"], entry["ticker"])):
        categorized[item["category"]].append(item)

    lines = [
        "# Research Watchlist",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
        f"- Source journal path: `{target_journal_path.as_posix()}`",
        f"- Generated timestamp: `{generated_at}`",
        f"- Total journal rows reviewed: `{len(parsed_rows)}`",
        f"- Tickers reviewed: `{len(grouped)}`",
        "- Scoring heuristic: Reward recent bullish technical-only/core signals, penalize bearish all/no-news signals, flag disagreement or comparison-note friction, mark offline-demo rows as needs-validation, and reduce confidence when only one journal entry exists.",
    ]
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")

    for category in categories:
        lines.extend(
            [
                "",
                f"## {category}",
                "",
                "| Ticker | Latest Generated At | Score | Category | Latest Actions | Latest Consensus | Disagreement Flags | Latest Bull Case | Latest Bear Case | Latest Packet Path |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        items = categorized[category]
        if not items:
            lines.append("| - | - | - | - | - | - | - | - | - | - |")
            continue
        for item in items:
            lines.append(
                "| "
                + " | ".join(
                    [
                        item["ticker"],
                        item["latest_generated_at"] or "-",
                        str(item["score"]),
                        item["category"],
                        _summarize_mapping(item["latest_actions"]).replace("|", "/"),
                        _summarize_mapping(item["latest_consensus"]).replace("|", "/"),
                        (_short_text(item["disagreement_flags"]) if item["disagreement_flags"] else "-").replace("|", "/"),
                        str(item["latest_bull_case_summary"] or "-").replace("|", "/"),
                        str(item["latest_bear_case_summary"] or "-").replace("|", "/"),
                        str(item["latest_packet_path"] or "-").replace("|", "/"),
                    ]
                )
                + " |"
            )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_payload = {
        "generated_at": generated_at,
        "source_journal_path": target_journal_path.as_posix(),
        "rows_reviewed": len(parsed_rows),
        "ticker_count": len(grouped),
        "scoring_rules": WATCHLIST_SCORING_RULES,
        "per_ticker": scored_tickers,
        "categories": {category: [item["ticker"] for item in categorized[category]] for category in categories},
        "warnings": warnings,
    }
    _write_json(json_path, json_payload)

    return ResearchWatchlistArtifacts(
        journal_path=target_journal_path,
        markdown_path=markdown_path,
        json_path=json_path,
        rows_reviewed=len(parsed_rows),
        ticker_count=len(grouped),
        warnings=warnings,
    )
