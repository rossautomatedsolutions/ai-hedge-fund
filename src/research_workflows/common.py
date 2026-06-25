from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DECISION_SUMMARY_DISCLAIMER = (
    "Educational use only. This report is not financial advice, not an offer to buy or sell securities, "
    "and not a recommendation to take any trading action."
)
RESEARCH_JOURNAL_FIELDNAMES = [
    "generated_at",
    "ticker",
    "model",
    "data_mode",
    "offline_demo_data",
    "run_dir",
    "presets_analyzed",
    "action_by_preset",
    "confidence_by_preset",
    "consensus_by_preset",
    "comparison_notes",
    "bull_case",
    "bear_case",
    "key_disagreement_points",
    "data_limitations",
    "what_to_check_next_manually",
    "notable_risks_or_reasons_not_to_act",
    "research_packet_md_path",
    "research_packet_json_path",
]
HUMAN_REVIEW_STATUSES = [
    "Watchlist",
    "Reject",
    "Deep Research",
    "Paper Trade Candidate",
    "Trade Candidate",
]
HUMAN_REVIEW_LOG_FIELDNAMES = [
    "reviewed_at",
    "ticker",
    "human_status",
    "review_notes",
    "validation_checklist_path",
    "latest_research_packet_path",
    "latest_watchlist_category",
    "latest_watchlist_score",
    "latest_actions",
    "latest_consensus",
    "disagreement_flags",
    "data_mode",
    "offline_demo_warning",
    "source_journal_path",
    "source_watchlist_path",
]


@dataclass
class ResearchPacketArtifacts:
    markdown_path: Path
    json_path: Path
    payload: dict[str, Any]


@dataclass
class ResearchJournalArtifacts:
    journal_path: Path
    rows_written: int


@dataclass
class ResearchJournalReviewArtifacts:
    journal_path: Path
    markdown_path: Path
    json_path: Path
    ticker: str | None
    entries_reviewed: int
    warnings: list[str]


@dataclass
class ResearchWatchlistArtifacts:
    journal_path: Path
    markdown_path: Path
    json_path: Path
    rows_reviewed: int
    ticker_count: int
    warnings: list[str]


@dataclass
class ValidationChecklistArtifacts:
    journal_path: Path
    watchlist_path: Path | None
    markdown_path: Path
    json_path: Path
    ticker: str
    warnings: list[str]


@dataclass
class HumanReviewLogArtifacts:
    log_path: Path
    rows_written: int
    ticker: str
    human_status: str
    warnings: list[str]


@dataclass
class HumanReviewSummaryArtifacts:
    log_path: Path
    markdown_path: Path
    json_path: Path
    ticker: str | None
    rows_reviewed: int
    reviewed_ticker_count: int
    warnings: list[str]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _csv_row_list(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _default_research_journal_path() -> Path:
    return Path("outputs") / "research_journal.csv"


def _default_research_journal_review_markdown_path(ticker: str | None = None) -> Path:
    if ticker:
        return Path("outputs") / f"research_journal_review_{ticker.upper()}.md"
    return Path("outputs") / "research_journal_review.md"


def _default_research_watchlist_markdown_path() -> Path:
    return Path("outputs") / "research_watchlist.md"


def _default_research_watchlist_json_path() -> Path:
    return Path("outputs") / "research_watchlist.json"


def _default_validation_checklist_markdown_path(ticker: str) -> Path:
    return Path("outputs") / f"validation_checklist_{ticker.upper()}.md"


def _default_validation_checklist_json_path(ticker: str) -> Path:
    return Path("outputs") / f"validation_checklist_{ticker.upper()}.json"


def _default_human_review_log_path() -> Path:
    return Path("outputs") / "human_review_log.csv"


def _default_human_review_summary_markdown_path() -> Path:
    return Path("outputs") / "human_review_summary.md"


def _parse_json_cell(value: str, *, cell_name: str, ticker: str, generated_at: str, warnings: list[str]) -> Any:
    raw_value = (value or "").strip()
    if not raw_value:
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        warnings.append(
            f"Malformed JSON in `{cell_name}` for `{ticker or 'UNKNOWN'}` at `{generated_at or 'unknown time'}`; raw value shown."
        )
        return {"_raw": value, "_malformed": True}


def _as_list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [f"{key}: {val}" for key, val in value.items() if str(key).strip() or str(val).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_dict_of_strings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        item_text = str(item).strip()
        if key_text:
            normalized[key_text] = item_text
    return normalized


def _normalize_theme_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).replace("\r", "\n")
        for separator in (";", "|"):
            text = text.replace(separator, "\n")
        items = [part for line in text.split("\n") for part in line.split(" - ")]
    normalized: list[str] = []
    for item in items:
        text = str(item).strip(" -\t")
        if text:
            normalized.append(text)
    return normalized


def _summarize_mapping(value: Any) -> str:
    mapping = _as_dict_of_strings(value)
    if mapping:
        return ", ".join(f"{key}: {val}" for key, val in sorted(mapping.items()))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    text = str(value or "").strip()
    return text or "-"


def _summarize_notes(value: Any) -> str:
    items = _as_list_of_strings(value)
    if items:
        return "; ".join(items)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    text = str(value or "").strip()
    return text or "-"


def _theme_counts(values: list[Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for idx, value in enumerate(values):
        seen_in_entry: set[str] = set()
        for item in _normalize_theme_items(value):
            normalized = item.lower()
            if normalized in seen_in_entry:
                continue
            seen_in_entry.add(normalized)
            counts[normalized] = counts.get(normalized, 0) + 1
            first_seen.setdefault(normalized, idx)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))
    return [{"theme": theme, "count": count} for theme, count in ordered]


def _display_theme_counts(values: list[Any]) -> list[str]:
    themes = _theme_counts(values)
    if not themes:
        return ["None noted."]
    return [f"{entry['theme']} ({entry['count']}x)" for entry in themes]


def _safe_generated_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat())
    except ValueError:
        return (1, value)


def _journal_review_output_paths(
    *,
    ticker: str | None,
    output_path: Path | str | None,
) -> tuple[Path, Path]:
    markdown_path = Path(output_path) if output_path else _default_research_journal_review_markdown_path(ticker)
    json_path = markdown_path.with_suffix(".json")
    return markdown_path, json_path


def _watchlist_output_paths(output_path: Path | str | None) -> tuple[Path, Path]:
    markdown_path = Path(output_path) if output_path else _default_research_watchlist_markdown_path()
    json_path = markdown_path.with_suffix(".json")
    return markdown_path, json_path


def _validation_output_paths(*, ticker: str, output_path: Path | str | None) -> tuple[Path, Path]:
    markdown_path = Path(output_path) if output_path else _default_validation_checklist_markdown_path(ticker)
    json_path = markdown_path.with_suffix(".json")
    return markdown_path, json_path


def _human_review_summary_output_paths(output_path: Path | str | None) -> tuple[Path, Path]:
    markdown_path = Path(output_path) if output_path else _default_human_review_summary_markdown_path()
    json_path = markdown_path.with_suffix(".json")
    return markdown_path, json_path


def _stringify_reasoning(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)
