from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from src.offline_demo_data import OFFLINE_DEMO_DATA_STATUS, OFFLINE_DEMO_DISCLAIMER

from .common import DECISION_SUMMARY_DISCLAIMER, ResearchPacketArtifacts, _csv_row_list, _stringify_reasoning, _write_json


def _parse_vote_summary(vote_summary: Any) -> dict[str, int]:
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for part in str(vote_summary or "").split(","):
        label, _, value = part.strip().partition("=")
        normalized_label = label.strip().lower()
        if normalized_label not in counts:
            continue
        try:
            counts[normalized_label] = int(value.strip())
        except ValueError:
            counts[normalized_label] = 0
    return counts


def _split_sentences(text: Any) -> list[str]:
    normalized = _stringify_reasoning(text).replace("\n", " ").strip()
    if not normalized:
        return []
    chunks = [chunk.strip(" -") for chunk in normalized.split(".") if chunk.strip()]
    return chunks or [normalized]


def _packet_data_mode(rows: list[dict[str, Any]], *, offline_demo_data: bool) -> str:
    if offline_demo_data:
        return "offline_demo"
    data_statuses = {str(row.get("data_status") or "").strip() for row in rows}
    if OFFLINE_DEMO_DATA_STATUS in data_statuses or "fixture_data" in data_statuses:
        return "offline_demo"
    return "live"


def _best_reasoning_excerpt(rows: list[dict[str, Any]], *, preferred_consensus: str | None = None) -> str:
    candidate_rows = rows
    if preferred_consensus:
        filtered = [row for row in rows if str(row.get("analyst_consensus") or "").strip().lower() == preferred_consensus]
        if filtered:
            candidate_rows = filtered
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            0 if str(row.get("run_status") or "").strip().lower() == "success" else 1,
            -int(str(row.get("confidence") or "0") or "0"),
        ),
    )
    for row in candidate_rows:
        for sentence in _split_sentences(row.get("reasoning")):
            return sentence
    return ""


def _manual_checks_for_ticker(packet_rows: list[dict[str, Any]], *, data_mode: str) -> list[str]:
    checks = [
        "Validate the latest price action, volume, and trend context with current charts.",
        "Recheck company-specific news, filings, and catalysts that may have changed since the run.",
        "Confirm whether the portfolio-manager action still matches the latest analyst signals after refreshing data.",
    ]
    if data_mode == "offline_demo":
        checks.insert(0, "Refresh the run with live data before treating any research angle as current.")
    if any(str(row.get("comparison_note") or "").strip() for row in packet_rows):
        checks.append("Inspect the preset rows with conflict notes to see whether the disagreement comes from votes, rationale, or both.")
    return checks


def _data_limitations_for_ticker(packet_rows: list[dict[str, Any]], *, data_mode: str, preset_count: int) -> list[str]:
    limitations = [
        "This packet summarizes agent output and is intended for educational research review, not trading instructions.",
        "Portfolio-manager actions are final outputs and may differ from simple analyst vote counts.",
    ]
    if data_mode == "offline_demo":
        limitations.append("This packet is based on static fixture data and should not be treated as current market analysis.")
    else:
        limitations.append("Live-data runs can still age quickly and require current-data validation before any real-world use.")
    if preset_count <= 1:
        limitations.append("Only one analyst preset was analyzed, so cross-preset disagreement is limited.")
    if any(str(row.get("run_status") or "").strip().lower() != "success" for row in packet_rows):
        limitations.append("At least one preset did not complete successfully, so the comparison is incomplete.")
    return limitations


def _notable_risks_for_ticker(packet_rows: list[dict[str, Any]], *, data_mode: str) -> list[str]:
    risks: list[str] = []
    if data_mode == "offline_demo":
        risks.append("Static fixture inputs can diverge materially from the current market regime.")
    if any(str(row.get("analyst_consensus") or "").strip().lower() == "mixed" for row in packet_rows):
        risks.append("Agent disagreement remains unresolved, which weakens confidence in any single research angle.")
    if any(str(row.get("comparison_note") or "").strip() for row in packet_rows):
        risks.append("One or more presets show action-versus-consensus conflict, so the final action may not reflect broad analyst alignment.")
    if any(str(row.get("run_status") or "").strip().lower() != "success" for row in packet_rows):
        risks.append("Failed preset runs reduce coverage and can hide contradictory signals.")
    if not risks:
        risks.append("Even aligned presets can miss catalyst, liquidity, or macro risks that require manual review.")
    return risks


def _build_ticker_packet(
    ticker: str,
    packet_rows: list[dict[str, Any]],
    *,
    model: str,
    data_mode: str,
    preset_order: list[str] | None = None,
    preset_run_dirs: dict[str, str] | None = None,
) -> dict[str, Any]:
    preset_positions = {preset: index for index, preset in enumerate(preset_order or [])}
    ordered_rows = sorted(
        packet_rows,
        key=lambda row: (
            preset_positions.get(str(row.get("analyst_preset") or "").strip(), len(preset_positions)),
            str(row.get("analyst_preset") or ""),
        ),
    )
    presets = [str(row.get("analyst_preset") or "").strip() for row in ordered_rows]
    preset_summaries: list[dict[str, Any]] = []
    disagreement_points: list[str] = []
    confidence_by_preset: dict[str, str] = {}
    action_by_preset: dict[str, str] = {}
    consensus_by_preset: dict[str, str] = {}
    comparison_notes: list[str] = []

    for row in ordered_rows:
        preset = str(row.get("analyst_preset") or "").strip()
        action = str(row.get("action") or "").strip() or "FAILED"
        confidence = str(row.get("confidence") or "").strip() or "-"
        vote_summary = str(row.get("analyst_vote_summary") or "").strip() or "bullish=0, bearish=0, neutral=0"
        consensus = str(row.get("analyst_consensus") or "").strip() or "none"
        comparison_note = str(row.get("comparison_note") or "").strip()
        reasoning = _stringify_reasoning(row.get("reasoning"))
        run_status = str(row.get("run_status") or "").strip() or "unknown"

        action_by_preset[preset] = action
        confidence_by_preset[preset] = confidence
        consensus_by_preset[preset] = consensus
        if comparison_note:
            comparison_notes.append(f"{preset}: {comparison_note}")
            disagreement_points.append(f"{preset}: {comparison_note}")

        counts = _parse_vote_summary(vote_summary)
        if consensus == "mixed":
            disagreement_points.append(
                f"{preset}: analyst votes are mixed ({vote_summary}), which is a signal to investigate rather than a directional takeaway."
            )
        elif action.lower() in {"buy", "sell", "short"} and consensus not in {"none", ""}:
            expected = "bullish" if action.lower() == "buy" else "bearish"
            if consensus != expected:
                disagreement_points.append(
                    f"{preset}: final action is `{action}` while analyst consensus is `{consensus}`, indicating agent disagreement."
                )

        preset_summaries.append(
            {
                "preset": preset,
                "action": action,
                "confidence": confidence,
                "analyst_vote_summary": vote_summary,
                "analyst_consensus": consensus,
                "comparison_note": comparison_note,
                "reasoning": reasoning,
                "run_status": run_status,
                "data_status": str(row.get("data_status") or "").strip(),
                "run_dir": (preset_run_dirs or {}).get(preset) or str(row.get("run_dir") or "").strip(),
                "log_path": str(row.get("log_path") or "").strip(),
                "vote_counts": counts,
            }
        )

    technical_row = next((row for row in ordered_rows if str(row.get("analyst_preset") or "").strip() == "technical-only"), None)
    broader_rows = [row for row in ordered_rows if str(row.get("analyst_preset") or "").strip() != "technical-only"]

    bull_case_parts: list[str] = []
    if technical_row and str(technical_row.get("action") or "").strip().lower() == "buy":
        bull_case_parts.append("Technical-only output is bullish, which can be treated as a momentum research angle to investigate.")
    bullish_excerpt = _best_reasoning_excerpt(ordered_rows, preferred_consensus="bullish")
    if bullish_excerpt:
        bull_case_parts.append(f"Representative bullish signal: {bullish_excerpt}.")
    elif any(str(row.get("action") or "").strip().lower() == "buy" for row in ordered_rows):
        bull_case_parts.append("At least one preset still produced a bullish directional action, but it requires current-data validation.")
    else:
        bull_case_parts.append("No preset produced a clean bullish alignment, so upside arguments remain tentative.")

    bear_case_parts: list[str] = []
    if broader_rows and any(str(row.get("analyst_consensus") or "").strip().lower() in {"bearish", "mixed"} for row in broader_rows):
        bear_case_parts.append("Broader presets are mixed or bearish, suggesting the bullish technical read may not survive wider review.")
    bearish_excerpt = _best_reasoning_excerpt(ordered_rows, preferred_consensus="bearish")
    if bearish_excerpt:
        bear_case_parts.append(f"Representative bearish signal: {bearish_excerpt}.")
    elif any(str(row.get("action") or "").strip().lower() in {"sell", "short"} for row in ordered_rows):
        bear_case_parts.append("One or more presets still lean bearish on the final action, which is a cautionary signal to investigate.")
    else:
        bear_case_parts.append("No preset produced a decisive bearish consensus, but downside risks still require manual review.")

    if technical_row and broader_rows:
        broader_consensus = {str(row.get("analyst_consensus") or "").strip().lower() for row in broader_rows}
        if str(technical_row.get("analyst_consensus") or "").strip().lower() == "bullish" and broader_consensus & {"mixed", "bearish"}:
            disagreement_points.append(
                "Technical-only is bullish while broader presets are mixed or bearish, so the main research angle requires wider current-data validation."
            )

    if not disagreement_points:
        disagreement_points.append("No major preset conflict was flagged, but the packet still requires current-data validation.")

    return {
        "ticker": ticker,
        "model": model,
        "data_mode": data_mode,
        "presets_analyzed": presets,
        "preset_summaries": preset_summaries,
        "action_by_preset": action_by_preset,
        "confidence_by_preset": confidence_by_preset,
        "consensus_by_preset": consensus_by_preset,
        "comparison_notes": comparison_notes,
        "bull_case": " ".join(bull_case_parts).strip(),
        "bear_case": " ".join(bear_case_parts).strip(),
        "key_disagreement_points": disagreement_points,
        "data_limitations": _data_limitations_for_ticker(ordered_rows, data_mode=data_mode, preset_count=len(ordered_rows)),
        "what_to_check_next_manually": _manual_checks_for_ticker(ordered_rows, data_mode=data_mode),
        "notable_risks_or_reasons_not_to_act": _notable_risks_for_ticker(ordered_rows, data_mode=data_mode),
    }


def _render_research_packet_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Research Packet",
        "",
        DECISION_SUMMARY_DISCLAIMER,
        "",
        "This packet is designed for educational research review. It highlights research angles, agent disagreement, and items that require current-data validation.",
        "",
    ]
    if payload.get("offline_demo_data"):
        lines.extend(
            [
                f"Offline/demo data note: {OFFLINE_DEMO_DISCLAIMER}",
                "This packet is based on static fixture data and should not be treated as current market analysis.",
                "",
            ]
        )

    lines.extend(
        [
            "## Run Context",
            "",
            f"- Model: `{payload.get('model', '')}`",
            f"- Data mode: `{payload.get('data_mode', '')}`",
            f"- Presets analyzed: `{', '.join(payload.get('presets_analyzed', []))}`",
            f"- Run directory: `{payload.get('run_dir', '')}`",
            "",
        ]
    )

    for ticker_packet in payload.get("tickers", []):
        lines.extend(
            [
                f"## {ticker_packet['ticker']}",
                "",
                f"- Ticker: `{ticker_packet['ticker']}`",
                f"- Model: `{ticker_packet['model']}`",
                f"- Data mode: `{ticker_packet['data_mode']}`",
                f"- Presets analyzed: `{', '.join(ticker_packet['presets_analyzed'])}`",
                "",
                "| Preset | Final Action | Confidence | Analyst Votes | Analyst Consensus | Comparison Notes | Run Status |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for preset_summary in ticker_packet["preset_summaries"]:
            analyst_votes = preset_summary["analyst_vote_summary"].replace("|", "\\|")
            comparison_note = (preset_summary["comparison_note"] or "-").replace("|", "\\|")
            lines.append(
                f"| {preset_summary['preset']} | {preset_summary['action']} | {preset_summary['confidence']} | "
                f"{analyst_votes} | {preset_summary['analyst_consensus']} | {comparison_note} | {preset_summary['run_status']} |"
            )

        lines.extend(["", "### Final Action By Preset", ""])
        for preset in ticker_packet["presets_analyzed"]:
            lines.append(
                f"- `{preset}`: action=`{ticker_packet['action_by_preset'].get(preset, '-')}`, "
                f"confidence=`{ticker_packet['confidence_by_preset'].get(preset, '-')}`, "
                f"analyst_consensus=`{ticker_packet['consensus_by_preset'].get(preset, '-')}`"
            )

        lines.extend(["", "### Bull Case", "", ticker_packet["bull_case"], "", "### Bear Case", "", ticker_packet["bear_case"], "", "### Key Disagreement Points", ""])
        for item in ticker_packet["key_disagreement_points"]:
            lines.append(f"- {item}")

        lines.extend(["", "### Data Limitations", ""])
        for item in ticker_packet["data_limitations"]:
            lines.append(f"- {item}")

        lines.extend(["", "### What To Check Next Manually", ""])
        for item in ticker_packet["what_to_check_next_manually"]:
            lines.append(f"- {item}")

        lines.extend(["", "### Notable Risks / Reasons Not To Act", ""])
        for item in ticker_packet["notable_risks_or_reasons_not_to_act"]:
            lines.append(f"- {item}")

        lines.append("")

    return "\n".join(lines) + "\n"


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
        "comparison_note": row.get("comparison_note", ""),
        "reasoning": row.get("reasoning", ""),
        "data_status": row.get("data_status", ""),
        "run_status": row.get("run_status", ""),
        "failure_classification": row.get("failure_classification", ""),
        "run_dir": run_dir.as_posix(),
        "log_path": (run_dir / "logs" / f"{ticker}.log").as_posix() if ticker else "",
    }


def write_research_packet(
    run_dir: Path,
    *,
    config: Any,
    comparison_rows: list[dict[str, Any]] | None = None,
    decision_rows: list[dict[str, Any]] | None = None,
    preset_run_dirs: dict[str, str] | None = None,
) -> ResearchPacketArtifacts:
    source_rows = comparison_rows if comparison_rows is not None else decision_rows
    if source_rows is not None:
        packet_rows = [dict(row) for row in source_rows]
    elif (run_dir / "preset_comparison.csv").exists():
        packet_rows = _csv_row_list(run_dir / "preset_comparison.csv")
    else:
        decision_csv_rows = _csv_row_list(run_dir / "combined_decisions.csv")
        packet_rows = [_comparison_row_from_decision_row(row, run_dir=run_dir) for row in decision_csv_rows]

    ordered_presets: list[str] = []
    seen_presets: set[str] = set()
    for row in packet_rows:
        preset = str(row.get("analyst_preset") or "").strip()
        if preset and preset not in seen_presets:
            ordered_presets.append(preset)
            seen_presets.add(preset)

    data_mode = _packet_data_mode(packet_rows, offline_demo_data=config.offline_demo_data)
    tickers = sorted({str(row.get("ticker") or "").strip() for row in packet_rows if str(row.get("ticker") or "").strip()})
    ticker_packets = [
        _build_ticker_packet(
            ticker,
            [row for row in packet_rows if str(row.get("ticker") or "").strip() == ticker],
            model=config.model,
            data_mode=data_mode,
            preset_order=ordered_presets,
            preset_run_dirs=preset_run_dirs,
        )
        for ticker in tickers
    ]
    payload = {
        "generated_at": datetime.now().isoformat(),
        "run_dir": run_dir.as_posix(),
        "model": config.model,
        "data_mode": data_mode,
        "offline_demo_data": config.offline_demo_data,
        "presets_analyzed": ordered_presets,
        "tickers": ticker_packets,
    }
    markdown_path = run_dir / "research_packet.md"
    json_path = run_dir / "research_packet.json"
    markdown_path.write_text(_render_research_packet_markdown(payload), encoding="utf-8")
    _write_json(json_path, payload)
    return ResearchPacketArtifacts(markdown_path=markdown_path, json_path=json_path, payload=payload)
