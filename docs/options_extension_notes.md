# Options Research Extension Notes

This project currently produces underlying-equity and ETF research signals. A future options research layer could plug in after those signals are generated, without changing the current analyst or portfolio decision logic.

## Suggested integration point

The cleanest extension point is after `run_hedge_fund()` returns ticker-level decisions and analyst signals. A follow-on module could consume:

- the final underlying signal
- analyst reasoning summaries
- confidence values
- the evaluation window used for the run

## Possible future inputs

- option chain snapshot
- implied volatility rank or percentile
- realized volatility
- earnings and event calendar data
- bid/ask spread and liquidity filters
- DTE and strike selection rules
- defined-risk strategy mapping rules

## Guardrails

- Do not infer option chain details that were not supplied.
- Do not fabricate Greeks, liquidity, IV rank, earnings dates, strikes, expirations, or execution assumptions.
- Keep any future options layer educational and research-oriented unless the project explicitly changes scope.

## Possible future module shape

A future module such as `src/extensions/options_research.py` could expose a function that accepts a completed underlying research result and returns structured, optional annotations for downstream reporting.
