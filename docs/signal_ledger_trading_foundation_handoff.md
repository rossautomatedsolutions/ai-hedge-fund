# Signal Ledger And Trading Foundation Handoff

Audit date: August 28, 2026

## Scope

This repository now owns the signal-export contract and the Trading Foundation handoff bundle.

- Writable repo: `C:\GitHub\ai-hedge-fund`
- Read-only upstream repo: `C:\GitHub\Trading_Foundation`
- Goal: export AI Hedge Fund analyst conclusions into a canonical machine-readable artifact that can be handed to an existing Trading Foundation backtest engine without modifying Trading Foundation source.

## AI Hedge Fund audit summary

`src/basket_runner.py` already had the narrowest safe place to capture point-in-time-oriented context because it records:

- requested tickers
- resolved `start_date` and `end_date`
- analyst preset and selected analysts
- model name and model provider
- offline demo mode
- per-ticker data-check status
- per-ticker final portfolio decision
- raw `analyst_signals` returned by `run_hedge_fund(...)`

The new export path persists that run-time state into `ticker_results.json` so repeated exports can reuse the same source run instead of rerunning analysts.

## Trading Foundation audit summary

Relevant engines inspected:

1. `C:\GitHub\Trading_Foundation\backtesting\run_backtest_from_trades.py`
   - Best fit.
   - Programmatic entrypoint: `backtesting.run_backtest_from_trades.run`
   - Natural contract: external trade rows with required columns `symbol`, `entry_date`, and `side`
   - Appropriate for individual signal evaluation and simple portfolio-style replay

2. `C:\GitHub\Trading_Foundation\backtesting\backtest_runner.py`
   - Underlying execution engine used by `run_backtest_from_trades.py`
   - Supports equity-style mapped trades, risk sizing, concurrency limits, expectancy-style diagnostics, win/loss metrics, Sharpe, drawdown, and equity traces
   - Not safe to auto-run from AIHF as-is because it writes outputs under relative `backtesting/backtests/...` and `core/outputs/...`

3. `C:\GitHub\Trading_Foundation\backtesting\single_strategy_runner.py`
   - Oriented toward Trading Foundation-native strategy callbacks, not external AIHF signal artifacts

4. `C:\GitHub\Trading_Foundation\backtesting\batch_strategy_runner.py`
   - Oriented toward parameter sweeps across internal Trading Foundation strategies, not the AIHF handoff contract

Additional constraint:

- `C:\GitHub\Trading_Foundation\data\data_fetcher.py` writes cache and diagnostics under `core/cache` and `core/outputs`, which breaks the read-only upstream boundary for automated invocation from AIHF.

## Selected integration path

AI Hedge Fund uses a two-stage handoff:

1. Export a canonical signal ledger inside AIHF.
2. Export a Trading Foundation handoff bundle containing a TF-compatible trades CSV plus a manifest that names the selected TF engine and its input contract.

This is the narrowest safe path because:

- it reuses Trading Foundation's existing mapped-trade backtest contract
- it does not modify Trading Foundation
- it does not auto-run a TF engine that still writes inside the TF tree by default
- it keeps all new exporter, schema, tests, and docs inside AIHF

## Output contract

When `--export-signal-ledger` is used, AIHF writes:

- `signal_ledger.csv`
- `signal_ledger.json`
- `signal_ledger_manifest.json`
- `trading_foundation_trades.csv`
- `trading_foundation_handoff_manifest.json`

Key behaviors:

- Missing analysts are exported as explicit `abstain` rows, never silently converted to `neutral`.
- `signal_score` is normalized to positive, negative, zero, or `null` for abstain/unmapped.
- `source_signal_label` preserves the raw source label, including portfolio-manager actions such as `buy`.
- Offline demo artifacts remain exportable for contract testing but are marked ineligible for backtesting.
- Historical-window live research also fails closed unless point-in-time safety can be demonstrated from the source code and captured metadata.

## Current limitation by design

AIHF intentionally does not auto-run Trading Foundation from this export path.

Reason:

- the current Trading Foundation engine and data fetcher still write outputs, cache files, or diagnostics inside the Trading Foundation repository tree

The handoff bundle is therefore owner-machine ready, but invocation remains a deliberate downstream step until Trading Foundation exposes a fully externalized output/cache contract.
