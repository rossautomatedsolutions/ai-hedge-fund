# Basket Runner Agent Matrix

This note maps the current `basket_runner` analyst presets to the analyst agents defined in [`src/utils/analysts.py`](/C:/GitHub/ai-hedge-fund/src/utils/analysts.py) and the data helpers they call under [`src/tools/api.py`](/C:/GitHub/ai-hedge-fund/src/tools/api.py).

## Presets

| Preset | Agents included |
| --- | --- |
| `all` | Every analyst in `ANALYST_ORDER` |
| `core` | `fundamentals_analyst`, `technical_analyst`, `valuation_analyst` |
| `no-news` | Same as `all`, except `sentiment_analyst` and `news_sentiment_analyst` |
| `technical-only` | `technical_analyst` |

Notes:
- `risk_management_agent` and `portfolio_manager` always run, but they are not controlled by `--analyst-preset`.
- `fast-data-mode` reduces timeouts/retries and skips optional slow news/insider helpers.

## Offline Demo Coverage

- Current supported offline demo tickers: `AAPL`, `BB`, `GME`, `MSFT`, `NVDA`
- Offline demo preserves `data_status=offline_demo` and the static-data disclaimer.
- Every analyst below can work in offline demo mode for a supported ticker because the fixture format covers all helper families currently used by analysts:
  `prices`, `financial_metrics`, `line_items`, `company_facts`/`market_cap`, `company_news`, and `insider_trades`.

## Analyst Matrix

| Analyst key | Included in presets | Data it appears to require | Offline demo today | Live-mode limitation without `FINANCIAL_DATASETS_API_KEY` |
| --- | --- | --- | --- | --- |
| `aswath_damodaran` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes, for supported fixture tickers | Usually blocked before run if required base endpoints cannot authenticate |
| `ben_graham` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes | Same base-endpoint limitation |
| `bill_ackman` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes | Same base-endpoint limitation |
| `cathie_wood` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes | Same base-endpoint limitation |
| `charlie_munger` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap`, `insider_trades`, `company_news` | Yes | Base endpoints may block the run; news/insider context is also unavailable without live access |
| `fundamentals_analyst` | `all`, `core`, `no-news` | `financial_metrics` | Yes | Usually blocked before run if financial metrics cannot authenticate |
| `growth_analyst` | `all`, `no-news` | `financial_metrics`, `insider_trades` | Yes | Base endpoints may block the run; insider trade context also depends on live access |
| `michael_burry` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap`, `insider_trades`, `company_news` | Yes | Base endpoints may block the run; news/insider context is also unavailable without live access |
| `mohnish_pabrai` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes | Same base-endpoint limitation |
| `nassim_taleb` | `all`, `no-news` | `prices`, `financial_metrics`, `line_items`, `market_cap`, `insider_trades`, `company_news` | Yes | Base endpoints may block the run; also needs news/insider context when live |
| `news_sentiment_analyst` | `all` | `company_news` | Yes | Excluded from `no-news`; live mode needs reachable news data |
| `peter_lynch` | `all`, `no-news` | `line_items`, `market_cap`, `insider_trades`, `company_news` | Yes | Base endpoints may block the run; news/insider context is also unavailable without live access |
| `phil_fisher` | `all`, `no-news` | `line_items`, `market_cap`, `insider_trades`, `company_news` | Yes | Base endpoints may block the run; news/insider context is also unavailable without live access |
| `rakesh_jhunjhunwala` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes | Same base-endpoint limitation |
| `sentiment_analyst` | `all` | `insider_trades`, `company_news` | Yes | Excluded from `no-news`; live mode needs reachable news/insider data |
| `stanley_druckenmiller` | `all`, `no-news` | `prices`, `financial_metrics`, `line_items`, `market_cap`, `insider_trades`, `company_news` | Yes | Base endpoints may block the run; also needs news/insider context when live |
| `technical_analyst` | `all`, `core`, `no-news`, `technical-only` | `prices` | Yes | Usually blocked before run if price data cannot authenticate |
| `valuation_analyst` | `all`, `core`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes | Usually blocked before run if required base endpoints cannot authenticate |
| `warren_buffett` | `all`, `no-news` | `financial_metrics`, `line_items`, `market_cap` | Yes | Same base-endpoint limitation |

## What Actually Blocks Live Mode

- `basket_runner` always runs `run_ticker_data_check()` before the LLM/agent workflow.
- That data check currently probes the required base endpoints:
  `prices`, `financial_metrics`, `line_items`, and `company_facts`.
- If those calls fail, the ticker stops before any analyst runs.
- Without `FINANCIAL_DATASETS_API_KEY`, the common outcomes are:
  `missing_api_key`, `unauthorized_401`, or `partial_data`.
- The only exception is if the provider returns public `200` responses for all required base endpoints for that ticker/date range; the tests cover that as a possible but non-default path.
