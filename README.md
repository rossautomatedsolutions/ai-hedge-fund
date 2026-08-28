# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

This system employs several agents working together:

1. Aswath Damodaran Agent - The Dean of Valuation, focuses on story, numbers, and disciplined valuation
2. Ben Graham Agent - The godfather of value investing, only buys hidden gems with a margin of safety
3. Bill Ackman Agent - An activist investor, takes bold positions and pushes for change
4. Cathie Wood Agent - The queen of growth investing, believes in the power of innovation and disruption
5. Charlie Munger Agent - Warren Buffett's partner, only buys wonderful businesses at fair prices
6. Michael Burry Agent - The Big Short contrarian who hunts for deep value
7. Mohnish Pabrai Agent - The Dhandho investor, who looks for doubles at low risk
8. Nassim Taleb Agent - The Black Swan risk analyst, focuses on tail risk, antifragility, and asymmetric payoffs
9. Peter Lynch Agent - Practical investor who seeks "ten-baggers" in everyday businesses
10. Phil Fisher Agent - Meticulous growth investor who uses deep "scuttlebutt" research 
11. Rakesh Jhunjhunwala Agent - The Big Bull of India
12. Stanley Druckenmiller Agent - Macro legend who hunts for asymmetric opportunities with growth potential
13. Warren Buffett Agent - The oracle of Omaha, seeks wonderful companies at a fair price
14. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
15. Sentiment Agent - Analyzes market sentiment and generates trading signals
16. Fundamentals Agent - Analyzes fundamental data and generates trading signals
17. Technicals Agent - Analyzes technical indicators and generates trading signals
18. Risk Manager - Calculates risk metrics and sets position limits
19. Portfolio Manager - Makes final trading decisions and generates orders

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application](#️-web-application)
- [How to Contribute](#how-to-contribute)
- [Feature Requests](#feature-requests)
- [License](#license)

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.

### 1. Clone the Repository

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Set up API keys

Create a `.env` file for your API keys:
```bash
# Create .env file for your API keys (in the root directory)
cp .env.example .env
```

Open and edit the `.env` file to add your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=your-openai-api-key

# For getting financial data to power the hedge fund
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

**Important**: You must set at least one LLM API key (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`) for the hedge fund to work. 

## How to Run

### ⌨️ Command Line Interface

You can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### Quick Start

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

#### Run the AI Hedge Fund
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

You can optionally specify the start and end dates to make decisions over a specific time period.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

#### Run the Backtester
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


Note: The `--ollama`, `--start-date`, and `--end-date` flags work for the backtester, as well!

#### Research Workflow CLI

`src/basket_runner.py` is the bounded research workflow CLI for educational basket analysis, preset comparison, and human review preparation. It does not place live trades, submit broker orders, or mutate external brokerage accounts.

Install the project dependencies first:

```bash
poetry install
```

Show the supported options:

```bash
poetry run python -B -m src.basket_runner --help
```

Run the full research-only workflow for one or more tickers:

```bash
poetry run python -B -m src.basket_runner --full-research-workflow --tickers BB,GME --offline-demo-data --continue-on-error --output-dir outputs\owner_smoke
```

`--full-research-workflow` coordinates the existing research components in one path:

- preset comparison across analyst presets
- `research_packet.md` and `research_packet.json`
- durable `research_journal.csv` append
- `research_watchlist.md` and `research_watchlist.json` refresh
- per-ticker `validation_checklist_<TICKER>.md` and `.json`
- existing human-review preparation, with optional later use of `--record-human-review` and `--review-human-reviews`

Useful standalone follow-up commands:

```bash
poetry run python -B -m src.basket_runner --research-watchlist --research-journal-path outputs\research_journal.csv
poetry run python -B -m src.basket_runner --validation-checklist --ticker BB --research-journal-path outputs\research_journal.csv --watchlist-path outputs\research_watchlist.json
poetry run python -B -m src.basket_runner --review-human-reviews --human-review-log-path outputs\human_review_log.csv
```

Signal-ledger handoff commands:

```bash
poetry run python -B -m src.basket_runner --tickers BB --analyst-preset core --export-signal-ledger --output-dir outputs\signal_handoff_demo
poetry run python -B -m src.basket_runner --export-signal-ledger --signal-ledger-source-run-dir outputs\ras_ollama_basket_runs\20260828_010203 --signal-ledger-output outputs\reexported_signal_bundle
```

`--export-signal-ledger` writes an AIHF-owned machine-readable bundle for Trading Foundation handoff:

- `signal_ledger.csv`
- `signal_ledger.json`
- `signal_ledger_manifest.json`
- `trading_foundation_trades.csv`
- `trading_foundation_handoff_manifest.json`

Important ledger behavior:

- missing analysts are exported as explicit `abstain` rows rather than silently becoming `neutral`
- offline demo rows stay exportable for contract testing but are marked `is_backtest_eligible=false`
- current live research and historical-window live research also fail closed until point-in-time safety is proven from captured metadata and source behavior
- AIHF does not auto-run Trading Foundation from this path because the current TF engine still writes outputs/cache artifacts inside the Trading Foundation repository tree

See `docs/signal_ledger_trading_foundation_handoff.md` for the audit conclusion and selected integration path.

Output behavior:

- If you pass `--output-dir`, the timestamped basket comparison run is written under `<output-dir>\<timestamp>\...`.
- In `--full-research-workflow` mode, the durable journal/watchlist/validation files default to the same `--output-dir` root unless you explicitly override their paths.
- If you do not pass `--output-dir`, the workflow keeps the existing repo-level defaults under `outputs\`.
- Generated output artifacts stay out of tracked source directories, and `outputs/` remains ignored by git.

Journal and partial-failure behavior:

- The research journal is append-only. This task does not add a new deduplication scheme because no existing identity/deduplication contract was present in the workflow helpers.
- `--continue-on-error` keeps the workflow moving when one or more preset runs fail, and the JSON result reports `workflow_status`, failed/partial tickers, and warning messages so incomplete research is not mistaken for a clean success.
- `--offline-demo-data` is supported for bounded demos and smoke tests, but those artifacts still require current-data validation before any real-world interpretation.

Repository boundary:

- This repository is research-only and does not place live trades.
- `finance_decision_engine` remains a separate deterministic evidence/decision repository.
- No AI Hedge Fund to `finance_decision_engine` integration is being implemented in this workflow task.

### 🖥️ Web Application

The new way to run the AI Hedge Fund is through our web application that provides a user-friendly interface. This is recommended for users who prefer visual interfaces over command line tools.

Please see detailed instructions on how to install and run the web application [here](https://github.com/virattt/ai-hedge-fund/tree/main/app).

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />


## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
