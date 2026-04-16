# Kronos Trading

This package adds an Alpaca-backed research loop on top of the core Kronos model:

- fetch historical bars from Alpaca
- run a Kronos forecast-driven strategy
- backtest the strategy on those bars
- run the same strategy logic against Alpaca paper trading
- compare the strategy against benchmark curves generated from Alpaca data

## Environment

Set your Alpaca credentials before running the CLI:

```bash
export ALPACA_API_KEY_ID=...
export ALPACA_API_SECRET_KEY=...
```

The runner defaults to the `iex` feed so it works with a basic paper-trading setup for US equities.

## Run A Backtest

```bash
python -m kronos_trading.backtest.run \
  --symbol AAPL \
  --benchmark-symbols SPY,QQQ \
  --start 2025-01-02 \
  --end 2025-03-31 \
  --timeframe 5Min \
  --lookback 400 \
  --pred-len 1 \
  --threshold-bps 20
```

The backtest runner defaults to `NeoQuasar/Kronos-base`, and if you do not pass `--device` it will let `KronosPredictor` auto-select `CUDA`, then `MPS`, then `CPU`.

Outputs are written to `backtest_runs/<symbol>_<timestamp>/`:

- `strategy_results.csv`
- `strategy_trades.csv`
- `benchmark_curves.csv`
- `summary.json`

## Run Paper Trading

Start with a dry run so the script prints decisions without submitting paper orders:

```bash
python -m kronos_trading.run_paper \
  --symbols AAPL,MSFT \
  --timeframe 15Min \
  --lookback 400 \
  --threshold-bps 20 \
  --portfolio-fraction 0.10 \
  --device mps \
  --dry-run
```

To keep polling on an interval:

```bash
python -m kronos_trading.run_paper \
  --symbols AAPL,MSFT \
  --timeframe 15Min \
  --poll-seconds 60 \
  --portfolio-fraction 0.10 \
  --device mps \
  --dry-run
```

The paper runner defaults to `NeoQuasar/Kronos-base`, but you can plug in fine-tuned checkpoints later:

```bash
python -m kronos_trading.run_paper \
  --symbols AAPL,MSFT \
  --model-id ./outputs/models/alpaca_predictor/checkpoints/best_model \
  --tokenizer-id ./outputs/models/alpaca_tokenizer/checkpoints/best_model \
  --dry-run
```

## Notes

- The backtest runner only uses Alpaca historical data and does not place orders.
- The paper runner places Alpaca paper orders unless you pass `--dry-run`.
- The paper runner uses the Alpaca paper trading endpoint `https://paper-api.alpaca.markets`.
- Benchmarks currently include buy-and-hold on the traded symbol, a momentum baseline, and any additional benchmark symbols you pass in.
- If you pass multiple benchmark symbols, the runner also computes an equal-weight benchmark.
