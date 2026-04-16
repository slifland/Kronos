# Alpaca Finetuning Pipeline

This workflow prepares Kronos finetuning datasets from Alpaca historical bars instead of Qlib.

## What Alpaca Can Supply

Alpaca can supply the historical OHLCV bars for finetuning. The current implementation uses:

- stock bars from the Alpaca historical market data API
- configurable bar sizes such as `15Min`
- configurable feed and adjustment mode

The pipeline does **not** try to discover S&P 500 constituents for you. Provide your training universe through `finetune_alpaca/symbols.txt` or edit `symbols_file` in [config.py](/Users/sethlifland/dev/Kronos/finetune_alpaca/config.py).

## Credentials

Set Alpaca credentials before preprocessing:

```bash
export ALPACA_API_KEY_ID=...
export ALPACA_API_SECRET_KEY=...
```

## Step 1: Define the universe

Create `finetune_alpaca/symbols.txt` with one symbol per line. For an S&P 500 experiment, put your constituent list there.

## Step 2: Build train/val/test pickles

```bash
python finetune_alpaca/alpaca_data_preprocess.py
```

This writes:

- `train_data.pkl`
- `val_data.pkl`
- `test_data.pkl`
- `metadata.json`

into `data/alpaca_processed_datasets/` by default.

## Step 3: Finetune the tokenizer

```bash
python finetune_alpaca/train_tokenizer.py
```

On Apple Silicon, the single-process path will automatically prefer `MPS`. On CUDA machines, it will prefer `CUDA`. If you want multi-GPU CUDA training, you can still use:

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS finetune_alpaca/train_tokenizer.py
```

## Step 4: Finetune the predictor

```bash
python finetune_alpaca/train_predictor.py
```

The default pretrained predictor is now `NeoQuasar/Kronos-base`.

## Step 5: Benchmark the finetuned model

Point the Alpaca backtest runner at the fine-tuned checkpoints:

```bash
python -m kronos_trading.backtest.run \
  --symbol AAPL \
  --start 2025-01-02 \
  --end 2025-03-31 \
  --timeframe 15Min \
  --model-id ./outputs/models/alpaca_predictor/checkpoints/best_model \
  --tokenizer-id ./outputs/models/alpaca_tokenizer/checkpoints/best_model
```

## Notes

- The default config is tuned for a first-pass `15Min` experiment.
- Single-process runs auto-select `CUDA`, then `MPS`, then `CPU`.
- The default predictor base model is `Kronos-base`.
- Alpaca credentials are required for preprocessing, but not for local training after the pickles are written.
- The default `feed` is `iex`. If you have a higher-tier Alpaca market data subscription, change the feed in the config.
