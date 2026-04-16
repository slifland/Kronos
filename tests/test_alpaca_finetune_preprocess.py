from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune_alpaca.alpaca_data_preprocess import load_symbols, split_symbol_frame
from finetune_alpaca.config import AlpacaFinetuneConfig


def test_load_symbols_skips_comments_and_dedupes(tmp_path):
    path = tmp_path / "symbols.txt"
    path.write_text("\n# comment\naapl\nMSFT\nAAPL\n", encoding="utf-8")

    symbols = load_symbols(path)

    assert symbols == ["AAPL", "MSFT"]


def test_split_symbol_frame_respects_ranges():
    config = AlpacaFinetuneConfig()
    config.timezone = "America/New_York"
    config.train_time_range = ["2024-01-01", "2024-01-31"]
    config.val_time_range = ["2024-02-01", "2024-02-29"]
    config.test_time_range = ["2024-03-01", "2024-03-31"]

    index = pd.to_datetime(
        [
            "2024-01-15 09:30",
            "2024-02-15 09:30",
            "2024-03-15 09:30",
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.0, 2.0, 3.0],
            "vol": [10.0, 20.0, 30.0],
            "amt": [10.0, 40.0, 90.0],
        },
        index=index,
    )

    split = split_symbol_frame(frame, config)

    assert len(split["train"]) == 1
    assert len(split["val"]) == 1
    assert len(split["test"]) == 1
