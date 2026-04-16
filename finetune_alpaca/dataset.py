import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from finetune_alpaca.config import AlpacaFinetuneConfig


class AlpacaPickleDataset(Dataset):
    """Dataset loader for Alpaca-prepared train/val/test pickle files."""

    def __init__(self, data_type: str = "train"):
        self.config = AlpacaFinetuneConfig()
        if data_type not in ["train", "val", "test"]:
            raise ValueError("data_type must be one of: train, val, test")
        self.data_type = data_type
        self.py_rng = random.Random(self.config.seed)

        self.data_path = f"{self.config.dataset_path}/{data_type}_data.pkl"
        if data_type == "train":
            self.n_samples = self.config.n_train_iter
        elif data_type == "val":
            self.n_samples = self.config.n_val_iter
        else:
            self.n_samples = 0

        with open(self.data_path, "rb") as handle:
            self.data = pickle.load(handle)

        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list
        self.symbols = list(self.data.keys())
        self.indices = []

        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            if "datetime" not in df.columns:
                df = df.rename(columns={df.columns[0]: "datetime"})

            df["minute"] = df["datetime"].dt.minute
            df["hour"] = df["datetime"].dt.hour
            df["weekday"] = df["datetime"].dt.weekday
            df["day"] = df["datetime"].dt.day
            df["month"] = df["datetime"].dt.month
            self.data[symbol] = df[self.feature_list + self.time_feature_list]

            num_samples = len(df) - self.window + 1
            for start_idx in range(max(0, num_samples)):
                self.indices.append((symbol, start_idx))

        if data_type in {"train", "val"}:
            self.n_samples = min(self.n_samples, len(self.indices))
        else:
            self.n_samples = len(self.indices)

    def set_epoch_seed(self, epoch: int):
        self.py_rng.seed(self.config.seed + epoch)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.data_type in {"train", "val"}:
            sampled_idx = self.py_rng.randint(0, len(self.indices) - 1)
        else:
            sampled_idx = idx

        symbol, start_idx = self.indices[sampled_idx]
        df = self.data[symbol]
        window_df = df.iloc[start_idx : start_idx + self.window]

        x = window_df[self.feature_list].values.astype(np.float32)
        x_stamp = window_df[self.time_feature_list].values.astype(np.float32)

        past_x = x[: self.config.lookback_window]
        x_mean = np.mean(past_x, axis=0)
        x_std = np.std(past_x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)

