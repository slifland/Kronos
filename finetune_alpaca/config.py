import os


class AlpacaFinetuneConfig:
    """Configuration for Alpaca-backed Kronos finetuning."""

    def __init__(self):
        # Data source and universe
        self.symbols_file = "./finetune_alpaca/symbols.txt"
        self.timeframe = "15Min"
        self.feed = "iex"
        self.adjustment = "raw"
        self.timezone = "America/New_York"
        self.chunk_size = 100

        # Date ranges
        # Keep validation and test strictly later than train to reduce overfitting pressure.
        self.dataset_begin_time = "2018-01-01"
        self.dataset_end_time = "2025-12-31"
        self.train_time_range = ["2018-01-01", "2023-06-30"]
        self.val_time_range = ["2023-07-01", "2024-06-30"]
        self.test_time_range = ["2024-07-01", "2025-12-31"]

        # Windowing and features
        self.lookback_window = 400
        self.predict_window = 1
        self.max_context = 512
        self.feature_list = ["open", "high", "low", "close", "vol", "amt"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]
        self.clip = 5.0

        # Dataset output
        self.dataset_path = "./data/alpaca_processed_datasets"
        self.metadata_path = f"{self.dataset_path}/metadata.json"
        self.require_complete_bars = True

        # Model paths
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"
        self.finetuned_tokenizer_path = "./outputs/models/alpaca_tokenizer/checkpoints/best_model"
        self.finetuned_predictor_path = "./outputs/models/alpaca_predictor/checkpoints/best_model"

        # Training hyperparameters
        self.epochs = 30
        self.log_interval = 100
        self.batch_size = 50
        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5
        self.accumulation_steps = 1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        self.seed = 100
        self.num_workers = 2

        # Logging and output
        self.use_comet = False
        self.comet_config = {
            "api_key": os.getenv("COMET_API_KEY", ""),
            "project_name": "Kronos-Alpaca-Finetune",
            "workspace": os.getenv("COMET_WORKSPACE", ""),
        }
        self.use_wandb = True
        self.wandb_config = {
            "project": os.getenv("WANDB_PROJECT", "Kronos-Alpaca-Finetune"),
            "entity": os.getenv("WANDB_ENTITY", ""),
            "name": os.getenv("WANDB_NAME", ""),
            "tags": ["alpaca_finetune"],
        }
        self.comet_tag = "alpaca_finetune"
        self.comet_name = "alpaca_finetune"
        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = "alpaca_tokenizer"
        self.predictor_save_folder_name = "alpaca_predictor"
