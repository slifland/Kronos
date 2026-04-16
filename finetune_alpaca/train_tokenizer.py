import json
import os
import sys
import time
from time import gmtime, strftime
from pathlib import Path

import comet_ml
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.utils.training_utils import cleanup_ddp, format_time, get_model_size, set_seed, setup_ddp
from finetune_alpaca.config import AlpacaFinetuneConfig
from finetune_alpaca.dataset import AlpacaPickleDataset
from finetune_alpaca.runtime import detect_best_device, is_distributed_run, unwrap_model
from model.kronos import KronosTokenizer


def create_dataloaders(config, rank, world_size, distributed, device):
    train_dataset = AlpacaPickleDataset("train")
    valid_dataset = AlpacaPickleDataset("val")
    use_pin_memory = device.type == "cuda"

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        shuffle=not distributed,
        num_workers=config.get("num_workers", 2),
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, device, config, save_dir, logger, rank, world_size, distributed):
    start_time = time.time()
    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config,
        rank,
        world_size,
        distributed,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["tokenizer_learning_rate"],
        weight_decay=config["adam_weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config["tokenizer_learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.03,
        div_factor=10,
    )

    best_val_loss = float("inf")
    batch_idx_global_train = 0

    for epoch_idx in range(config["epochs"]):
        epoch_start_time = time.time()
        model.train()
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch_idx)
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        train_iter = enumerate(train_loader)
        if rank == 0:
            train_iter = tqdm(
                train_iter,
                total=len(train_loader),
                desc=f"Tokenizer train epoch {epoch_idx + 1}/{config['epochs']}",
                leave=False,
            )

        for i, (ori_batch_x, _) in train_iter:
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)
            current_batch_total_loss = 0.0

            for j in range(config["accumulation_steps"]):
                start_idx = j * (ori_batch_x.shape[0] // config["accumulation_steps"])
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config["accumulation_steps"])
                batch_x = ori_batch_x[start_idx:end_idx]

                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                loss = (recon_loss_pre + recon_loss_all + bsq_loss) / 2
                current_batch_total_loss += loss.item()
                (loss / config["accumulation_steps"]).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0 and (batch_idx_global_train + 1) % config["log_interval"] == 0:
                avg_loss = current_batch_total_loss / config["accumulation_steps"]
                print(
                    f"[Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )
            if rank == 0 and logger:
                logger.log_metric("train_tokenizer_loss_batch", current_batch_total_loss, step=batch_idx_global_train)

            if rank == 0 and hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(loss=f"{current_batch_total_loss:.4f}")
            batch_idx_global_train += 1

        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0
        val_iter = val_loader
        if rank == 0:
            val_iter = tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Tokenizer val epoch {epoch_idx + 1}/{config['epochs']}",
                leave=False,
            )
        with torch.no_grad():
            for ori_batch_x, _ in val_iter:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)
                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)

        if distributed:
            val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
            val_count_tensor = torch.tensor(val_sample_count_rank, device=device)
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_sum_tensor.item() / val_count_tensor.item()
        else:
            avg_val_loss = tot_val_loss_sum_rank / max(val_sample_count_rank, 1)

        if rank == 0:
            print(f"Epoch {epoch_idx + 1}: val_loss={avg_val_loss:.4f}, elapsed={format_time(time.time() - epoch_start_time)}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                unwrap_model(model).save_pretrained(save_path)
                print(f"Best tokenizer saved to {save_path}")

    return {"best_val_loss": best_val_loss, "total_time": format_time(time.time() - start_time)}


def main(config):
    distributed = is_distributed_run()
    if distributed:
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device(detect_best_device())
    set_seed(config["seed"], rank)

    save_dir = os.path.join(config["save_path"], config["tokenizer_save_folder_name"])
    comet_logger = None
    if rank == 0:
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        if config["use_comet"]:
            comet_logger = comet_ml.Experiment(
                api_key=config["comet_config"]["api_key"],
                project_name=config["comet_config"]["project_name"],
                workspace=config["comet_config"]["workspace"],
            )
            comet_logger.add_tag(config["comet_tag"])
            comet_logger.set_name(config["comet_name"])
            comet_logger.log_parameters(config)

    if distributed:
        dist.barrier()

    model = KronosTokenizer.from_pretrained(config["pretrained_tokenizer_path"]).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Training tokenizer on device: {device}")
        print(f"Tokenizer model size: {get_model_size(unwrap_model(model))}")

    result = train_model(model, device, config, save_dir, comet_logger, rank, world_size, distributed)
    if rank == 0:
        with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()), "result": result}, handle, indent=2)
        if comet_logger:
            comet_logger.end()

    if distributed:
        cleanup_ddp()


if __name__ == "__main__":
    main(AlpacaFinetuneConfig().__dict__)
