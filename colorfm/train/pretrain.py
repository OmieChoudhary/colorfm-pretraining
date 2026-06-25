from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colorfm.data import SpectralDataset, make_mask
from colorfm.models.spectral_transformer import SpectralTransformer
from colorfm.objectives.losses import masked_reconstruction_loss, lab_regression_loss, spectral_smoothness_loss
from colorfm.utils.experiment import (
    append_jsonl,
    count_parameters,
    load_yaml,
    prepare_run_dir,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
)


def compute_losses(
    out: dict[str, torch.Tensor],
    spectrum: torch.Tensor,
    lab: torch.Tensor,
    mask: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    objectives = cfg.get("objectives", {})
    recon_weight = float(objectives.get("reconstruction_weight", 1.0))
    lab_weight = float(objectives.get("lab_weight", cfg.get("training", {}).get("lab_loss_weight", 0.0)))
    smooth_weight = float(objectives.get("smoothness_weight", 0.0))

    recon_loss = masked_reconstruction_loss(out["reconstruction"], spectrum, mask)
    lab_loss = lab_regression_loss(out["lab"], lab)
    smooth_loss = spectral_smoothness_loss(out["reconstruction"])
    loss = recon_weight * recon_loss + lab_weight * lab_loss + smooth_weight * smooth_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "lab_loss": float(lab_loss.detach().cpu()),
        "smoothness_loss": float(smooth_loss.detach().cpu()),
    }


def evaluate(
    model: SpectralTransformer,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    cfg: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon_loss": 0.0, "lab_loss": 0.0, "smoothness_loss": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            spectrum = batch["spectrum"].to(device)
            lab = batch["lab"].to(device)
            mask = make_mask(spectrum.shape[0], spectrum.shape[1], mask_ratio, device)
            out = model(spectrum, mask)
            _, losses = compute_losses(out, spectrum, lab, mask, cfg)
            bsz = spectrum.shape[0]
            for key in totals:
                totals[key] += losses[key] * bsz
            n += bsz
    return {key: value / max(n, 1) for key, value in totals.items()}


def train_from_config(cfg: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    training = cfg.get("training", {})
    torch.set_num_threads(int(training.get("torch_num_threads", 1)))
    device = resolve_device(training.get("device", "auto"))
    run_dir = prepare_run_dir(cfg, config_path)
    save_yaml(cfg, run_dir / "config.yaml")

    train_ds = SpectralDataset(cfg["train_csv"])
    val_ds = SpectralDataset(cfg["val_csv"])
    batch_size = int(training.get("batch_size", 64))
    num_workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    seq_len = train_ds.spectra.shape[1]
    model = SpectralTransformer(seq_len=seq_len, **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 5e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(training.get("epochs", 1)), 1),
        eta_min=float(training.get("min_lr", 1e-6)),
    )

    mask_ratio = float(training.get("mask_ratio", 0.30))
    epochs = int(training.get("epochs", 1))
    max_train_batches = training.get("max_train_batches")
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)

    best_val = float("inf")
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    metrics_path = run_dir / "metrics" / "metrics.jsonl"
    model_params = count_parameters(model)
    start = time.time()

    print(f"run_dir={run_dir}")
    print(f"device={device} train_rows={len(train_ds)} val_rows={len(val_ds)} params={model_params:,}")

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "recon_loss": 0.0, "lab_loss": 0.0, "smoothness_loss": 0.0}
        n = 0
        epoch_start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and step > max_train_batches:
                break
            spectrum = batch["spectrum"].to(device)
            lab = batch["lab"].to(device)
            mask = make_mask(spectrum.shape[0], spectrum.shape[1], mask_ratio, device)
            out = model(spectrum, mask)
            loss, losses = compute_losses(out, spectrum, lab, mask, cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
            optimizer.step()

            bsz = spectrum.shape[0]
            for key in totals:
                totals[key] += losses[key] * bsz
            n += bsz
        scheduler.step()

        train_metrics = {f"train_{key}": value / max(n, 1) for key, value in totals.items()}
        val_metrics_raw = evaluate(model, val_loader, device, mask_ratio, cfg)
        val_metrics = {f"val_{key}": value for key, value in val_metrics_raw.items()}
        samples_per_sec = n / max(time.time() - epoch_start, 1e-9)
        record = {
            "epoch": epoch,
            "run_name": cfg.get("run_name", run_dir.name),
            "lr": scheduler.get_last_lr()[0],
            "samples_per_sec": samples_per_sec,
            "model_params": model_params,
            **train_metrics,
            **val_metrics,
        }
        append_jsonl(record, metrics_path)
        print(
            f"epoch={epoch:03d} train_loss={record['train_loss']:.6f} "
            f"val_loss={record['val_loss']:.6f} val_recon={record['val_recon_loss']:.6f} "
            f"val_lab={record['val_lab_loss']:.6f} samples/s={samples_per_sec:.1f}"
        )

        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            torch.save({"model_state": model.state_dict(), "config": cfg, "seq_len": seq_len, "run_dir": str(run_dir)}, ckpt_path)

    final = {
        "run_name": cfg.get("run_name", run_dir.name),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "best_val_loss": best_val,
        "model_params": model_params,
        "elapsed_sec": time.time() - start,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
    }
    save_json(final, run_dir / "final_metrics.json")
    print(f"saved checkpoint={ckpt_path}")
    print(f"saved final_metrics={run_dir / 'final_metrics.json'}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    train_from_config(cfg, args.config)


if __name__ == "__main__":
    main()
