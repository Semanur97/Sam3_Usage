"""SAM3 mask_decoder fine-tuning (Coco_T bubble instance segmentation).

Colab'dan:
    from coco_t.config import Config
    from coco_t.train import train
    history = train(Config(epochs=10))
"""
import gc
import json
import os
import random

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .config import Config
from .data import CocoSAMDataset
from .losses import bce_dice_loss, instance_iou_dice
from .model import (boxes_to_model_space, encode_image, load_model, predict_logits,
                    save_decoder, set_train_mode)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, processor, dataset, device, cfg: Config,
              optimizer=None, scaler=None, order=None):
    """Bir epoch. optimizer verilirse eğitim, verilmezse doğrulama.

    Görüntü başına: encoder 1 kez, kutular `box_chunk`'lık gruplar halinde decoder'dan geçer,
    gradyanlar biriktirilip görüntü başına tek optimizer adımı atılır (eski davranışla aynı).
    Returns: {"loss", "iou", "dice"}  (iou/dice sadece doğrulamada dolu)
    """
    training = optimizer is not None
    set_train_mode(model, training)
    use_amp = cfg.amp and device.type == "cuda"
    order = order if order is not None else range(len(dataset))

    losses, ious, dices = [], [], []
    for idx in tqdm(order, leave=False, desc="Train" if training else "Val"):
        s = dataset[idx]
        N = len(s["boxes"])
        if N == 0:
            continue
        emb, scale = encode_image(model, processor, s["image"], device)
        boxes = boxes_to_model_space(s["boxes"], scale).to(device)
        gt_all = torch.from_numpy(s["masks"]).to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        img_loss, preds = 0.0, []
        for i in range(0, N, cfg.box_chunk):
            sl = slice(i, i + cfg.box_chunk)
            n = len(boxes[sl])
            with torch.set_grad_enabled(training), \
                    torch.autocast("cuda", enabled=use_amp):
                logits = predict_logits(model, emb, boxes[sl], s["size"])
                loss = bce_dice_loss(logits, gt_all[sl]) * (n / N)  # instance ortalaması
            if training:
                scaler.scale(loss).backward()
            else:
                preds.append((torch.sigmoid(logits) > cfg.mask_threshold).cpu().numpy())
            img_loss += loss.item()

        if training:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            iou, dice = instance_iou_dice(np.concatenate(preds), s["masks"])
            ious.extend(iou)
            dices.extend(dice)
        losses.append(img_loss)
        del emb, gt_all
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
    }


def train(cfg: Config = None, model=None, processor=None, train_ds=None, val_ds=None):
    """Tam eğitim. model/processor/dataset verilmezse cfg'den yüklenir (testlerde verilir)."""
    cfg = cfg or Config()
    seed_everything(cfg.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    os.makedirs(cfg.save_dir, exist_ok=True)

    if model is None:
        model, processor = load_model(cfg.model_id, device)
    model.to(device)
    train_ds = train_ds or CocoSAMDataset(cfg.train_ann_path, cfg.base_dir, cfg.max_side)
    val_ds = val_ds or CocoSAMDataset(cfg.val_ann_path, cfg.base_dir, cfg.max_side)
    print(f"Train: {len(train_ds)} görüntü | Val: {len(val_ds)} görüntü")

    params = [p for p in model.mask_decoder.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)
    scaler = GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")

    # Epoch 0: fine-tune öncesi (zero-shot) doğrulama → karşılaştırma için referans
    with torch.no_grad():
        v0 = run_epoch(model, processor, val_ds, device, cfg)
    print(f"Epoch 00 (zero-shot) | Val loss {v0['loss']:.4f} | mIoU {v0['iou']:.4f} | Dice {v0['dice']:.4f}")
    history = {"train_loss": [], "val_loss": [v0["loss"]], "val_iou": [v0["iou"]],
               "val_dice": [v0["dice"]], "config": cfg.to_dict()}

    best = float("inf")
    ckpt_path = cfg.path(cfg.ckpt_name)
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(len(train_ds))   # eskiden karıştırılmıyordu
        tr = run_epoch(model, processor, train_ds, device, cfg, optimizer, scaler, order)
        with torch.no_grad():
            va = run_epoch(model, processor, val_ds, device, cfg)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_iou"].append(va["iou"])
        history["val_dice"].append(va["dice"])
        msg = (f"Epoch {epoch:02d}/{cfg.epochs} | Train {tr['loss']:.4f} | Val {va['loss']:.4f}"
               f" | mIoU {va['iou']:.4f} | Dice {va['dice']:.4f}")
        if va["loss"] < best:
            best = va["loss"]
            save_decoder(model, ckpt_path, epoch=epoch, val_loss=va["loss"],
                         val_iou=va["iou"], val_dice=va["dice"], config=cfg.to_dict())
            msg += "  ✅ kaydedildi"
        print(msg)

    with open(cfg.path("coco_t_sam3_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, cfg.path("coco_t_sam3_curve.png"))
    print(f"\nBitti. En iyi val loss: {best:.4f} → {ckpt_path}")
    return history


def plot_history(history, save_path=None):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_tr = range(1, len(history["train_loss"]) + 1)
    ep_va = range(0, len(history["val_loss"]))
    a1.plot(ep_tr, history["train_loss"], label="Train", lw=2)
    a1.plot(ep_va, history["val_loss"], label="Val", lw=2)
    a1.set(xlabel="Epoch", ylabel="Loss (BCE + Dice)", title="Loss")
    a1.legend()
    a2.plot(ep_va, history["val_iou"], label="Val mIoU", lw=2)
    a2.plot(ep_va, history["val_dice"], label="Val Dice", lw=2)
    a2.set(xlabel="Epoch (0 = zero-shot)", title="Val metrikleri")
    a2.legend()
    fig.suptitle("SAM3 mask_decoder fine-tuning — Coco_T")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Grafik → {save_path}")
    return fig


if __name__ == "__main__":
    train()
