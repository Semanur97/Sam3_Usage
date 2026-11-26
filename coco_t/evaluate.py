"""Zero-shot SAM3 vs fine-tuned SAM3 — AYNI GT kutularıyla, tüm val seti üzerinde.

Eski notebook zero-shot'u otomatik maske üretimiyle (kutusuz) çalıştırıp fine-tuned'ı GT
kutularıyla çalıştırıyordu; bu adil bir karşılaştırma değil. Burada ikisi de aynı kutuları alır,
tek fark mask_decoder ağırlıkları.

Colab'dan:
    from coco_t.evaluate import compare
    results = compare(Config())
"""
import csv
import json

import numpy as np
import torch

from .config import Config
from .data import CocoSAMDataset
from .losses import instance_iou_dice
from .model import load_decoder, load_model, predict_masks


def evaluate(model, processor, dataset, device, cfg: Config, desc=""):
    """Her görüntü için mIoU/Dice; ayrıca tüm instance'lar üzerinden genel ortalama."""
    from tqdm.auto import tqdm

    model.eval()
    rows, all_iou, all_dice = [], [], []
    for idx in tqdm(range(len(dataset)), leave=False, desc=desc):
        s = dataset[idx]
        if len(s["boxes"]) == 0:
            continue
        pred = predict_masks(model, processor, s["image"], s["boxes"], s["size"], device,
                             cfg.box_chunk, cfg.mask_threshold, cfg.amp)
        iou, dice = instance_iou_dice(pred, s["masks"])
        rows.append({"image_id": s["image_id"], "n_instances": len(iou),
                     "miou": float(iou.mean()), "dice": float(dice.mean())})
        all_iou.extend(iou)
        all_dice.extend(dice)
    return {"miou": float(np.mean(all_iou)), "dice": float(np.mean(all_dice)),
            "n_instances": len(all_iou), "per_image": rows}


def compare(cfg: Config = None, ckpt_path=None, model=None, processor=None, val_ds=None,
            vis_index=0):
    cfg = cfg or Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or cfg.path(cfg.ckpt_name)
    if model is None:
        model, processor = load_model(cfg.model_id, device)
    model.to(device)
    val_ds = val_ds or CocoSAMDataset(cfg.val_ann_path, cfg.base_dir, cfg.max_side)

    # 1) Zero-shot (orijinal ağırlıklar)
    with torch.no_grad():
        zs = evaluate(model, processor, val_ds, device, cfg, "Zero-shot")
        s = val_ds[vis_index]
        zs_vis = predict_masks(model, processor, s["image"], s["boxes"], s["size"], device,
                               cfg.box_chunk, cfg.mask_threshold, cfg.amp)

    # 2) Fine-tuned (aynı model, sadece decoder ağırlıkları değişir)
    meta = load_decoder(model, ckpt_path, device)
    print(f"Checkpoint: epoch={meta.get('epoch')} val_loss={meta.get('val_loss')}")
    with torch.no_grad():
        ft = evaluate(model, processor, val_ds, device, cfg, "Fine-tuned")
        ft_vis = predict_masks(model, processor, s["image"], s["boxes"], s["size"], device,
                               cfg.box_chunk, cfg.mask_threshold, cfg.amp)

    print(f"\n{'':12s}{'mIoU':>8s}{'Dice':>8s}   ({zs['n_instances']} instance, {len(val_ds)} görüntü)")
    print(f"{'Zero-shot':12s}{zs['miou']:8.4f}{zs['dice']:8.4f}")
    print(f"{'Fine-tuned':12s}{ft['miou']:8.4f}{ft['dice']:8.4f}")

    # Kaydet
    with open(cfg.path("coco_t_eval.json"), "w") as f:
        json.dump({"zero_shot": {k: v for k, v in zs.items() if k != "per_image"},
                   "fine_tuned": {k: v for k, v in ft.items() if k != "per_image"},
                   "checkpoint": ckpt_path, "checkpoint_meta": meta}, f, indent=2, default=str)
    with open(cfg.path("coco_t_eval_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "n_instances", "zs_miou", "zs_dice", "ft_miou", "ft_dice"])
        for a, b in zip(zs["per_image"], ft["per_image"]):
            w.writerow([a["image_id"], a["n_instances"], f"{a['miou']:.4f}", f"{a['dice']:.4f}",
                        f"{b['miou']:.4f}", f"{b['dice']:.4f}"])

    plot_comparison(s, zs_vis, ft_vis, zs, ft, cfg.path("coco_t_sam3_comparison.png"))
    return {"zero_shot": zs, "fine_tuned": ft}


def plot_comparison(sample, zs_masks, ft_masks, zs_res, ft_res, save_path=None, alpha=0.45):
    import matplotlib.pyplot as plt

    img = np.array(sample["image"])
    H, W = img.shape[:2]

    def overlay(ax, masks, title, color=None):
        ax.imshow(img)
        if len(masks):
            ov = np.zeros((H, W, 4), np.float32)
            cols = plt.cm.tab20(np.linspace(0, 1, 20))
            for i, m in enumerate(masks):
                c = color if color is not None else cols[i % 20][:3]
                ov[m > 0] = [*c, alpha]
            ax.imshow(ov)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    overlay(axes[0], sample["masks"], f"Ground Truth\n{len(sample['masks'])} bubble", (1, .2, .2))
    overlay(axes[1], zs_masks, f"SAM3 zero-shot (GT kutular)\nval mIoU {zs_res['miou']:.3f}")
    overlay(axes[2], ft_masks, f"SAM3 fine-tuned (GT kutular)\nval mIoU {ft_res['miou']:.3f}")
    fig.suptitle("Bubble instance segmentation — SAM3 zero-shot vs fine-tuned", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Görsel → {save_path}")
    return fig


if __name__ == "__main__":
    compare()
