"""SAM3 (box-prompt / tracker) yükleme, dondurma, tahmin ve checkpoint yardımcıları.

Önemli hız notu: vision encoder donduruldugu için görüntü embedding'i görüntü başına
BİR kez hesaplanır, sonra tüm kutular mask_decoder'dan parça parça (box_chunk) geçer.
Eski notebook her kutu için encoder'ı baştan çalıştırıyordu (90 kutu = 90 encoder geçişi).
"""
import numpy as np
import torch
import torch.nn.functional as F


def load_model(model_id: str = "facebook/sam3", device="cuda", token=None):
    """Model + processor yükler, sadece mask_decoder'ı eğitilebilir bırakır."""
    from transformers import AutoModelForMaskGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model = AutoModelForMaskGeneration.from_pretrained(model_id, token=token).to(device)
    freeze_all_but_decoder(model)
    return model, processor


def freeze_all_but_decoder(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def set_train_mode(model, training: bool):
    """Encoder hep eval modunda kalır; sadece decoder train/eval arasında geçer."""
    model.eval()
    if training:
        model.mask_decoder.train()


@torch.no_grad()
def encode_image(model, processor, pil_img, device):
    """Görüntüyü encoder'dan bir kez geçirir.

    Returns:
        image_embeddings: list[Tensor]  (forward'a image_embeddings= olarak verilir)
        scale: (sx, sy)  küçültülmüş görüntü pikselinden model girdisi pikseline çarpan
    """
    pixel_values = processor(images=pil_img, return_tensors="pt")["pixel_values"].to(device)
    _, _, h_res, w_res = pixel_values.shape
    sx, sy = w_res / pil_img.width, h_res / pil_img.height
    image_embeddings = model.get_image_embeddings(pixel_values)
    return image_embeddings, (sx, sy)


def boxes_to_model_space(boxes: np.ndarray, scale) -> torch.Tensor:
    sx, sy = scale
    b = torch.as_tensor(boxes, dtype=torch.float32).clone()
    b[:, [0, 2]] *= sx
    b[:, [1, 3]] *= sy
    return b


def predict_logits(model, image_embeddings, boxes_model: torch.Tensor, out_size):
    """Bir grup kutu için maske logit'leri → [n, H, W] (out_size çözünürlüğünde)."""
    out = model(
        image_embeddings=image_embeddings,
        input_boxes=boxes_model.unsqueeze(0),   # [1, n, 4]
        multimask_output=False,
    )
    pred = out.pred_masks[0, :, 0]               # [n, h_low, w_low]
    return F.interpolate(pred.unsqueeze(1).float(), size=out_size,
                         mode="bilinear", align_corners=False).squeeze(1)


@torch.no_grad()
def predict_masks(model, processor, pil_img, boxes, out_size, device,
                  box_chunk=16, threshold=0.5, amp=True):
    """Kutu-prompt'lu çıkarım → [N, H, W] bool numpy."""
    if len(boxes) == 0:
        return np.zeros((0, *out_size), dtype=bool)
    emb, scale = encode_image(model, processor, pil_img, device)
    bm = boxes_to_model_space(boxes, scale).to(device)
    outs = []
    use_amp = amp and torch.cuda.is_available()
    for i in range(0, len(bm), box_chunk):
        with torch.autocast("cuda", enabled=use_amp):
            logits = predict_logits(model, emb, bm[i:i + box_chunk], out_size)
        outs.append((torch.sigmoid(logits) > threshold).cpu().numpy())
    return np.concatenate(outs)


# ── Checkpoint ──────────────────────────────────────────────
def save_decoder(model, path, **meta):
    """Sadece mask_decoder'ı kaydeder (~17 MB; tam model ~1.8 GB idi)."""
    torch.save({"mask_decoder": model.mask_decoder.state_dict(), **meta}, path)


def load_decoder(model, path, device="cpu"):
    """Yeni (sadece decoder) veya eski (tam model_state_dict) checkpoint'i yükler."""
    ckpt = torch.load(path, map_location=device)
    if "mask_decoder" in ckpt:
        model.mask_decoder.load_state_dict(ckpt["mask_decoder"])
    elif "model_state_dict" in ckpt:  # eski notebook formatı
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Tanınmayan checkpoint formatı: {list(ckpt)[:5]}")
    return {k: v for k, v in ckpt.items() if k not in ("mask_decoder", "model_state_dict")}
