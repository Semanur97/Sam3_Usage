"""CPU smoke testi: gerçek ağırlık/GPU/Drive gerekmez.

Küçük, rastgele başlatılmış bir Sam3Tracker modeli + sahte COCO veri seti (daireler) ile
eğitim → checkpoint → karşılaştırma zincirinin baştan sona çalıştığını kontrol eder.

    python -m pytest tests -q
"""
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from coco_t.config import Config
from coco_t.data import CocoSAMDataset
from coco_t.evaluate import compare
from coco_t.model import freeze_all_but_decoder, load_decoder, save_decoder
from coco_t.train import train

INPUT = 1008  # SAM3 girdi çözünürlüğü


class FakeProcessor:
    """Gerçek processor gibi: PIL → [1, 3, 1008, 1008] normalize edilmiş tensor."""

    def __call__(self, images, return_tensors="pt"):
        a = np.asarray(images.resize((INPUT, INPUT)), dtype=np.float32) / 255.0
        a = (a - 0.5) / 0.5
        return {"pixel_values": torch.from_numpy(a).permute(2, 0, 1)[None]}


def tiny_model():
    from transformers import Sam3TrackerConfig, Sam3TrackerModel

    cfg = Sam3TrackerConfig()
    bb = cfg.vision_config.backbone_config
    bb.hidden_size, bb.intermediate_size = 32, 64
    bb.num_hidden_layers, bb.num_attention_heads = 2, 2
    bb.global_attn_indexes = [1]
    torch.manual_seed(0)
    model = Sam3TrackerModel(cfg)
    freeze_all_but_decoder(model)
    return model


def make_coco(root, name, n_images, rng):
    """Her görüntüde birkaç beyaz daire (bubble), polygon + RLE karışık."""
    images, anns, aid = [], [], 1
    for i in range(n_images):
        w, h = 700, 450  # max_side=256 ile küçültmeyi de test eder
        img = Image.new("RGB", (w, h), (30, 30, 30))
        d = ImageDraw.Draw(img)
        for k in range(3):
            cx, cy, r = rng.integers(60, w - 60), rng.integers(60, h - 60), rng.integers(15, 50)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 220, 220))
            mm =Image.new("L", (w, h), 0)
            ImageDraw.Draw(mm).ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)
            m = np.asfortranarray(np.array(mm, np.uint8))
            if k % 2:  # RLE
                rle = coco_mask.encode(m)
                rle["counts"] = rle["counts"].decode()
                seg = rle
            else:      # polygon
                t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
                seg = [list(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1).ravel().astype(float))]
            anns.append({"id": aid, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "bbox": [float(cx - r), float(cy - r), float(2 * r), float(2 * r)],
                         "area": float(np.pi * r * r), "segmentation": seg})
            aid += 1
        fn = f"{name}_{i}.png"
        img.save(os.path.join(root, fn))
        images.append({"id": i, "file_name": fn, "width": w, "height": h})
    path = os.path.join(root, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "bubble"}]}, f)
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("coco_t"))
    rng = np.random.default_rng(0)
    make_coco(root, "train", 2, rng)
    make_coco(root, "val", 1, rng)
    cfg = Config(base_dir=root, train_ann="train.json", val_ann="val.json", save_dir=root,
                 max_side=256, epochs=2, box_chunk=2, amp=False)
    return cfg, tiny_model(), FakeProcessor()


def test_dataset_shapes(setup):
    cfg, _, _ = setup
    ds = CocoSAMDataset(cfg.train_ann_path, cfg.base_dir, cfg.max_side)
    s = ds[0]
    H, W = s["size"]
    assert max(H, W) == 256
    assert s["masks"].shape == (3, H, W) and s["boxes"].shape == (3, 4)
    assert s["masks"].sum() > 0 and set(np.unique(s["masks"])) <= {0.0, 1.0}
    assert (s["boxes"][:, 2] <= W + 1e-3).all() and (s["boxes"][:, 3] <= H + 1e-3).all()


def test_train_and_compare(setup):
    cfg, model, proc = setup
    enc_before = {k: v.clone() for k, v in model.vision_encoder.state_dict().items()}
    dec_before = {k: v.clone() for k, v in model.mask_decoder.state_dict().items()}

    hist = train(cfg, model=model, processor=proc)
    assert len(hist["train_loss"]) == cfg.epochs
    assert len(hist["val_loss"]) == cfg.epochs + 1          # + epoch 0 (zero-shot)
    assert all(np.isfinite(hist["train_loss"]))
    assert os.path.exists(cfg.path(cfg.ckpt_name))
    assert os.path.exists(cfg.path("coco_t_sam3_curve.png"))

    # encoder donmuş kalmalı, decoder değişmeli
    for k, v in model.vision_encoder.state_dict().items():
        assert torch.equal(v, enc_before[k]), k
    assert any(not torch.equal(v, dec_before[k]) for k, v in model.mask_decoder.state_dict().items())

    # compare: aynı model nesnesi; önce orijinal decoder'a geri dön (zero-shot'u taklit)
    model.mask_decoder.load_state_dict(dec_before)
    res = compare(cfg, model=model, processor=proc)
    for key in ("zero_shot", "fine_tuned"):
        assert 0.0 <= res[key]["miou"] <= 1.0
    assert os.path.exists(cfg.path("coco_t_eval.json"))
    assert os.path.exists(cfg.path("coco_t_sam3_comparison.png"))


def test_checkpoint_roundtrip_and_legacy(setup, tmp_path):
    _, model, _ = setup
    p = tmp_path / "dec.pt"
    save_decoder(model, p, epoch=3)
    assert load_decoder(model, p)["epoch"] == 3
    legacy = tmp_path / "legacy.pt"                         # eski notebook formatı
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, legacy)
    assert load_decoder(model, legacy)["epoch"] == 1


def test_chunking_matches_single_pass(setup):
    """Kutuları parça parça işlemek, hepsini tek seferde işlemekle aynı sonucu vermeli."""
    from coco_t.model import boxes_to_model_space, encode_image, predict_logits

    cfg, model, proc = setup
    s = CocoSAMDataset(cfg.val_ann_path, cfg.base_dir, cfg.max_side)[0]
    model.eval()
    with torch.no_grad():
        emb, sc = encode_image(model, proc, s["image"], "cpu")
        b = boxes_to_model_space(s["boxes"], sc)
        full = predict_logits(model, emb, b, s["size"])
        parts = torch.cat([predict_logits(model, emb, b[i:i + 1], s["size"]) for i in range(len(b))])
    assert torch.allclose(full, parts, atol=1e-4)
