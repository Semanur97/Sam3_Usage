"""Tüm ayarlar tek yerde. Colab'da değiştirmek için:

    from coco_t.config import Config
    cfg = Config(epochs=20, lr=5e-5)
"""
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ── Yollar (Google Drive) ────────────────────────────────
    base_dir: str = "/content/drive/MyDrive/PhD/Coco_T"   # görüntüler + COCO json
    train_ann: str = "instances_train.json"               # base_dir'e göre
    val_ann: str = "instances_val.json"
    save_dir: str = "/content/drive/MyDrive/PhD"          # checkpoint, grafik, sonuç

    # ── Model ────────────────────────────────────────────────
    model_id: str = "facebook/sam3"
    ckpt_name: str = "coco_t_sam3_decoder_best.pt"        # sadece mask_decoder ağırlıkları

    # ── Veri ────────────────────────────────────────────────
    max_side: int = 512        # VRAM için görüntünün uzun kenarı bu boyuta küçültülür

    # ── Eğitim ──────────────────────────────────────────────
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    grad_clip: float = 1.0
    box_chunk: int = 16        # bir forward'da işlenen kutu sayısı (OOM olursa düşür)
    amp: bool = True           # mixed precision (sadece GPU'da etkin)
    seed: int = 42

    # ── Değerlendirme ───────────────────────────────────────
    mask_threshold: float = 0.5

    def path(self, name: str) -> str:
        return os.path.join(self.save_dir, name)

    @property
    def train_ann_path(self) -> str:
        return os.path.join(self.base_dir, self.train_ann)

    @property
    def val_ann_path(self) -> str:
        return os.path.join(self.base_dir, self.val_ann)

    def to_dict(self) -> dict:
        return asdict(self)
