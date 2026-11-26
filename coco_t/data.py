"""Coco_T veri seti: COCO JSON (polygon veya RLE) → görüntü + kutular + maskeler."""
import os

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def decode_segmentation(seg, h: int, w: int) -> np.ndarray:
    """COCO segmentation (polygon / RLE / sıkıştırılmamış RLE) → [h, w] uint8 maske."""
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), str):
            m = coco_mask.decode(seg)
        else:
            m = coco_mask.decode(coco_mask.frPyObjects(seg, h, w))
    else:
        m = coco_mask.decode(coco_mask.frPyObjects(seg, h, w))
    if m.ndim == 3:  # birden fazla polygon parçası
        m = m.any(axis=2)
    return m.astype(np.uint8)


class CocoSAMDataset(Dataset):
    """Her örnek bir görüntü ve içindeki tüm instance'lar.

    Döndürülen sözlük:
        image : PIL.Image (RGB), uzun kenarı en fazla `max_side`
        boxes : [N, 4] float32, xyxy, küçültülmüş görüntü koordinatlarında
        masks : [N, H, W] float32 (0/1)
        size  : (H, W)
        image_id : COCO görüntü id'si
    """

    def __init__(self, ann_file: str, img_dir: str, max_side: int | None = 512):
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.img_dir = img_dir
        self.max_side = max_side

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.imgs[img_id]
        h, w = info["height"], info["width"]

        image = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")

        scale = 1.0
        if self.max_side:
            scale = min(self.max_side / max(h, w), 1.0)
        new_w, new_h = (int(w * scale), int(h * scale)) if scale < 1.0 else (w, h)
        if scale < 1.0:
            image = image.resize((new_w, new_h), Image.BILINEAR)

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, masks = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            m = decode_segmentation(ann["segmentation"], h, w)
            if scale < 1.0:
                m = np.array(Image.fromarray(m).resize((new_w, new_h), Image.NEAREST))
                x, y, bw, bh = x * scale, y * scale, bw * scale, bh * scale
            boxes.append([x, y, x + bw, y + bh])
            masks.append(m.astype(np.float32))

        return {
            "image": image,
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32),
            "masks": np.stack(masks) if masks else np.zeros((0, new_h, new_w), np.float32),
            "size": (new_h, new_w),
            "image_id": img_id,
        }
