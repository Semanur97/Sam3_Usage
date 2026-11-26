# Sam3_Usage — PhD projesi notları

## Ne bu?
SAM3'ün (`facebook/sam3`, transformers'ta `Sam3TrackerModel`, box-prompt) sadece
`mask_decoder`'ını fine-tune ederek **Coco_T** veri setinde bubble (kabarcık) instance
segmentation. Vision encoder donmuş. Ana odak Coco_T; `Sam3_trials.ipynb` eski denemeler
(Kvasir-SEG vb.) içindir, dokunma.

## Çalışma düzeni
- Kod `coco_t/` paketinde. Kullanıcı kodu **Google Colab'da (L4 GPU)** çalıştırır:
  `notebooks/colab_coco_t.ipynb` repoyu `git pull` ile çeker ve `train(cfg)` / `compare(cfg)` çağırır.
- Kullanıcının yerel bilgisayarında kod çalışmaz. Değişiklikler bir branch'te yapılır ve PR açılır.
- Notebook ince kalmalı: mantık `.py` dosyalarına gider, notebook sadece çağırır.
- Veri ve çıktılar Google Drive'da: `/content/drive/MyDrive/PhD/Coco_T` (görüntüler +
  `instances_train.json`, `instances_val.json`, COCO formatı, polygon/RLE). Çıktılar `/content/drive/MyDrive/PhD`.
  Checkpoint ve veri repoya girmez (`*.pt` gitignore'da).

## Dosyalar
- `coco_t/config.py` — tüm hiperparametreler ve yollar (`Config` dataclass)
- `coco_t/data.py` — `CocoSAMDataset`: görüntüyü `max_side`'a küçültür, kutu/maske döner
- `coco_t/model.py` — yükleme, dondurma, `encode_image` (görüntü başına 1 kez), `predict_logits`, checkpoint (sadece decoder)
- `coco_t/losses.py` — BCE + Dice (instance başına), IoU/Dice metrikleri
- `coco_t/train.py` — eğitim döngüsü; epoch 0 = zero-shot referans
- `coco_t/evaluate.py` — zero-shot vs fine-tuned, **aynı GT kutularıyla**, tüm val seti

## Test
GPU ve gerçek ağırlık olmadan, küçük rastgele model + sahte COCO verisiyle:
```
python -m pytest tests -q
```
Her değişiklikten sonra çalıştır. Yeni davranış eklenirse `tests/test_smoke.py`'ye test ekle.

## Kurallar
- Kullanıcı Türkçe konuşur; print mesajları ve yorumlar Türkçe.
- Loss/metrik tanımını değiştirirsen PR açıklamasında belirt (tez sonuçlarını etkiler).
- Eski checkpoint formatı (`model_state_dict`) `load_decoder` ile yüklenebilir kalmalı.
