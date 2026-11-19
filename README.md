# Klasifikasi Fish Species Image Data: Simple_CNN vs Mobilenetv2 vs deit_tiny


Dataset: ["🐟🐟🐟Fish Species Image Data"]https://www.kaggle.com/datasets/sripaadsrinivasan/fish-species-image-data

## Struktur repo
```
train.py            # entry point (argparse)
src/dataset.py      # DataLoader per fold + pre-processing (CLAHE opsional) + augmentasi
src/model.py        # ConvNeXt-Tiny, Custom CNN, EfficientNet-B0, RegNetY-800MF
src/train_eval.py   # loop training, validasi, testing, 5-fold CV
src/utils.py        # metrik, plot learning curve / confusion matrix / ROC, ringkasan
Results/            # gambar & tabel hasil
```

## Metode
| Komponen | Pilihan |
|---|---|
| Model utama |  (pretrained ImageNet, head diganti 6 kelas) |
| Model pembanding | Custom CNN 4 blok konv (BatchNorm, Dropout 0.4), from scratch |
| Pre-processing | (a) Resize 224×224 + normalisasi ImageNet; (b) + CLAHE pada kanal L (LAB) |
| Augmentasi online | H/V flip, rotasi ±15°, color jitter (hanya data train) |
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Optimizer | AdamW (lr 5e-5 ConvNeXt / 1e-3 Custom CNN, weight decay 1e-2) + CosineAnnealing |
| Epoch / batch | 15 / 16, mixed precision (AMP) di GPU |
| Data train | Original `Train` + Augmented `Train` per fold |
| Validasi | Original `Valid` (checkpoint terbaik berdasar val loss) |
| Testing | Original `Test` (evaluasi akhir) |
| Evaluasi | 5-fold: Confusion Matrix, Accuracy, Precision, Recall, F1 (macro), ROC-AUC (OvR macro) |

`Valid` dan `Test` tidak pernah dipakai untuk training.

## Cara menjalankan
```bash
pip install -r requirements.txt

python train.py --model deit_tiny --data_dir $D --results_dir Results/deit_tiny
python train.py --model mobilenetv2 --data_dir $D --results_dir Results/mobilenetv2
python train.py --model simple_cnn    --data_dir $D --results_dir Results/simple_cnn
```
Tes cepat 1 fold: tambahkan `--fold 1 --epochs 2`.
Bobot model disimpan di `checkpoints/` dan di-ignore oleh git.

## Hasil (rerata ± std 5-fold, data Test)
> ISI setelah training selesai, salin dari `Results/<run>/<model>_5fold_metrics_summary.md`

| Metrik | Simple CNN | Mobilenetv2 | deit_tiny |
|---|---|---|---|
| Accuracy | 0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx |
| Precision (macro) | | | |
| Recall (macro) | | | |
| F1-score (macro) | | | |
| ROC-AUC (macro OvR) | | | |

### Gambar
> Ganti nama file sesuai hasil di folder `Results/`
![Confusion matrix fold 1](Results/convnext_tiny/convnext_tiny_confusion_matrix_fold1.png)
![Learning curve fold 1](Results/convnext_tiny/convnext_tiny_learning_curve_fold1.png)
![ROC fold 1](Results/convnext_tiny/convnext_tiny_roc_curve_fold1.png)

## Analisis
> ISI: kelas mana yang paling sering salah diprediksi (lihat confusion matrix), apakah ada overfitting (lihat learning curve), pengaruh CLAHE, dan perbandingan pretrained vs from scratch.
