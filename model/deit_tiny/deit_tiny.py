
# ==========================================
# MODEL DEIT-TINY
# Vision Transformer untuk Klasifikasi Ikan
# ==========================================

import torch
import torch.nn as nn
import timm

from torch.optim import AdamW


# ==========================================
# 1. KONFIGURASI MODEL
# ==========================================

MODEL_NAME = "DeiT-Tiny"
MODEL_ARCHITECTURE = "deit_tiny_patch16_224"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

PRETRAINED = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. MEMBUAT ARSITEKTUR MODEL
# ==========================================

def create_model(num_classes):

    model = timm.create_model(
        MODEL_ARCHITECTURE,
        pretrained=PRETRAINED,
        num_classes=num_classes
    )

    # Bekukan seluruh parameter backbone
    for param in model.parameters():
        param.requires_grad = False

    # Latih classifier head
    for param in model.head.parameters():
        param.requires_grad = True

    # Jika tersedia, latih distillation head juga
    if hasattr(model, "head_dist"):
        for param in model.head_dist.parameters():
            param.requires_grad = True

    return model.to(DEVICE)


# ==========================================
# 3. LOSS FUNCTION
# ==========================================

def create_loss_function():

    return nn.CrossEntropyLoss()


# ==========================================
# 4. OPTIMIZER
# ==========================================

def create_optimizer(model):

    trainable_parameters = filter(
        lambda param: param.requires_grad,
        model.parameters()
    )

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    return optimizer


# ==========================================
# 5. INFORMASI MODEL
# ==========================================

def show_model_config():

    print("========================================")
    print("          KONFIGURASI DEIT-TINY")
    print("========================================")

    print("Nama Model       :", MODEL_NAME)
    print("Arsitektur       :", MODEL_ARCHITECTURE)
    print("Image Size       :", IMAGE_SIZE)
    print("Batch Size       :", BATCH_SIZE)
    print("Epoch            :", EPOCHS)
    print("Learning Rate    :", LEARNING_RATE)
    print("Weight Decay     :", WEIGHT_DECAY)
    print("Pretrained       :", PRETRAINED)
    print("Device           :", DEVICE)

    print("========================================")


# ==========================================
# 6. INFORMASI PARAMETER
# ==========================================

def show_model_summary(model):

    total_parameters = sum(
        param.numel()
        for param in model.parameters()
    )

    trainable_parameters = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    print("Total Parameters     :", f"{total_parameters:,}")
    print("Trainable Parameters :", f"{trainable_parameters:,}")


# ==========================================
# 7. TEST MODEL
# ==========================================

if __name__ == "__main__":

    show_model_config()

    # Contoh jumlah kelas dataset
    NUM_CLASSES = 483

    model = create_model(NUM_CLASSES)

    print("\nMODEL BERHASIL DIBUAT")

    show_model_summary(model)
