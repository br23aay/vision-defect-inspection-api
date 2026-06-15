"""
train.py
--------
Trains a MobileNetV2 binary image classifier (PASS / FAIL) using
TensorFlow transfer learning. Runs entirely on CPU — no GPU required.

Architecture:
  MobileNetV2 (ImageNet weights, frozen) → GlobalAveragePooling → Dropout → Dense(1, sigmoid)

Usage:
    python model/train.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/images"
MODEL_PATH  = "model/defect_model.keras"
METRICS_PATH = "model/training_metrics.json"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16       # Small batch for CPU
EPOCHS_FROZEN  = 5     # Train head only
EPOCHS_FINETUNE = 3    # Fine-tune last 20 MobileNetV2 layers
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_model():
    """MobileNetV2 + custom classification head."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze base initially

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Binary: PASS=0, FAIL=1

    model = models.Model(inputs, outputs)
    return model, base


def get_data_generators():
    """Image data generators with augmentation for training."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        seed=SEED,
    )

    val_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        seed=SEED,
    )

    return train_gen, val_gen


def train():
    print("=" * 60)
    print("Vision Defect Inspection — Model Training")
    print("TensorFlow", tf.__version__, "| CPU mode")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────
    train_gen, val_gen = get_data_generators()
    class_indices = train_gen.class_indices
    print(f"\nClasses: {class_indices}")
    print(f"Training samples:   {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    # ── Phase 1: Train head only ─────────────────────────────────────────────
    print(f"\n[Phase 1] Training classification head ({EPOCHS_FROZEN} epochs)...")
    model, base = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        verbose=1,
    )

    # ── Phase 2: Fine-tune last 20 layers ────────────────────────────────────
    print(f"\n[Phase 2] Fine-tuning last 20 MobileNetV2 layers ({EPOCHS_FINETUNE} epochs)...")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # Lower LR for fine-tuning
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        verbose=1,
    )

    # ── Save model ───────────────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # ── Save class map ───────────────────────────────────────────────────────
    class_map = {v: k for k, v in class_indices.items()}  # {0: 'fail', 1: 'pass'}
    with open("model/class_map.json", "w") as f:
        json.dump(class_map, f)

    # ── Save training metrics ─────────────────────────────────────────────────
    all_acc  = history1.history["accuracy"]  + history2.history["accuracy"]
    all_val  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    all_loss = history1.history["loss"] + history2.history["loss"]

    metrics = {
        "final_train_accuracy": round(float(all_acc[-1]), 4),
        "final_val_accuracy":   round(float(all_val[-1]), 4),
        "final_loss":           round(float(all_loss[-1]), 4),
        "total_epochs":         EPOCHS_FROZEN + EPOCHS_FINETUNE,
        "class_map":            class_map,
        "img_size":             list(IMG_SIZE),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Final train accuracy: {metrics['final_train_accuracy']:.2%}")
    print(f"  Final val accuracy:   {metrics['final_val_accuracy']:.2%}")
    print(f"  Metrics saved → {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train()
