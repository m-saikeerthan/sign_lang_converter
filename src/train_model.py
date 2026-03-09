"""
Train the ISL gesture classifier model.
Loads landmarks, engineers features, trains improved Dense neural net,
exports to both .h5 and .tflite.

Improvements over v1:
  - Extended features (distances + angles): 42 → 60 features
  - Deeper architecture with residual connection
  - Rotation + scaling augmentation
  - Standardization (mean + std)
  - Learning rate scheduling (ReduceLROnPlateau)
"""

import os
import sys
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Add src to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LANDMARKS_DIR, PROJECT_ROOT, MODEL_H5_PATH, MODEL_TFLITE_PATH,
    LABELS_PATH, MEAN_PATH, STD_PATH, NUM_FEATURES, NUM_EXTENDED_FEATURES,
    MIN_SAMPLES_PER_CLASS, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, EARLY_STOP_PATIENCE
)
from feature_engineer import compute_extended_features

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Add, Activation
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ─── Augmentation Functions ──────────────────────────────────────────

def mirror_landmarks(X_raw):
    """Flip x-coordinates to simulate opposite hand.
    Works on raw 42-coord data BEFORE feature engineering."""
    X_mirror = X_raw.copy()
    X_mirror[:, 0::2] *= -1  # negate all x values
    return X_mirror


def augment_noise(X, scale=0.02, n_aug=1):
    """Add small random noise augmentations."""
    augmented = [X]
    for _ in range(n_aug):
        noise = np.random.normal(0, scale, X.shape).astype(np.float32)
        augmented.append(X + noise)
    return np.concatenate(augmented, axis=0)


def augment_rotation(X_raw, max_angle_deg=15):
    """Rotate all landmarks by a random angle (operates on raw 42-coord data).
    Each sample gets a different random rotation."""
    X_rot = X_raw.copy()
    n = len(X_rot)
    angles = np.random.uniform(-max_angle_deg, max_angle_deg, n) * np.pi / 180

    for i in range(n):
        cos_a = np.cos(angles[i])
        sin_a = np.sin(angles[i])
        for j in range(21):  # 21 landmarks
            x = X_rot[i, j * 2]
            y = X_rot[i, j * 2 + 1]
            X_rot[i, j * 2] = x * cos_a - y * sin_a
            X_rot[i, j * 2 + 1] = x * sin_a + y * cos_a

    return X_rot


def augment_scale(X_raw, min_scale=0.9, max_scale=1.1):
    """Scale all landmarks by a random factor (operates on raw 42-coord data)."""
    X_sc = X_raw.copy()
    scales = np.random.uniform(min_scale, max_scale, len(X_sc))
    for i in range(len(X_sc)):
        X_sc[i] *= scales[i]
    return X_sc


# ─── Model Builder ───────────────────────────────────────────────────

def build_model(num_features, num_classes):
    """Build improved Dense model with residual connection."""
    inputs = Input(shape=(num_features,))

    # First block
    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Second block with residual
    x1 = Dense(128, activation="relu")(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)

    # Residual path: project x to 128 dims to match
    x_skip = Dense(128)(x)
    x = Add()([x1, x_skip])
    x = Activation("relu")(x)

    # Third block
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Fourth block
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Output
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ─── Training Pipeline ───────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  ISL Model Training — v2 (Accuracy Improved)")
    print("=" * 60)

    # ── 1. Load raw landmark data ─────────────────────────────────
    print("\n[ISL] Loading landmark data...")

    X_raw = []
    y = []

    for file in sorted(os.listdir(LANDMARKS_DIR)):
        if not file.endswith(".npy"):
            continue
        word = file.replace(".npy", "")
        data = np.load(os.path.join(LANDMARKS_DIR, file))

        # Filter to correct feature size
        if data.ndim != 2 or data.shape[1] != NUM_FEATURES:
            print(f"  Skipping {word}: shape {data.shape} (expected ?, {NUM_FEATURES})")
            continue

        # Balance classes
        data = data[:MIN_SAMPLES_PER_CLASS]

        for sample in data:
            X_raw.append(sample)
            y.append(word)

        print(f"  {word}: {len(data)} samples")

    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y)

    print(f"\n[ISL] Total: {len(X_raw)} samples, {len(set(y))} classes")

    # ── 2. Mirror augmentation (on raw 42 coords) ────────────────
    print("[ISL] Adding mirrored (hand-flip) data...")
    X_mirror = mirror_landmarks(X_raw)
    X_raw = np.concatenate([X_raw, X_mirror], axis=0)
    y = np.concatenate([y, y], axis=0)
    print(f"[ISL] After mirroring: {len(X_raw)} samples")

    # ── 3. Rotation + Scale augmentation (on raw 42 coords) ──────
    print("[ISL] Adding rotation + scale augmented data...")
    X_rot = augment_rotation(X_raw, max_angle_deg=15)
    X_sc = augment_scale(X_raw, min_scale=0.9, max_scale=1.1)
    X_raw_aug = np.concatenate([X_raw, X_rot, X_sc], axis=0)
    y_aug = np.concatenate([y, y, y], axis=0)
    print(f"[ISL] After rotation+scale: {len(X_raw_aug)} samples")

    # ── 4. Feature engineering (42 → 60 features) ────────────────
    print("[ISL] Computing extended features (distances + angles)...")
    X = compute_extended_features(X_raw_aug)
    print(f"[ISL] Feature shape: {X.shape} ({NUM_FEATURES} raw → {X.shape[1]} extended)")

    # ── 5. Encode labels ─────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_aug)
    y_categorical = to_categorical(y_encoded)
    np.save(LABELS_PATH, le.classes_)
    print(f"[ISL] Labels saved: {list(le.classes_)}")

    # ── 6. Standardization (mean + std) ──────────────────────────
    print("[ISL] Standardizing features (mean + std)...")
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8  # avoid division by zero
    X = (X - mean) / std
    np.save(MEAN_PATH, mean)
    np.save(STD_PATH, std)
    print(f"[ISL] Mean and std saved")

    # ── 7. Noise augmentation (on extended features) ─────────────
    print("[ISL] Adding noise augmentation...")
    X_noise = augment_noise(X, scale=0.02, n_aug=1)
    y_noise = np.concatenate([y_categorical, y_categorical], axis=0)
    y_enc_noise = np.concatenate([y_encoded, y_encoded], axis=0)
    print(f"[ISL] After noise: {len(X_noise)} samples")

    # ── 8. Train/val split ───────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_noise, y_noise,
        test_size=0.2,
        random_state=42,
        stratify=y_enc_noise
    )
    print(f"[ISL] Train: {len(X_train)}, Val: {len(X_val)}")

    # ── 9. Build model ───────────────────────────────────────────
    num_classes = len(le.classes_)
    model = build_model(NUM_EXTENDED_FEATURES, num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ── 10. Callbacks ────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # ── 11. Train ────────────────────────────────────────────────
    print("\n[ISL] Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    # ── 12. Save Keras model ─────────────────────────────────────
    model.save(MODEL_H5_PATH)
    print(f"\n[ISL] Keras model saved: {MODEL_H5_PATH}")

    # ── 13. Export TFLite ────────────────────────────────────────
    print("[ISL] Converting to TFLite...")
    try:
        saved_model_dir = os.path.join(PROJECT_ROOT, "_temp_saved_model")
        model.export(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(MODEL_TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        tflite_size = os.path.getsize(MODEL_TFLITE_PATH) / 1024
        print(f"[ISL] TFLite model saved: {MODEL_TFLITE_PATH} ({tflite_size:.1f} KB)")
    except Exception as e:
        print(f"[ISL] TFLite export failed: {e}")
        print("[ISL] You can convert manually with: python src/convert_to_tflite.py")

    # ── 14. Validation report ────────────────────────────────────
    print("\n[ISL] Validation Results:")
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_labels = np.argmax(y_val, axis=1)
    y_pred_labels = np.argmax(y_val_pred, axis=1)
    print(classification_report(y_val_labels, y_pred_labels,
                                target_names=le.classes_))

    print("=" * 60)
    print(f"  Training complete! Classes: {list(le.classes_)}")
    print(f"  Features: {NUM_EXTENDED_FEATURES} (42 raw + 18 engineered)")
    print("=" * 60)


if __name__ == "__main__":
    train()
