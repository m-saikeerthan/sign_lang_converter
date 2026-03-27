"""
Train the ISL gesture classifier model.
Loads landmarks, engineers two-hand features, trains improved Dense neural net,
exports to both .h5 and .tflite.

Supports both old single-hand (42-dim) and new two-hand (84-dim) data.
Single-hand data is automatically zero-padded for the second hand.
"""

import os
import sys
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LANDMARKS_DIR, PROJECT_ROOT, MODEL_H5_PATH, MODEL_TFLITE_PATH,
    LABELS_PATH, MEAN_PATH, STD_PATH, NUM_FEATURES,
    NUM_TWO_HAND_RAW, NUM_TWO_HAND_EXTENDED,
    MIN_SAMPLES_PER_CLASS, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, EARLY_STOP_PATIENCE
)
from feature_engineer import compute_two_hand_features

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Add, Activation
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ─── Augmentation ────────────────────────────────────────────────────

def mirror_landmarks(X_raw):
    """Flip x-coordinates for both hands."""
    X_mirror = X_raw.copy()
    X_mirror[:, 0::2] *= -1
    return X_mirror


def augment_noise(X, scale=0.02, n_aug=1):
    augmented = [X]
    for _ in range(n_aug):
        noise = np.random.normal(0, scale, X.shape).astype(np.float32)
        augmented.append(X + noise)
    return np.concatenate(augmented, axis=0)


def augment_rotation(X_raw, max_angle_deg=15):
    """Rotate landmarks (works on 84-dim two-hand data)."""
    X_rot = X_raw.copy()
    n = len(X_rot)
    num_landmarks = X_rot.shape[1] // 2  # 42 landmarks (21 per hand × 2)
    angles = np.random.uniform(-max_angle_deg, max_angle_deg, n) * np.pi / 180

    for i in range(n):
        cos_a = np.cos(angles[i])
        sin_a = np.sin(angles[i])
        for j in range(num_landmarks):
            x = X_rot[i, j * 2]
            y = X_rot[i, j * 2 + 1]
            X_rot[i, j * 2] = x * cos_a - y * sin_a
            X_rot[i, j * 2 + 1] = x * sin_a + y * cos_a
    return X_rot


def augment_scale(X_raw, min_scale=0.9, max_scale=1.1):
    X_sc = X_raw.copy()
    scales = np.random.uniform(min_scale, max_scale, len(X_sc))
    for i in range(len(X_sc)):
        X_sc[i] *= scales[i]
    return X_sc


def pad_to_two_hand(data):
    """Pad single-hand (42-dim) data to two-hand (84-dim) with zeros."""
    if data.shape[1] == NUM_TWO_HAND_RAW:
        return data
    elif data.shape[1] == NUM_FEATURES:
        zeros = np.zeros((data.shape[0], NUM_FEATURES), dtype=np.float32)
        return np.concatenate([data, zeros], axis=1)
    else:
        return None


# ─── Model ───────────────────────────────────────────────────────────

def build_model(num_features, num_classes):
    inputs = Input(shape=(num_features,))

    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x1 = Dense(128, activation="relu")(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)

    x_skip = Dense(128)(x)
    x = Add()([x1, x_skip])
    x = Activation("relu")(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


# ─── Training Pipeline ───────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  ISL Model Training — v3 (Two-Hand Support)")
    print("=" * 60)

    # 1. Load raw landmark data
    print("\n[ISL] Loading landmark data...")
    X_raw = []
    y = []

    for file in sorted(os.listdir(LANDMARKS_DIR)):
        if not file.endswith(".npy"):
            continue
        word = file.replace(".npy", "")
        data = np.load(os.path.join(LANDMARKS_DIR, file))

        if data.ndim != 2:
            print(f"  Skipping {word}: shape {data.shape}")
            continue

        # Auto-pad single-hand to two-hand
        data = pad_to_two_hand(data)
        if data is None:
            print(f"  Skipping {word}: unsupported feature size")
            continue

        data = data[:MIN_SAMPLES_PER_CLASS]
        for sample in data:
            X_raw.append(sample)
            y.append(word)
        print(f"  {word}: {len(data)} samples")

    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y)
    print(f"\n[ISL] Total: {len(X_raw)} samples, {len(set(y))} classes")
    print(f"[ISL] Feature dim: {X_raw.shape[1]} (two-hand raw)")

    # 2. Mirror augmentation
    print("[ISL] Adding mirrored data...")
    X_mirror = mirror_landmarks(X_raw)
    X_raw = np.concatenate([X_raw, X_mirror], axis=0)
    y = np.concatenate([y, y], axis=0)

    # 3. Rotation + Scale
    print("[ISL] Adding rotation + scale...")
    X_rot = augment_rotation(X_raw)
    X_sc = augment_scale(X_raw)
    X_raw_aug = np.concatenate([X_raw, X_rot, X_sc], axis=0)
    y_aug = np.concatenate([y, y, y], axis=0)

    # 4. Feature engineering (84 → 120)
    print("[ISL] Computing two-hand extended features...")
    X = compute_two_hand_features(X_raw_aug)
    print(f"[ISL] Extended shape: {X.shape}")

    # 5. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_aug)
    y_categorical = to_categorical(y_encoded)
    np.save(LABELS_PATH, le.classes_)

    # 6. Standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X = (X - mean) / std
    np.save(MEAN_PATH, mean)
    np.save(STD_PATH, std)

    # 7. Noise augmentation
    X_noise = augment_noise(X, scale=0.02, n_aug=1)
    y_noise = np.concatenate([y_categorical, y_categorical], axis=0)
    y_enc_noise = np.concatenate([y_encoded, y_encoded], axis=0)
    print(f"[ISL] Final dataset: {len(X_noise)} samples")

    # 8. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_noise, y_noise, test_size=0.2, random_state=42, stratify=y_enc_noise
    )

    # 9. Build & compile
    model = build_model(NUM_TWO_HAND_EXTENDED, len(le.classes_))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 10. Train
    print("\n[ISL] Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ],
        verbose=1
    )

    # 11. Save
    model.save(MODEL_H5_PATH)
    print(f"\n[ISL] Keras model saved: {MODEL_H5_PATH}")

    # 12. TFLite export
    print("[ISL] Converting to TFLite...")
    try:
        saved_dir = os.path.join(PROJECT_ROOT, "_temp_saved_model")
        model.export(saved_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(MODEL_TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        shutil.rmtree(saved_dir, ignore_errors=True)
        print(f"[ISL] TFLite saved: {MODEL_TFLITE_PATH} ({os.path.getsize(MODEL_TFLITE_PATH)/1024:.1f} KB)")
    except Exception as e:
        print(f"[ISL] TFLite export failed: {e}")

    # 13. Report
    print("\n[ISL] Validation Results:")
    y_pred = model.predict(X_val, verbose=0)
    print(classification_report(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1),
                                target_names=le.classes_))

    print(f"\n{'=' * 60}")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Features: {NUM_TWO_HAND_EXTENDED} (84 raw → 120 extended)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
