import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "../data/landmarks"

X = []
y = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        word = file.replace(".npy", "")
        data = np.load(os.path.join(DATA_PATH, file))

        # Balance classes by limiting to minimum size
        min_samples = 140  # use lowest class count (ALL)
        data = data[:min_samples]

        for sample in data:
            X.append(sample)
            y.append(word)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Build improved model
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(42,)))
model.add(Dropout(0.4))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(le.classes_), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

model.save("../model.h5")

print("Training complete and model saved.")
print("Classes:", le.classes_)
