import os
import numpy as np

LANDMARK_PATH = "../data/landmarks"

for file in os.listdir(LANDMARK_PATH):
    if file.endswith(".npy"):
        data = np.load(os.path.join(LANDMARK_PATH, file))
        print(file.replace(".npy", ""), "->", data.shape)
