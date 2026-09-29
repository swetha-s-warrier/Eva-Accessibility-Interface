"""
train_base_model.py
----------------------
Trains the lightweight base gaze model on the cached features/targets
produced by prepare_dataset.py, and saves it for gaze_model.py to load.

Usage:
    python dataset/train_base_model.py
"""

import os
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CACHE_DIR = "dataset/cache"
OUTPUT_PATH = "models/base_gaze_model.pkl"


def main():
    features = np.load(os.path.join(CACHE_DIR, "features.npy"))
    targets = np.load(os.path.join(CACHE_DIR, "targets.npy"))

    if len(features) < 200:
        raise SystemExit(f"Only {len(features)} usable samples - too few to train on. "
                          f"Check the prepare_dataset.py detection-failure warnings.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.1, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    test_score = model.score(X_test_scaled, y_test)
    preds = model.predict(X_test_scaled)
    mean_abs_error = np.mean(np.abs(preds - y_test))
    print(f"Validation R^2: {test_score:.3f}")
    print(f"Mean absolute error (normalized 0-1 screen units): {mean_abs_error:.3f}")

    os.makedirs("models", exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    print(f"Saved base gaze model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
