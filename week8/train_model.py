# week8/train_model.py

import os
import numpy as np
import pandas as pd

# FIX: import only what you use (no star import)
from pycaret.classification import (
    setup,
    compare_models,
    finalize_model,
    plot_model,
    save_model,
)


def generate_synthetic_dataset(
    n_samples: int = 1000, random_state: int = 42
) -> pd.DataFrame:
    """Create a tiny synthetic dataset for demo/training."""
    rng = np.random.default_rng(random_state)

    # (Optional) Keep feature names explicit; actually use them so Ruff doesn't flag "unused".
    feature_names = [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token",
    ]

    X = rng.integers(
        0, 2, size=(n_samples, len(feature_names))
    )  # simple binary toy features
    y = rng.integers(0, 2, size=n_samples)

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    return df


def main() -> None:
    print("Generating synthetic dataset...")
    data = generate_synthetic_dataset()

    print("Initializing PyCaret Setup...")
    # FIX: don't assign to a variable you don't use (remove: s = ...)
    setup(data=data, target="label", session_id=42, verbose=False)

    print("Comparing models...")
    best_model = compare_models(n_select=1, include=["rf", "et", "lightgbm"])

    print("Finalizing model...")
    final_model = finalize_model(best_model)

    # Save feature importance plot
    print("Saving feature importance plot...")
    os.makedirs("models", exist_ok=True)
    plot_model(final_model, plot="feature", save=True)

    # PyCaret saves as "Feature Importance.png" in CWD; move it into models/
    plot_src = "Feature Importance.png"
    plot_dst = os.path.join("models", "feature_importance.png")
    if os.path.exists(plot_src):
        os.replace(plot_src, plot_dst)

    # Save model
    print("Saving model...")
    model_base = os.path.join("models", "pycaret_model")
    save_model(final_model, model_base)
    print("Model and plot saved successfully.")


if __name__ == "__main__":
    main()
