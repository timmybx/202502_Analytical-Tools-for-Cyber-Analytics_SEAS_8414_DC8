# week8/train_model.py
# Part 1: Threat Profile Data Generation + baseline PyCaret model training.
# - Adds "threat_profile" (state|crime|hacktivist|benign) for enrichment/clustering.
# - Adds "has_political_keyword" feature (used for Hacktivist signal).
# - Trains classifier ONLY on the 12 features app.py supplies at inference time.

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    finalize_model,
    plot_model,
    save_model,
)

# ---- IMPORTANT: This must match app.py's input schema exactly ----
CLASSIFIER_FEATURES: List[str] = [
    "having_IP_Address",
    "URL_Length",  # -1 short, 0 normal, +1 long
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",  # -1 none, 0 one, +1 many
    "SSLfinal_State",  # -1 none, 0 suspicious, +1 trusted
    "Abnormal_URL",
    "URL_of_Anchor",  # count-like: 0/1/2
    "Links_in_tags",  # count-like: 0/1/2
    "SFH",  # simple flag-ish: -1/0/1
]

# Extra feature(s) reserved for enrichment/clustering (NOT used by classifier)
ENRICHMENT_ONLY_FEATURES: List[str] = [
    "has_political_keyword",  # drives Hacktivist signal
]

ALL_FEATURES: List[str] = CLASSIFIER_FEATURES + ENRICHMENT_ONLY_FEATURES


def _bern_pm1(rng: np.random.Generator, p_positive: float) -> int:
    """Return +1 with probability p_positive, else -1."""
    return 1 if rng.random() < p_positive else -1


def _tri_choice(
    rng: np.random.Generator, p_neg: float, p_zero: float, p_pos: float
) -> int:
    """Return -1, 0, or +1 with given probabilities (sum must be 1)."""
    x = rng.random()
    if x < p_neg:
        return -1
    if x < p_neg + p_zero:
        return 0
    return 1


def _small_count(rng: np.random.Generator, p0: float, p1: float, p2: float) -> int:
    """Return 0, 1, or 2 with given probabilities (sum must be 1)."""
    x = rng.random()
    if x < p0:
        return 0
    if x < p0 + p1:
        return 1
    return 2


def _row_for_profile(
    rng: np.random.Generator, profile: str
) -> Dict[str, int | float | str]:
    """
    Create one synthetic row with feature values conditioned on a threat profile.
    Profiles:
      - "state"      (State-Sponsored)
      - "crime"      (Organized Cybercrime)
      - "hacktivist" (Hacktivist)
      - "benign"
    """
    f: Dict[str, int | float | str] = {}

    if profile == "state":
        # High sophistication: valid SSL, subtle deception, longer-lived / well-formed
        f["having_IP_Address"] = _bern_pm1(rng, 0.10)  # mostly no IP literal
        f["URL_Length"] = _tri_choice(rng, 0.15, 0.55, 0.30)  # mostly normal/long
        f["Shortining_Service"] = _bern_pm1(rng, 0.10)  # rarely shortened
        f["having_At_Symbol"] = _bern_pm1(rng, 0.10)
        f["double_slash_redirecting"] = _bern_pm1(rng, 0.15)  # usually clean
        f["Prefix_Suffix"] = _bern_pm1(rng, 0.60)  # subtle deception sometimes
        f["having_Sub_Domain"] = _tri_choice(rng, 0.25, 0.45, 0.30)
        f["SSLfinal_State"] = _tri_choice(rng, 0.05, 0.15, 0.80)  # mostly trusted
        f["Abnormal_URL"] = _bern_pm1(rng, 0.15)
        f["URL_of_Anchor"] = _small_count(rng, 0.70, 0.25, 0.05)
        f["Links_in_tags"] = _small_count(rng, 0.70, 0.25, 0.05)
        f["SFH"] = _tri_choice(rng, 0.10, 0.70, 0.20)
        f["has_political_keyword"] = 1 if rng.random() < 0.15 else 0

    elif profile == "crime":
        # Noisy: shorteners, IP in URL, abnormal structures, odd tags; short-lived feel
        f["having_IP_Address"] = _bern_pm1(rng, 0.70)
        f["URL_Length"] = _tri_choice(rng, 0.10, 0.20, 0.70)  # skew long
        f["Shortining_Service"] = _bern_pm1(rng, 0.70)
        f["having_At_Symbol"] = _bern_pm1(rng, 0.30)
        f["double_slash_redirecting"] = _bern_pm1(rng, 0.60)
        f["Prefix_Suffix"] = _bern_pm1(rng, 0.50)
        f["having_Sub_Domain"] = _tri_choice(rng, 0.20, 0.30, 0.50)
        f["SSLfinal_State"] = _tri_choice(
            rng, 0.50, 0.30, 0.20
        )  # often none/suspicious
        f["Abnormal_URL"] = _bern_pm1(rng, 0.65)
        f["URL_of_Anchor"] = _small_count(rng, 0.20, 0.40, 0.40)
        f["Links_in_tags"] = _small_count(rng, 0.20, 0.40, 0.40)
        f["SFH"] = _tri_choice(rng, 0.60, 0.25, 0.15)
        f["has_political_keyword"] = 1 if rng.random() < 0.10 else 0

    elif profile == "hacktivist":
        # Opportunistic + topical: political keyword signal; mixed hygiene
        f["having_IP_Address"] = _bern_pm1(rng, 0.30)
        f["URL_Length"] = _tri_choice(rng, 0.20, 0.40, 0.40)
        f["Shortining_Service"] = _bern_pm1(rng, 0.30)
        f["having_At_Symbol"] = _bern_pm1(rng, 0.20)
        f["double_slash_redirecting"] = _bern_pm1(rng, 0.35)
        f["Prefix_Suffix"] = _bern_pm1(rng, 0.35)
        f["having_Sub_Domain"] = _tri_choice(rng, 0.30, 0.40, 0.30)
        f["SSLfinal_State"] = _tri_choice(rng, 0.30, 0.30, 0.40)
        f["Abnormal_URL"] = _bern_pm1(rng, 0.40)
        f["URL_of_Anchor"] = _small_count(rng, 0.45, 0.35, 0.20)
        f["Links_in_tags"] = _small_count(rng, 0.45, 0.35, 0.20)
        f["SFH"] = _tri_choice(rng, 0.35, 0.45, 0.20)
        f["has_political_keyword"] = 1 if rng.random() < 0.80 else 0  # key signal

    else:  # benign
        f["having_IP_Address"] = _bern_pm1(rng, 0.05)
        f["URL_Length"] = _tri_choice(rng, 0.40, 0.50, 0.10)  # mostly short/normal
        f["Shortining_Service"] = _bern_pm1(rng, 0.05)
        f["having_At_Symbol"] = _bern_pm1(rng, 0.05)
        f["double_slash_redirecting"] = _bern_pm1(rng, 0.05)
        f["Prefix_Suffix"] = _bern_pm1(rng, 0.10)
        f["having_Sub_Domain"] = _tri_choice(rng, 0.50, 0.40, 0.10)
        f["SSLfinal_State"] = _tri_choice(rng, 0.05, 0.15, 0.80)  # mostly trusted
        f["Abnormal_URL"] = _bern_pm1(rng, 0.05)
        f["URL_of_Anchor"] = _small_count(rng, 0.75, 0.20, 0.05)
        f["Links_in_tags"] = _small_count(rng, 0.75, 0.20, 0.05)
        f["SFH"] = _tri_choice(rng, 0.05, 0.80, 0.15)
        f["has_political_keyword"] = 0

    f["threat_profile"] = profile
    f["label"] = 0 if profile == "benign" else 1
    return f


def generate_synthetic_dataset(
    n_state: int = 400,
    n_crime: int = 400,
    n_hacktivist: int = 300,
    n_benign: int = 900,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a dataset with clear patterns that a clustering algorithm can discover."""
    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, int | float | str]] = []

    for _ in range(n_state):
        rows.append(_row_for_profile(rng, "state"))
    for _ in range(n_crime):
        rows.append(_row_for_profile(rng, "crime"))
    for _ in range(n_hacktivist):
        rows.append(_row_for_profile(rng, "hacktivist"))
    for _ in range(n_benign):
        rows.append(_row_for_profile(rng, "benign"))

    df = pd.DataFrame(rows)
    df = df[ALL_FEATURES + ["threat_profile", "label"]]  # order for readability
    return df


def main() -> None:
    print("Generating synthetic dataset with threat profiles...")
    data = generate_synthetic_dataset()

    # Persist the training data (useful for debugging and the Streamlit app)
    os.makedirs("models", exist_ok=True)
    data_path = os.path.join("models", "training_data.csv")
    data.to_csv(data_path, index=False)
    print(f"[✓] Saved training dataset with profiles to: {data_path}")

    # --- Classification training; ignore enrichment fields ---
    print("Initializing PyCaret Setup...")
    s = setup(
        data=data,
        target="label",
        session_id=42,
        verbose=False,
        ignore_features=["threat_profile"] + ENRICHMENT_ONLY_FEATURES,
    )
    _ = s  # quiet linters

    print("Comparing models...")
    best_model = compare_models(n_select=1, include=["rf", "et", "lightgbm"])

    print("Finalizing model...")
    final_model = finalize_model(best_model)

    # Feature importance plot
    print("Saving feature importance plot...")
    plot_model(final_model, plot="feature", save=True)
    src = "Feature Importance.png"
    dst = os.path.join("models", "feature_importance.png")
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"[✓] Saved feature importance plot to: {dst}")

    # Model artifact (must match app.py expectation)
    print("Saving model...")
    model_base = os.path.join("models", "phishing_url_detector")
    save_model(final_model, model_base)
    print(f"[✓] Model saved to: {model_base}.pkl")

    print("Done.")


if __name__ == "__main__":
    main()
