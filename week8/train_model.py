# week8/train_model.py
# Part 1: Threat Profile Data Generation + baseline PyCaret model training.
# Part 2: Unsupervised clustering to build a "threat actor profiler" (3 clusters).
#
# - Adds "threat_profile" (state|crime|hacktivist|benign) for enrichment/clustering.
# - Adds "has_political_keyword" feature (used for Hacktivist signal).
# - Trains classifier ONLY on the 12 features app.py supplies at inference time.
# - Trains a separate K-Means clustering model (k=3) on a focused subset of features
#   to better separate actor profiles (esp. Hacktivist).

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from pycaret.classification import (
    setup as cls_setup,
    compare_models,
    finalize_model,
    plot_model,
    save_model as cls_save_model,
)
from pycaret import clustering as pcl  # use namespace to avoid setup() name clash


# ---- IMPORTANT: Must match app.py's classifier input schema exactly ----
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
    "URL_of_Anchor",  # 0/1/2
    "Links_in_tags",  # 0/1/2
    "SFH",  # -1/0/1
]

# Extra feature(s) used for enrichment/attribution (NOT used by classifier)
ENRICHMENT_ONLY_FEATURES: List[str] = [
    "has_political_keyword",
]

ALL_FEATURES: List[str] = CLASSIFIER_FEATURES + ENRICHMENT_ONLY_FEATURES

# ---- Focused subset for clustering/attribution ----
# Using a smaller, discriminative set helps K-Means carve out clearer clusters.
CLUSTERING_FEATURES: List[str] = [
    "has_political_keyword",  # strong Hacktivist signal
    "SSLfinal_State",  # State vs. Crime separator
    "Shortining_Service",  # Crime signal
    "having_IP_Address",  # Crime signal
    "Prefix_Suffix",  # State can use subtle deception
    "Abnormal_URL",  # Crime + some Hacktivist
    "URL_Length",  # Crime often long
]


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
    Synthetic row with feature values conditioned on a threat profile.
      - "state"      (State-Sponsored)
      - "crime"      (Organized Cybercrime)
      - "hacktivist" (Hacktivist)
      - "benign"
    """
    f: Dict[str, int | float | str] = {}

    if profile == "state":
        # High sophistication: valid SSL, subtle deception, longer-lived / well-formed
        f["having_IP_Address"] = _bern_pm1(rng, 0.10)
        f["URL_Length"] = _tri_choice(rng, 0.15, 0.55, 0.30)
        f["Shortining_Service"] = _bern_pm1(rng, 0.10)
        f["having_At_Symbol"] = _bern_pm1(rng, 0.10)
        f["double_slash_redirecting"] = _bern_pm1(rng, 0.15)
        f["Prefix_Suffix"] = _bern_pm1(rng, 0.60)
        f["having_Sub_Domain"] = _tri_choice(rng, 0.25, 0.45, 0.30)
        f["SSLfinal_State"] = _tri_choice(rng, 0.05, 0.15, 0.80)  # mostly trusted
        f["Abnormal_URL"] = _bern_pm1(rng, 0.15)
        f["URL_of_Anchor"] = _small_count(rng, 0.70, 0.25, 0.05)
        f["Links_in_tags"] = _small_count(rng, 0.70, 0.25, 0.05)
        f["SFH"] = _tri_choice(rng, 0.10, 0.70, 0.20)
        f["has_political_keyword"] = 1 if rng.random() < 0.15 else 0

    elif profile == "crime":
        # Noisy: shorteners, IP in URL, abnormal structures; short-lived feel
        f["having_IP_Address"] = _bern_pm1(rng, 0.70)
        f["URL_Length"] = _tri_choice(rng, 0.10, 0.20, 0.70)  # skew long
        f["Shortining_Service"] = _bern_pm1(rng, 0.70)
        f["having_At_Symbol"] = _bern_pm1(rng, 0.30)
        f["double_slash_redirecting"] = _bern_pm1(rng, 0.60)
        f["Prefix_Suffix"] = _bern_pm1(rng, 0.50)
        f["having_Sub_Domain"] = _tri_choice(rng, 0.20, 0.30, 0.50)
        f["SSLfinal_State"] = _tri_choice(rng, 0.50, 0.30, 0.20)
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
        f["URL_Length"] = _tri_choice(rng, 0.40, 0.50, 0.10)
        f["Shortining_Service"] = _bern_pm1(rng, 0.05)
        f["having_At_Symbol"] = _bern_pm1(rng, 0.05)
        f["double_slash_redirecting"] = _bern_pm1(rng, 0.05)
        f["Prefix_Suffix"] = _bern_pm1(rng, 0.10)
        f["having_Sub_Domain"] = _tri_choice(rng, 0.50, 0.40, 0.10)
        f["SSLfinal_State"] = _tri_choice(rng, 0.05, 0.15, 0.80)
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

    # --------------------------
    # Classification workflow
    # --------------------------
    print("Initializing PyCaret Classification Setup...")
    _ = cls_setup(
        data=data,
        target="label",
        session_id=42,
        verbose=False,
        ignore_features=["threat_profile"] + ENRICHMENT_ONLY_FEATURES,
    )

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

    # Model artifact (matches app.py expectation)
    print("Saving classifier model...")
    cls_base = os.path.join("models", "phishing_url_detector")
    cls_save_model(final_model, cls_base)
    print(f"[✓] Classifier saved to: {cls_base}.pkl")

    # --------------------------
    # Clustering workflow (Part 2)
    # --------------------------
    print(
        "Initializing PyCaret Clustering Setup (malicious rows only, focused features)..."
    )
    malicious = data.loc[data["label"] == 1].copy()

    # Focused, discriminative features for clustering
    cluster_df = malicious[CLUSTERING_FEATURES].copy()

    if len(cluster_df) < 3:
        raise RuntimeError("Not enough malicious rows for 3 clusters.")

    # Minimal, version-friendly setup
    _ = pcl.setup(
        data=cluster_df,
        session_id=42,
        verbose=False,
        preprocess=True,  # scaling/encoding as needed
    )

    print("Creating K-Means clustering model with k=3...")
    kmeans3 = pcl.create_model("kmeans", num_clusters=3)

    # (Optional) Inspect/preview clusters for debugging:
    # preview = pcl.assign_model(kmeans3)
    # preview.to_csv(os.path.join("models", "cluster_preview.csv"), index=False)

    print("Saving clustering model (threat actor profiler)...")
    pcl.save_model(kmeans3, os.path.join("models", "threat_actor_profiler"))
    print("[✓] Clustering model saved to: models/threat_actor_profiler.pkl")

    print("Done.")


if __name__ == "__main__":
    main()
