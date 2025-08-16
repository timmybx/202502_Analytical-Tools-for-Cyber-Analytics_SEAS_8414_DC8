# Filename: 2_analyze_domain.py
# Train/export (as-is), score --domain with MOJO; if DGA, explain with SHAP (KernelExplainer) and
# generate a prescriptive playbook via the professor's Gemini helper.

import argparse
import importlib.util
import os
from pathlib import Path
import asyncio
import pandas as pd
import h2o
import shap
    
# --- dynamic import helpers (filenames start with digits) ---
def _load_module(file_name: str, module_name: str):
    here = Path(__file__).resolve().parent
    mod_path = here / file_name
    spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_train_export_module():
    return _load_module("1_train_and_export.py", "train_export_mod")

def _load_genai_module():
    # professor’s helper provides: async def generate_playbook(xai_findings, api_key)
    return _load_module("4_generate_prescriptive_playbook.py", "genai_mod")

# --- scoring with MOJO (short-lived H2O cluster) ---
def _predict_with_mojo(mojo_zip_path: str, length: int, entropy: float):
    """Score a single row with the exported MOJO and return (label, probs_dict)."""
    h2o.init()
    try:
        mojo = h2o.import_mojo(mojo_zip_path)
        frame = h2o.H2OFrame(pd.DataFrame([{"length": length, "entropy": entropy}]))
        pred_df = mojo.predict(frame).as_data_frame()

        label = str(pred_df.loc[0, "predict"]).lower()

        # Keep only numeric prob columns; exclude the 'predict' string column.
        prob_cols = [
            c for c in pred_df.columns
            if c.lower() != "predict" and pd.api.types.is_numeric_dtype(pred_df[c])
        ]
        probs = {c: float(pred_df.loc[0, c]) for c in prob_cols}
        return label, probs
    finally:
        h2o.cluster().shutdown(prompt=False)


# --- local SHAP using the professor's KernelExplainer pattern, adapted to ONE row ---
def _local_shap(native_model_path: str, background_csv: str, instance_df: pd.DataFrame):
    """
    Returns a dict of SHAP values for the single instance, e.g. {"length": val, "entropy": val}.
    Mirrors the 'Hands-On Lab 2' code (KernelExplainer + predict_wrapper), but runs on one row.
    """
    feats = ["length", "entropy"]

    h2o.init()
    try:
        # Load the saved native model (same as lab)
        best_model = h2o.load_model(native_model_path)

        # Background set for KernelExplainer: small head(50) of the training CSV (same as lab)
        test_df = pd.read_csv(background_csv)
        X_bg = test_df[feats].head(50)

        # predict_wrapper: build H2OFrame, call model.predict, return the 'dga' column (same as lab)
        def predict_wrapper(data):
            h2o_df = h2o.H2OFrame(pd.DataFrame(data, columns=feats))
            predictions = best_model.predict(h2o_df).as_data_frame()
            if "dga" in predictions.columns:
                return predictions["dga"].values
            # fallback if columns are p0/p1 (keeps it robust without changing the lab’s intent)
            num_cols = [c for c in predictions.columns if pd.api.types.is_numeric_dtype(predictions[c])]
            return predictions[num_cols[0]].values

        explainer = shap.KernelExplainer(predict_wrapper, X_bg)
        shap_vals = explainer.shap_values(instance_df[feats])   # <-- ONE row only

        # shap_vals for a single row is a length-2 vector in our case (length, entropy)
        arr = shap_vals[0] if hasattr(shap_vals, "__len__") else [shap_vals]
        return {"length": float(arr[0]), "entropy": float(arr[1])}
    finally:
        h2o.cluster().shutdown(prompt=False)


def _build_xai_findings(domain: str, length: int, entropy: float, probs: dict, shap_map: dict) -> str:
    # Use the highest available probability as "confidence"
    conf = max(probs.values()) if probs else None
    conf_str = f"{round(conf * 100.0, 1)}%" if conf is not None else "high"

    s_ent = shap_map.get("entropy")
    s_len = shap_map.get("length")

    parts = []
    parts.append(f"- Alert: Potential DGA domain detected.")
    parts.append(f"- Domain: '{domain}'")
    parts.append(f"- AI Model Explanation (from SHAP): The model flagged this domain with {conf_str} confidence.")
    parts.append(f"  The classification was primarily driven by:")
    if s_ent is not None:
        parts.append(f"  - A high 'entropy' value of {round(entropy, 3)} "
                     f"(SHAP contribution: {round(s_ent, 4)}), pushing towards 'dga'.")
    if s_len is not None:
        parts.append(f"  - A high 'length' value of {length} "
                     f"(SHAP contribution: {round(s_len, 4)}), pushing towards 'dga'.")
    return " ".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Train/export model, score a domain, and (if DGA) print a prescriptive playbook.")
    parser.add_argument("--domain", required=True, help="Domain name to analyze, e.g. kq3v9z7j1x5f8g2h.info")
    args = parser.parse_args()

    train_export_mod = _load_train_export_module()
    genai_mod = _load_genai_module()

    # 1) Train/export exactly as-is (Part 1)
    csv_path = train_export_mod.generate_training_csv('dga_dataset_train.csv', n_legit=100, n_dga=100)
    mojo_path, native_model_path, leaderboard_csv = train_export_mod.run_automl(csv_path, models_dir="./models")
    print(f"[✓] MOJO: {mojo_path}")
    print(f"[✓] Native model path: {native_model_path}")
    print(f"[✓] Leaderboard CSV: {leaderboard_csv}")

    # 2) Feature engineering for the single domain
    domain = args.domain
    length = len(domain)
    entropy = train_export_mod.get_entropy(domain)
    print(f"\n[i] Domain: {domain}")
    print(f"[i] Features -> length={length}, entropy={round(entropy, 6)}")

    # 3) Score with MOJO
    label, probs = _predict_with_mojo(mojo_path, length, entropy)
    print("\n=== Prediction ===")
    print(f"label: {label}")
    for k, v in sorted(probs.items()):
        print(f"{k}: {v:.6f}")

    # 4) If DGA, explain locally with SHAP and generate a prescriptive playbook
    if label == "dga":
        shap_map = _local_shap(
            native_model_path,
            csv_path,
            pd.DataFrame([{"length": length, "entropy": entropy}])
        )

        xai_findings = _build_xai_findings(domain, length, entropy, probs, shap_map)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("\n---")
            print("🚨 Error: GOOGLE_API_KEY environment variable not set.")
            print("To run this step, set your API key:")
            print("\nFor Linux/macOS, use:\n  export GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
            print("\nFor Windows (PowerShell), use:\n  $env:GOOGLE_API_KEY=\"YOUR_API_KEY_HERE\"")
            print("---")
            return

        playbook = asyncio.run(genai_mod.generate_playbook(xai_findings, api_key))
        print("\n=== Prescriptive Incident Response Plan ===")
        print(playbook)
    else:
        print("\nNo playbook generated (prediction = legit).")

if __name__ == "__main__":
    main()
