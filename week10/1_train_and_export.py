# Filename: 1_train_and_export.py
import csv
import random
import math
import h2o
from h2o.automl import H2OAutoML
import os


def get_entropy(s):
    p, lns = {}, float(len(s))
    for c in s:
        p[c] = p.get(c, 0) + 1
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())


def generate_training_csv(out_path="dga_dataset_train.csv", n_legit=100, n_dga=100):
    """Create a simple DGA vs legit training CSV."""
    header = ["domain", "length", "entropy", "class"]
    data = []

    # Legitimate domains
    legit_domains = ["google", "facebook", "amazon", "github", "wikipedia", "microsoft"]
    for _ in range(n_legit):
        domain = random.choice(legit_domains) + ".com"
        data.append([domain, len(domain), get_entropy(domain), "legit"])

    # DGA-like domains
    for _ in range(n_dga):
        length = random.randint(15, 25)
        domain = (
            "".join(
                random.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                for _ in range(length)
            )
            + ".com"
        )
        data.append([domain, len(domain), get_entropy(domain), "dga"])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"{out_path} created successfully.")
    return out_path


def run_automl(
    csv_path, models_dir="./models", model_zip_dir="./model", leader_name="DGA_Leader"
):
    """Run H2O AutoML on the given CSV and save artifacts.
    Returns:
        (mojo_path, model_dir_path, leaderboard_csv_path)
    """
    import shutil

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(model_zip_dir, exist_ok=True)

    h2o.init()
    train = h2o.import_file(csv_path)
    x = ["length", "entropy"]  # Features
    y = "class"  # Target
    train[y] = train[y].asfactor()

    aml = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1, nfolds=3)
    aml.train(x=x, y=y, training_frame=train)

    print("H2O AutoML process complete.")
    print("Leaderboard:")
    print(aml.leaderboard.head())

    # Save leaderboard to CSV (in ./models)
    leaderboard_csv_path = os.path.join(models_dir, "leaderboard.csv")
    aml.leaderboard.as_data_frame().to_csv(leaderboard_csv_path, index=False)
    print(f"Leaderboard saved to: {leaderboard_csv_path}")

    # Save native model (folder ./models/best_dga_model)
    custom_name = "best_dga_model"
    model_path = h2o.save_model(model=aml.leader, path=models_dir, force=True)
    print(f"Original saved model path: {model_path}")

    new_path = os.path.join(models_dir, custom_name)
    if os.path.exists(new_path):
        print(f"[i] Removing existing {new_path} before rename.")
        if os.path.isdir(new_path):
            shutil.rmtree(new_path)
        else:
            os.remove(new_path)
    os.rename(model_path, new_path)
    print(f"Renamed model path: {new_path}")

    # Export MOJO into ./models first (H2O will name it like MOJO_model_*.zip)…
    tmp_mojo_zip = aml.leader.download_mojo(path=models_dir)
    # …then copy/normalize to the assignment-required path ./model/DGA_Leader.zip
    final_mojo = os.path.join(model_zip_dir, f"{leader_name}.zip")
    if os.path.exists(final_mojo):
        os.remove(final_mojo)
    shutil.copy2(tmp_mojo_zip, final_mojo)
    print(f"[✓] Exported MOJO to: {final_mojo}")

    h2o.cluster().shutdown(prompt=False)  # deprecation-safe

    # Return the required MOJO path (./model/DGA_Leader.zip), native model dir, and leaderboard CSV
    return final_mojo, new_path, leaderboard_csv_path
