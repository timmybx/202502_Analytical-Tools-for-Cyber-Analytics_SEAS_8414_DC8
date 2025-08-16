# Manual Verification Guide

This guide shows how to run and verify the full pipeline with **one DGA‑like domain** and **one legitimate domain**. 
Small variations in probabilities are expected; focus on labels and whether a playbook is produced.

---

## 0) Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set your Gemini API key (required for the playbook step)

**Windows PowerShell**
```powershell
$env:GOOGLE_API_KEY = "YOUR_AI_STUDIO_KEY"
```

**macOS/Linux (bash/zsh)**
```bash
export GOOGLE_API_KEY="YOUR_AI_STUDIO_KEY"
```

> If `GOOGLE_API_KEY` is **not** set, the pipeline still performs detection, but will **skip** the playbook step with a clear message.

---

## 1) Test A — DGA‑like domain

Use a random, high‑entropy string as the domain (example below).

**Command**
```bash
python 2_analyze_domain.py --domain asdfqwerasdfa.com
```

**Expected console flow**
- H2O starts; AutoML trains (progress messages appear)
- Artifacts saved:
  - Native model → `./models/best_dga_model/`
  - Leaderboard → `./models/leaderboard.csv`
  - **MOJO** → `./model/DGA_Leader.zip`
- Features printed for the domain (length & entropy)
- **Prediction** shows label very likely `dga`
- **SHAP** runs for this single instance (KernelExplainer)
- **XAI Findings** printed (mentions high entropy/length with SHAP contributions)
- **Prescriptive Incident Response Plan** printed from Gemini (multi‑step playbook)

**Sample (abridged)**
```
=== Prediction ===
label: dga
p0: 0.9681
p1: 0.0319

=== XAI Findings (to Gemini) ===
- Alert: Potential DGA domain detected. - Domain: 'asdfqwerasdfa.com' - AI Model Explanation (from SHAP): ...
  - A high 'entropy' value of 4.3 (SHAP contribution: 0.12) ...
  - A high 'length' value of 14 (SHAP contribution: 0.07) ...

=== Prescriptive Incident Response Plan ===
1. Isolate ...
2. Investigate ...
...
```

---

## 2) Test B — Legitimate domain

**Command**
```bash
python 2_analyze_domain.py --domain google.com
```

**Expected console flow**
- H2O starts; AutoML trains
- Features printed for `google.com`
- **Prediction** shows label likely `legit`
- **No SHAP or playbook** (script prints “No playbook generated (prediction = legit).”)

**Sample (abridged)**
```
=== Prediction ===
label: legit
p0: 0.0813
p1: 0.9187

No playbook generated (prediction = legit).
```

---

## 3) Post‑run artifacts checklist

After either test, verify these files exist:

- `./dga_dataset_train.csv`
- `./models/best_dga_model/` (native H2O model directory)
- `./models/leaderboard.csv`
- `./model/DGA_Leader.zip` (**assignment‑required MOJO**)

---

## 4) Troubleshooting

- **Java not found**: Install JDK 17+ and ensure `java` is on your PATH.
- **“XGBoost is not available; skipping it”**: Harmless; AutoML continues with other algos.
- **GBM `min_rows` messages**: Expected with small data; AutoML continues regardless.
- **Gemini errors**: Ensure `GOOGLE_API_KEY` is set and valid; re‑run the test.
