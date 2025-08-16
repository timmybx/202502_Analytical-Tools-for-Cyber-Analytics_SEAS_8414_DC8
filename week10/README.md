# DGA Detection: Detection → Explanation → Prescription

This repository contains a small, command‑line pipeline that:
1) **Trains** an H2O AutoML model for DGA detection and **exports** a production MOJO  
2) **Scores** a single input domain with the MOJO  
3) If classified as **DGA**, produces a **local SHAP** explanation for that single prediction and  
4) Sends a structured **XAI summary** to **Google Gemini** to generate a **prescriptive incident response playbook**.

The code follows the professor’s “Hands‑On” labs closely (AutoML + MOJO; SHAP via **KernelExplainer** with a small background; GenAI bridge).

---

## Tech Stack
- **Python**, **H2O AutoML** (training + MOJO scoring)
- **SHAP** (KernelExplainer) for *single‑instance* explanation — adapted directly from the “Hands‑On Lab 2” example
- **Google Generative AI (Gemini)** for prescriptive playbook generation

---

## Repository Layout (key files)
```
.
├─ 1_train_and_export.py               # Part 1: train + export artifacts (called by the main app)
├─ 2_analyze_domain.py                 # Main CLI: runs Part 1, scores --domain, SHAP, Gemini
├─ 3_explain_model.py                  # (Professor example, kept for reference)
├─ 4_generate_prescriptive_playbook.py # (Professor example used by the main app)
├─ dga_dataset_train.csv               # Generated training CSV (created at runtime)
├─ models/                             # Native H2O model dir + leaderboard.csv (created at runtime)
│   ├─ best_dga_model/                 # Saved native model (folder)
│   └─ leaderboard.csv                 # AutoML leaderboard (CSV)
├─ model/                              # Assignment‑required normalized MOJO
│   └─ DGA_Leader.zip                  # ← Exported automatically by Part 1
├─ requirements.txt
├─ README.md
└─ TESTING.md
```

> **MOJO export behavior (assignment requirement):**  
> Part 1 **automatically** writes the production MOJO to `./model/DGA_Leader.zip` (no manual copy needed).

---

## Prerequisites
- **Python 3.12**
- **Java JDK 17+** (H2O launches a local JVM)
- Internet access (for Gemini)
- A **Gemini API key** set in `GOOGLE_API_KEY`

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set your Gemini key
**Windows PowerShell**
```powershell
$env:GOOGLE_API_KEY = "YOUR_AI_STUDIO_KEY"
```
**macOS/Linux**
```bash
export GOOGLE_API_KEY="YOUR_AI_STUDIO_KEY"
```

---

## Usage (single command, end‑to‑end)
```bash
python 2_analyze_domain.py --domain asdfqwerasdfa.com
```
What happens:
1. Generates `dga_dataset_train.csv` (simple synthetic set)
2. Runs H2O AutoML, saves artifacts:
   - Native model → `./models/best_dga_model/`
   - Leaderboard → `./models/leaderboard.csv`
   - **MOJO** → `./model/DGA_Leader.zip` (normalized for grading)
3. Computes features (`length`, `entropy`) for `--domain`
4. Scores with the MOJO and prints class + probabilities
5. If class == **dga**:
   - Runs **KernelExplainer** SHAP on that **single instance** (as in the lab)
   - Builds **xai_findings** dynamically from SHAP + features + confidence
   - Calls Gemini and prints the **prescriptive playbook**

---

## Three‑Stage Architecture
1. **Detection (H2O AutoML + MOJO)**  
   Train on simple features (`length`, `entropy`); export MOJO to `/model/DGA_Leader.zip`.
2. **Explanation (SHAP – local, single instance)**  
   KernelExplainer with a tiny background (`head(50)`) from the training CSV; extract per‑feature contributions.
3. **Prescription (GenAI)**  
   Convert SHAP to structured **xai_findings** and send to Gemini to produce an incident response **playbook**.

---

## Conformance Checklist (for grading)
- [x] **Code**: All Python scripts + `requirements.txt`
- [x] **Model**: MOJO at `./model/DGA_Leader.zip` (exported automatically by Part 1)
- [x] **Docs**: `README.md` and `TESTING.md`
- [x] **Automation**: GitHub Actions workflow `lint.yml` (Ruff) — add `.github/workflows/lint.yml`

---

## Troubleshooting
- **H2O/XGBoost warnings**: harmless if XGBoost is unavailable; AutoML continues.
- **GBM `min_rows` errors**: dataset is intentionally small; AutoML continues with other models.
- **Java not found**: install JDK 17+ and ensure `java` is on your PATH.
- **Gemini errors**: ensure `GOOGLE_API_KEY` is set and valid.
