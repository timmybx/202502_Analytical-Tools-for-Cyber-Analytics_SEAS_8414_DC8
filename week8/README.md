# GenAI‑Powered Mini‑SOAR for Phishing Analysis (with Threat Attribution)

This project is a prototype **Security Orchestration, Automation, and Response (SOAR)** application built with Python and Streamlit. It now goes beyond simple malicious/benign detection to include **Threat Attribution** and **GenAI‑driven prescriptions**.

- **Prediction (Classifier):** A PyCaret classification pipeline determines whether the submitted URL features look **Benign** or **Malicious**.
- **Attribution (Clustering):** If the verdict is malicious, a PyCaret clustering model assigns the sample to one of three **threat actor profiles**: **State‑Sponsored**, **Organized Cybercrime**, or **Hacktivist**.
- **Prescription (GenAI):** For malicious cases, the app calls a Generative AI provider (Gemini, OpenAI, or Grok) to draft a **prescriptive playbook** of actions and a user‑facing **communication draft**.

> For installation and platform details, see **[INSTALL.md](INSTALL.md)**. For hands‑on test flows (benign + each actor), see **[TESTING.md](TESTING.md)**.

---

## What’s New (Dual‑Model Architecture)

### 1) Classification model (malicious vs. benign)
- Trained with **PyCaret** on a synthetic dataset produced by `train_model.py`.
- Uses the same feature schema as the Streamlit UI, notably including the enrichment flag **`has_political_keyword`** so that topical/activist signals can influence the malicious/benign decision when appropriate.
- Saved to `models/phishing_url_detector.pkl`.

### 2) Clustering model (actor attribution)
- A **K‑Means** model (k=3) trained on a focused subset of discriminative features (e.g., `has_political_keyword`, SSL state, IP usage, shortener usage, etc.).
- At runtime, the app builds a **cluster → profile** mapping (majority vote over labeled training data) so dynamic cluster IDs are translated to human‑readable profiles:
  - **State‑Sponsored**
  - **Organized Cybercrime**
  - **Hacktivist**
- Saved to `models/threat_actor_profiler.pkl`.

### 3) Prescriptive step (GenAI)
- For malicious verdicts, the app calls your chosen LLM provider to produce an **incident response plan** and a suggested **communication draft**.
- Providers are configured via `.streamlit/secrets.toml` (see **INSTALL.md**).

---

## Repository Layout

```
week8/
├── app.py                    # Streamlit UI: prediction → attribution → prescription
├── train_model.py            # Generates data; trains classifier & clustering models
├── genai_prescriptions.py    # GenAI provider wrapper (Gemini/OpenAI/Grok)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md                 # (this file)
├── INSTALL.md
├── TESTING.md
└── .streamlit/
    └── secrets.toml          # API keys (not committed)
```

---

## Quick Start

> Prefer Docker + Make (simplest, reproducible). Native install instructions live in **INSTALL.md**.

1. **Create secrets file** (pick at least one provider):
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "sk-..."
   GEMINI_API_KEY = "AIza..."
   GROK_API_KEY   = "gsk_..."
   ```

2. **Build & run** (first run trains the models automatically):
   ```bash
   make up
   ```

3. Open the app: **http://localhost:8501**

4. To stop:
   ```bash
   make down
   ```

5. To view logs / clean generated artifacts:
   ```bash
   make logs
   make clean
   ```

---

## Using the App

1. In the left sidebar, set URL attributes (length, SSL state, sub‑domain complexity, etc.).
2. Toggle **“Contains political / activist keywords”** to simulate topical/activist content.
3. Click **“Analyze & Initiate Response”**.
4. Review tabs:
   - **Analysis Summary:** Classifier verdict + confidence.
   - **Visual Insights:** A simple risk bar chart and global feature importance.
   - **Prescriptive Plan:** LLM‑generated actions and a communication draft (malicious only).
   - **Threat Attribution:** Actor profile for malicious verdicts (State‑Sponsored / Organized Cybercrime / Hacktivist).

> Because clustering IDs are recomputed per training run, the app derives the label mapping from the training data so profile names remain stable (by majority vote).

---

## Notes & Limitations

- **Synthetic data:** The training data is simulated for classroom use; it encodes clear patterns for the three actor profiles and **should not** be used for production security decisions.
- **LLM outputs:** Treat prescriptive text as **draft guidance**. Validate against your organization’s IR playbooks.
- **Run‑to‑run variance:** AutoML selection and clustering can vary slightly; outcomes should remain consistent in spirit but may not be bit‑for‑bit identical.

---

## License

Educational use only. See repository license for details.
