# INSTALL.md — Setup & Deployment

This guide covers two paths: **Docker (recommended)** and **local install**. Either way, you’ll need API keys for at least one GenAI provider (Gemini, OpenAI, or Grok).

---

## Prerequisites

- **Docker** (Desktop or Engine) and **Docker Compose**
- **Make** (Linux/macOS preinstalled; on Windows use WSL or install via Chocolatey)
- API key(s) for:
  - Google **Gemini**
  - **OpenAI**
  - **Grok** (xAI)

> You only need one provider to run the prescriptive step.

---

## 1) Docker Path (Recommended)

1) Clone the repo and enter the project folder (e.g., `week8/`).

2) Create secrets file:
```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<'EOF'
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "AIza..."
GROK_API_KEY   = "gsk_..."
EOF
```
(Provide at least one key.)

3) Build & run:
```bash
make up
```

- On first run the container will:
  - install dependencies,
  - run `train_model.py` to produce `models/phishing_url_detector.pkl` and `models/threat_actor_profiler.pkl`,
  - start Streamlit on **http://localhost:8501**.

4) Stop / Logs / Clean:
```bash
make down      # stop containers
make logs      # tail app logs
make clean     # stop + remove generated models/data
```

---

## 2) Local Install (Advanced)

> If you cannot use Docker, you can run locally. Results may vary based on OS/Python tooling.

1) Create and activate a virtualenv (Python 3.10+ recommended):
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

2) Install requirements:
```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

3) Create secrets file:
```bash
mkdir -p .streamlit
# then edit .streamlit/secrets.toml and add your keys
```

4) Train models:
```bash
python train_model.py
```

5) Run the app:
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## Troubleshooting

- **“Model not found” in the app:** Wait for training to finish on first run, or run `python train_model.py` manually. Use `make logs` to watch progress.
- **Port in use (8501):** Stop any other Streamlit apps or change the port: `streamlit run app.py --server.port 8502`.
- **Missing API key:** The prescriptive tab will error if no provider key is present. Add a key to `.streamlit/secrets.toml` and restart.
- **PyCaret version quirks:** We use a minimal, version‑tolerant set of parameters for clustering. If you modify `train_model.py`, avoid passing unsupported kwargs to `pycaret.clustering.setup()` or `create_model()`.
- **Windows local installs:** If you hit build tool errors when compiling dependencies, prefer the Docker path.
