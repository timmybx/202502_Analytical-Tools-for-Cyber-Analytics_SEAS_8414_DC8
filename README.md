# SEAS 8414-DC8 · Analytical Tools for Cyber Analytics (Summer 2025, GWU)

> Coursework repository for George Washington University’s **SEAS 8414-DC8 – Analytical Tools for Cyber Analytics** (Summer 2025).  
> Weekly assignments, lab code, and small projects organized by module/week.

---

## 📚 Course Overview

**Analytical Tools for Cyber Analytics** introduces the collection, processing, visualization, and machine-assisted analysis of network traffic, system logs, malware features, vulnerabilities, and threat intelligence. Students will learn to deploy Docker-containerized Security Information and Event Management (SIEM), Endpoint Detection and Response (EDR), network monitoring, and threat intelligence platforms on Amazon Web Services (AWS), then ingest and preprocess real-world datasets—packet capture files (PCAPs), network flow records (NetFlow/IPFIX), system logs, Common Vulnerabilities and Exposures (CVEs), indicators of compromise (IOC) feeds, and malware feature sets—using the Python programming language, automated machine learning (AutoML) frameworks, and reinforcement learning techniques.

---

## 🗂️ Repository Layout

The repo is organized by week. Each folder contains the assignment code, any provided lab scaffolding, and (when applicable) a short README describing how to run that week’s work.

```
.
├─ week09/  # Cognitive Analytics: AI-driven decisioning—Natural Language Processing–powered Threat Intelligence & Intelligent Playbooks 
├─ week10/  # End-to-end pipeline (Detection → SHAP → GenAI playbook)
└─ ...
```

> Tip: open a week’s folder and look for a local `README.md` or `TESTING.md` for run steps and sample commands.

---

## 🧰 Core Tools & Platforms

- **Python 3.12** (preferred)  
- **Docker** & **AWS** (for lab hosting / services)  
- **H2O AutoML**, **SHAP**, **pandas** (data & modeling)  
- **Ruff** (lint/format), **GitHub Actions** (CI)  

Some weeks may include additional utilities (e.g., scapy, pyshark, boto3, etc.). Check each week’s `requirements.txt` or the week-specific README for details.

---

## 🚀 Quick Start (common pattern)

1) Clone the repo
```bash
git clone https://github.com/timmybx/202502_Analytical-Tools-for-Cyber-Analytics_SEAS_8414_DC8.git
cd 202502_Analytical-Tools-for-Cyber-Analytics_SEAS_8414_DC8
```

2) (Recommended) Create a virtual environment
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

3) Install week-specific dependencies  
Most weeks include a `requirements.txt` inside the week folder. For example, week 10:
```bash
cd week10
pip install -r requirements.txt
```

4) Run the week’s assignment
```bash
# Check the week’s README.md for exact commands
python some_script.py --help
```

---

## ✅ Code Quality & CI

- **Ruff** is used for linting & formatting locally:
  ```bash
  python -m pip install ruff
  python -m ruff check .
  python -m ruff format .
  ```

- **GitHub Actions** runs a simple CI workflow (see `.github/workflows/lint.yml`) on push/PR to keep style consistent.

---

## 🔒 Academic Integrity

This repository is intended to demonstrate personal coursework and learning. If you are a student referencing this repo, please follow your institution’s academic integrity policies—use code and writeups as guidance, not as a substitute for your own work.

---

## 🙏 Acknowledgements

- Course staff & materials for **SEAS 8414-DC8** (GWU)  
- Open-source communities behind H2O, SHAP, pandas, Docker, and many other tools used throughout the assignments

---

## 📝 Notes

- Some week folders may generate large artifacts (models, caches). These are typically ignored via `.gitignore`.  
- When an assignment *requires* including a model artifact for grading, the week’s README will call this out explicitly.
