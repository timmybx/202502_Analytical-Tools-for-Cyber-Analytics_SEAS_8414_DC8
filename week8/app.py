# app.py
import os
import re
import time
import pandas as pd
import streamlit as st

# PyCaret imports (alias to avoid name collisions)
from pycaret.classification import (
    load_model as load_cls_model,
    predict_model as predict_cls,
)
from pycaret.clustering import (
    load_model as load_cluster_model,
    predict_model as predict_cluster,
)

from genai_prescriptions import generate_prescription

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI-Powered Phishing SOAR", page_icon="🛡️", layout="wide"
)

# Friendly display names for profiles (from training)
PROFILE_DISPLAY = {
    "state": "State-Sponsored",
    "crime": "Organized Cybercrime",
    "hacktivist": "Hacktivist",
}
PROFILE_DESC = {
    "State-Sponsored": (
        "Well-resourced, stealthy operations; careful tradecraft, valid SSL, subtle deception."
    ),
    "Organized Cybercrime": (
        "Profit-driven, high-volume campaigns; URL shorteners, IP-in-URL, abnormal structures."
    ),
    "Hacktivist": (
        "Cause-driven, topical messaging; mixed hygiene; opportunistic and event-driven."
    ),
}


@st.cache_resource
def load_assets():
    """Load models, plot, and build a robust cluster→profile mapping."""
    cls_path = "models/phishing_url_detector"
    clu_path = "models/threat_actor_profiler"
    plot_path = "models/feature_importance.png"
    train_csv = "models/training_data.csv"

    cls_model = load_cls_model(cls_path) if os.path.exists(cls_path + ".pkl") else None
    clu_model = (
        load_cluster_model(clu_path) if os.path.exists(clu_path + ".pkl") else None
    )
    plot = plot_path if os.path.exists(plot_path) else None

    cluster_map = {}  # e.g., {0: "crime", 1: "state", 2: "hacktivist"}
    expected_cols = None  # columns the clustering model expects

    def _normalize_cluster_id(val):
        """Convert things like 'Cluster 0' or '0' to int 0."""
        if isinstance(val, (int, float)):
            try:
                return int(val)
            except Exception:
                return None
        m = re.search(r"-?\d+", str(val))
        return int(m.group(0)) if m else None

    # Build mapping via majority vote over the malicious training rows
    if clu_model and os.path.exists(train_csv):
        try:
            train_df = pd.read_csv(train_csv)
            mal = train_df.loc[train_df["label"] == 1].copy()
            # Must mirror the features used in clustering (drop label + threat_profile only)
            cluster_features = mal.drop(columns=["label", "threat_profile"])
            expected_cols = list(cluster_features.columns)

            preds = predict_cluster(clu_model, data=cluster_features)
            # Find a column that looks like cluster id
            cluster_col = next(
                (
                    c
                    for c in ("Cluster", "Label", "prediction_label")
                    if c in preds.columns
                ),
                None,
            )
            # If PyCaret used assign_model naming, try that next
            if cluster_col is None:
                from pycaret.clustering import assign_model as cluster_assign

                preds = cluster_assign(clu_model, data=cluster_features)
                cluster_col = next(
                    (
                        c
                        for c in ("Cluster", "Label", "prediction_label")
                        if c in preds.columns
                    ),
                    None,
                )

            if cluster_col is not None:
                mal["__cluster__"] = preds[cluster_col].map(_normalize_cluster_id)
                mal = mal.dropna(subset=["__cluster__"]).copy()
                mal["__cluster__"] = mal["__cluster__"].astype(int)
                # Majority vote: cluster id -> most frequent ground-truth profile
                cluster_map = (
                    mal.groupby("__cluster__")["threat_profile"]
                    .agg(lambda s: s.value_counts().idxmax())
                    .to_dict()
                )
        except Exception as e:
            st.warning(f"Unable to build cluster mapping automatically: {e}")

    return cls_model, clu_model, plot, cluster_map, expected_cols


cls_model, clu_model, feature_plot, CLUSTER_MAP, CLUSTER_EXPECTED_COLS = load_assets()

if not cls_model:
    st.error(
        "Model not found. Please wait for the initial training to complete, or check the container logs with `make logs` if the error persists."
    )
    st.stop()

# --- Sidebar for Inputs ---
with st.sidebar:
    st.title("🔬 URL Feature Input")
    st.write("Describe the characteristics of a suspicious URL below.")

    form_values = {
        "url_length": st.select_slider(
            "URL Length", options=["Short", "Normal", "Long"], value="Long"
        ),
        "ssl_state": st.select_slider(
            "SSL Certificate Status",
            options=["Trusted", "Suspicious", "None"],
            value="Suspicious",
        ),
        "sub_domain": st.select_slider(
            "Sub-domain Complexity", options=["None", "One", "Many"], value="One"
        ),
        "prefix_suffix": st.checkbox("URL has a Prefix/Suffix (e.g.,'-')", value=True),
        "has_ip": st.checkbox("URL uses an IP Address", value=False),
        "short_service": st.checkbox("Is it a shortened URL", value=False),
        "at_symbol": st.checkbox("URL contains '@' symbol", value=False),
        "abnormal_url": st.checkbox("Is it an abnormal URL", value=True),
        # NEW: enrichment feature used by clustering / attribution
        "has_political_keyword": st.checkbox(
            "Contains political / activist keywords",
            value=False,
            help="e.g., topical slogans, movement names, or activist language",
        ),
    }

    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button(
        "💥 Analyze & Initiate Response", use_container_width=True, type="primary"
    )

# --- Main Page ---
st.title("🛡️ GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info(
        "Please provide the URL features in the sidebar and click 'Analyze' to begin."
    )
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(
            feature_plot,
            caption="Feature importance from the trained model. This shows which features the model weighs most heavily when making a prediction.",
        )
    st.stop()

# --- Data Preparation (must match classifier training schema) ---
# ...existing imports and setup...

# --- Data Preparation (must match classifier training schema) ---
input_dict = {
    "having_IP_Address": 1 if form_values["has_ip"] else -1,
    "URL_Length": -1
    if form_values["url_length"] == "Short"
    else (0 if form_values["url_length"] == "Normal" else 1),
    "Shortining_Service": 1 if form_values["short_service"] else -1,
    "having_At_Symbol": 1 if form_values["at_symbol"] else -1,
    "double_slash_redirecting": -1,
    "Prefix_Suffix": 1 if form_values["prefix_suffix"] else -1,
    "having_Sub_Domain": -1
    if form_values["sub_domain"] == "None"
    else (0 if form_values["sub_domain"] == "One" else 1),
    "SSLfinal_State": -1
    if form_values["ssl_state"] == "None"
    else (0 if form_values["ssl_state"] == "Suspicious" else 1),
    "Abnormal_URL": 1 if form_values["abnormal_url"] else -1,
    "URL_of_Anchor": 0,
    "Links_in_tags": 0,
    "SFH": 0,
    # NEW: now part of the classifier input
    "has_political_keyword": 1 if form_values["has_political_keyword"] else 0,
}
input_data = pd.DataFrame([input_dict])

# Simple risk contribution for visualization
risk_scores = {
    "Bad SSL": 25 if input_dict["SSLfinal_State"] < 1 else 0,
    "Abnormal URL": 20 if input_dict["Abnormal_URL"] == 1 else 0,
    "Prefix/Suffix": 15 if input_dict["Prefix_Suffix"] == 1 else 0,
    "Shortened URL": 15 if input_dict["Shortining_Service"] == 1 else 0,
    "Complex Sub-domain": 10 if input_dict["having_Sub_Domain"] == 1 else 0,
    "Long URL": 10 if input_dict["URL_Length"] == 1 else 0,
    "Uses IP Address": 5 if input_dict["having_IP_Address"] == 1 else 0,
}
risk_df = pd.DataFrame(
    list(risk_scores.items()), columns=["Feature", "Risk Contribution"]
).sort_values("Risk Contribution", ascending=False)

# --- Analysis Workflow ---
with st.status("Executing SOAR playbook...", expanded=True) as status:
    st.write(
        "▶️ **Step 1: Predictive Analysis** - Running features through classification model."
    )
    time.sleep(0.5)
    prediction = predict_cls(cls_model, data=input_data)
    is_malicious = prediction["prediction_label"].iloc[0] == 1

    verdict = "MALICIOUS" if is_malicious else "BENIGN"
    st.write(f"▶️ **Step 2: Verdict Interpretation** - Model predicts **{verdict}**.")
    time.sleep(0.5)

    actor_profile = None
    cluster_id = None

    if is_malicious and clu_model:
        st.write(
            "▶️ **Step 3: Threat Attribution** - Assigning threat actor profile via clustering."
        )

        # Prepare a single-row input for clustering; fill all expected cols
        if CLUSTER_EXPECTED_COLS:
            row = {col: 0 for col in CLUSTER_EXPECTED_COLS}
            # copy classifier features that the clusterer also uses
            for k, v in input_dict.items():
                if k in row:
                    row[k] = v
            # include enrichment feature if the clusterer expects it
            if "has_political_keyword" in row:
                row["has_political_keyword"] = (
                    1 if form_values["has_political_keyword"] else 0
                )
            cluster_input = pd.DataFrame([row])
        else:
            # Fallback: pass what we have; also include enrichment feature
            cluster_input = pd.DataFrame(
                [
                    {
                        **input_dict,
                        "has_political_keyword": 1
                        if form_values["has_political_keyword"]
                        else 0,
                    }
                ]
            )

        try:
            clu_pred = predict_cluster(clu_model, data=cluster_input)
            cluster_col = next(
                (
                    c
                    for c in ("Cluster", "Label", "prediction_label")
                    if c in clu_pred.columns
                ),
                None,
            )
            if cluster_col is None:
                from pycaret.clustering import assign_model as cluster_assign

                clu_pred = cluster_assign(clu_model, data=cluster_input)
                cluster_col = next(
                    (
                        c
                        for c in ("Cluster", "Label", "prediction_label")
                        if c in clu_pred.columns
                    ),
                    None,
                )

            raw_id = clu_pred[cluster_col].iloc[0] if cluster_col else None
            # normalize id like "Cluster 0" -> 0
            m = re.search(r"-?\d+", str(raw_id)) if raw_id is not None else None
            cluster_id = int(m.group(0)) if m else None

            if cluster_id is not None and CLUSTER_MAP:
                base = CLUSTER_MAP.get(cluster_id)  # 'crime' | 'state' | 'hacktivist'
                if base:
                    actor_profile = PROFILE_DISPLAY.get(base, base)

            status.update(
                label="✅ SOAR Playbook Executed Successfully!",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            st.warning(f"Attribution step failed: {e}")
            status.update(
                label="⚠️ Classification done; attribution unavailable.",
                state="complete",
                expanded=False,
            )
    else:
        status.update(label="✅ Analysis Complete.", state="complete", expanded=False)

# --- Tabs for Organized Output ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 **Analysis Summary**",
        "📈 **Visual Insights**",
        "📜 **Prescriptive Plan**",
        "🕵️ **Threat Attribution**",
    ]
)

with tab1:
    st.subheader("Verdict and Key Findings")
    if is_malicious:
        st.error("**Prediction: Malicious Phishing URL**", icon="🚨")
    else:
        st.success("**Prediction: Benign URL**", icon="✅")

    st.metric(
        "Malicious Confidence Score",
        f"{prediction['prediction_score'].iloc[0]:.2%}"
        if is_malicious
        else f"{1 - prediction['prediction_score'].iloc[0]:.2%}",
    )
    st.caption("This score represents the model's confidence in its prediction.")

with tab2:
    st.subheader("Visual Analysis")
    st.write("#### Risk Contribution by Feature")
    st.bar_chart(risk_df.set_index("Feature"))
    st.caption(
        "A simplified view of which input features contributed most to a higher risk score."
    )

    if feature_plot:
        st.write("#### Model Feature Importance (Global)")
        st.image(
            feature_plot,
            caption="This plot shows which features the model found most important *overall* during its training.",
        )

with tab3:
    st.subheader("Actionable Response Plan")
    prescription = None
    if is_malicious:
        try:
            prescription = generate_prescription(
                genai_provider, {k: v for k, v in input_dict.items()}
            )
            st.success(
                "A prescriptive response plan has been generated by the AI.", icon="🤖"
            )
            st.json(prescription, expanded=False)
            st.write("#### Recommended Actions (for Security Analyst)")
            for i, action in enumerate(prescription.get("recommended_actions", []), 1):
                st.markdown(f"**{i}.** {action}")
            st.write("#### Communication Draft (for End-User/Reporter)")
            st.text_area(
                "Draft", prescription.get("communication_draft", ""), height=150
            )
        except Exception as e:
            st.error(f"Failed to generate prescription: {e}")
    else:
        st.info(
            "No prescriptive plan was generated because the URL was classified as benign."
        )

with tab4:
    st.subheader("Threat Attribution")
    if not is_malicious:
        st.info("Attribution runs only when the URL is predicted **MALICIOUS**.")
    elif not clu_model:
        st.warning("Clustering model not found. Train it first to enable attribution.")
    else:
        if actor_profile:
            st.success(f"**Predicted Actor Profile:** {actor_profile}")
            if cluster_id is not None:
                st.caption(f"(Cluster ID: {cluster_id})")
            st.caption(
                "Political/activist keywords present: "
                + ("Yes" if form_values["has_political_keyword"] else "No")
            )
            desc = PROFILE_DESC.get(actor_profile, "")
            if desc:
                st.write(desc)
        else:
            st.info(
                "Attribution is available, but the model could not confidently map this input to a profile."
            )
