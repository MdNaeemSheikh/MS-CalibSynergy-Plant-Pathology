import os
import json
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import timm
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="MS-CalibSynergy-Net | Plant Pathology Diagnostic System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border: 1px solid rgba(46, 204, 113, 0.35);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #d1f2eb;
        margin-bottom: 14px;
    }
    .badge-row { display: flex; flex-wrap: wrap; gap: 10px; }
    .metric-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f8f9f9;
    }
    .metric-pill span { color: #2ecc71; font-weight: 800; }
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; color: #8b949e; margin-bottom: 4px; }
    .kpi-value { font-size: 1.45rem; font-weight: 800; color: #f0f6fc; }
    .kpi-sub { font-size: 0.76rem; color: #58a6ff; margin-top: 3px; }
    .triage-pass {
        background: rgba(35, 134, 54, 0.18);
        border-left: 5px solid #2ea043;
        padding: 14px 18px;
        border-radius: 8px;
        color: #aff5b4;
        font-weight: 600;
        margin: 12px 0;
    }
    .triage-warn {
        background: rgba(210, 153, 34, 0.18);
        border-left: 5px solid #d29922;
        padding: 14px 18px;
        border-radius: 8px;
        color: #f8e3a1;
        font-weight: 600;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

HF_REPO_ID = os.getenv("HF_REPO_ID", "NaeemSheikh123/ms-calib-synergy-plant-disease")
MODEL_FILENAME = "ms_calib_synergy_fp16.pt"

@st.cache_resource
def load_synergy_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_path = MODEL_FILENAME
    if not os.path.exists(local_path):
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)

    bundle = torch.load(local_path, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    num_classes = cfg["num_classes"]

    class TransferLearningModel(nn.Module):
        def __init__(self, arch_name):
            super().__init__()
            self.backbone = timm.create_model(arch_name, pretrained=False, num_classes=0)
            in_features = self.backbone.num_features
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        def forward(self, x):
            return self.classifier(self.backbone(x))

    class AttentionStackingMetaLearner(nn.Module):
        def __init__(self, in_dim=3 * num_classes):
            super().__init__()
            self.attention_gate = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2), nn.SiLU(),
                nn.Linear(in_dim // 2, in_dim), nn.Sigmoid()
            )
            self.meta_head = nn.Sequential(
                nn.Linear(in_dim, 64), nn.BatchNorm1d(64),
                nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.meta_head(x * self.attention_gate(x) + x)

    dn = TransferLearningModel("densenet121").to(device)
    ef = TransferLearningModel("efficientnet_b0").to(device)
    rn = TransferLearningModel("resnet50").to(device)
    meta = AttentionStackingMetaLearner().to(device)

    dn.load_state_dict({k: v.float() for k, v in bundle["densenet121_fp16"].items()})
    ef.load_state_dict({k: v.float() for k, v in bundle["efficientnet_b0_fp16"].items()})
    rn.load_state_dict({k: v.float() for k, v in bundle["resnet50_fp16"].items()})
    meta.load_state_dict(bundle["meta_model_fp32"])

    for m in [dn, ef, rn, meta]:
        m.eval()

    return dn, ef, rn, meta, cfg, device

def compute_gradcam(model, target_layer, x_tensor, target_cls):
    acts, grads = [], []
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    model.zero_grad()
    out = model(x_tensor)
    out[0, target_cls].backward()
    h1.remove()
    h2.remove()
    w = torch.mean(grads[0], dim=(2, 3), keepdim=True)
    cam = F.relu(torch.sum(w * acts[0], dim=1).squeeze(0)).detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def make_heatmap_overlay(rgb_224, cam_norm):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_224, 0.52, heat, 0.48, 0)

dn_model, ef_model, rn_model, meta_net, CFG, device = load_synergy_system()
SHORT_NAMES = CFG["short_names"]
MEAN = np.array(CFG["dataset_mean"], dtype=np.float32)
STD = np.array(CFG["dataset_std"], dtype=np.float32)

with st.sidebar:
    st.markdown("### ⚙️ Diagnostic Control Panel")
    sample_files = sorted([f for f in os.listdir(".") if f.startswith("sample_") and f.upper().endswith(".JPG")])

    input_mode = st.radio(
        "Specimen Input Source:",
        ["Test Set Reference Specimens", "Upload Custom Image"],
        index=0
    )

    rgb_input = None
    specimen_label = "Uploaded Specimen"

    if input_mode == "Test Set Reference Specimens" and len(sample_files) > 0:
        selected_sample = st.selectbox("Select Reference Leaf:", sample_files, index=0)
        rgb_input = np.array(Image.open(selected_sample).convert("RGB"))
        specimen_label = selected_sample
    else:
        uploaded_file = st.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            rgb_input = np.array(Image.open(uploaded_file).convert("RGB"))
            specimen_label = uploaded_file.name
        elif len(sample_files) > 0:
            rgb_input = np.array(Image.open(sample_files[0]).convert("RGB"))
            specimen_label = f"{sample_files[0]} (Default)"

    st.markdown("---")
    st.markdown("### 🛡️ Inference Configuration")
    use_bilateral = st.toggle("Bilateral Sensor-Noise Filter", value=True, help="Applies edge-preserving bilateral filtering (d=5, sigma=35) prior to normalization.")
    use_4view_tta = st.toggle("4-View Test-Time Augmentation", value=True, help="Averages predictions across Original, Horizontal Flip, Vertical Flip, and 90° Rotation.")
    custom_temp = st.slider("Calibration Temperature (T)", min_value=0.20, max_value=2.00, value=float(round(CFG["opt_temp"], 2)), step=0.05)
    entropy_thresh = st.slider("Triage Entropy Threshold (τ)", min_value=0.15, max_value=1.20, value=float(CFG.get("entropy_threshold", 0.65)), step=0.05)

    st.markdown("---")
    st.markdown("**Developed by:**\n**Md Naeem Sheikh**\n*M.Sc. in CSE (Data Science)*")
    st.caption("Weights Registry: `NaeemSheikh123/ms-calib-synergy-plant-disease`")

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌿 MS-CalibSynergy-Net: Plant Pathology Diagnostic & Explainable AI System</div>
    <div class="hero-sub">Multi-Scale Calibrated Ensemble Architecture with Dual-Backbone Grad-CAM Attribution & Uncertainty-Aware Triage</div>
    <div class="badge-row">
        <div class="metric-pill">Test Accuracy: <span>96.39%</span></div>
        <div class="metric-pill">Macro F1: <span>0.9597</span></div>
        <div class="metric-pill">MCC: <span>0.9614</span></div>
        <div class="metric-pill">Calibration Error (ECE): <span>0.0124</span></div>
        <div class="metric-pill">95% Bootstrap CI: <span>[94.68%, 97.91%]</span></div>
        <div class="metric-pill">Split Validation: <span>Group-Isolated (0 Leakage)</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_studio, tab_benchmarks, tab_api = st.tabs([
    "🔬 Diagnostic & Explainability Studio",
    "📊 Experimental Benchmarks & XAI Audit",
    "⚡ FastAPI & Docker Deployment"
])

with tab_studio:
    if rgb_input is not None:
        t0 = time.time()
        img_resized = cv2.resize(rgb_input, (224, 224))
        img_proc = cv2.bilateralFilter(img_resized, d=5, sigmaColor=35, sigmaSpace=35) if use_bilateral else img_resized

        norm_img = ((img_proc.astype(np.float32) / 255.0) - MEAN) / STD
        x = torch.tensor(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)
        x_hf = torch.flip(x, dims=[3])
        x_vf = torch.flip(x, dims=[2])
        x_r90 = torch.rot90(x, k=1, dims=[2, 3])

        with torch.no_grad():
            def mv_p(m):
                p1 = F.softmax(m(x), dim=1)
                if not use_4view_tta:
                    return p1, p1
                p_mv = 0.45 * p1 + 0.25 * F.softmax(m(x_hf), dim=1) + 0.15 * F.softmax(m(x_vf), dim=1) + 0.15 * F.softmax(m(x_r90), dim=1)
                return p1, p_mv

            dn_p1, dn_mv = mv_p(dn_model)
            ef_p1, ef_mv = mv_p(ef_model)
            rn_p1, rn_mv = mv_p(rn_model)

            w = CFG["opt_w"]
            meta_orig = F.softmax(meta_net(torch.cat([torch.log(torch.clamp(p, 1e-6, 1.0)) for p in [dn_p1, ef_p1, rn_p1]], dim=1)), dim=1)
            stack_orig = 0.65 * meta_orig + 0.35 * (w[0]*dn_p1 + w[1]*ef_p1 + w[2]*rn_p1)

            meta_mv = F.softmax(meta_net(torch.cat([torch.log(torch.clamp(p, 1e-6, 1.0)) for p in [dn_mv, ef_mv, rn_mv]], dim=1)), dim=1)
            soft_mv = w[0]*dn_mv + w[1]*ef_mv + w[2]*rn_mv

            uncal = CFG["best_alpha"]*stack_orig + CFG["best_beta"]*soft_mv + CFG["best_gamma"]*meta_mv
            cal_logits = torch.log(torch.clamp(uncal, 1e-8, 1.0)) / custom_temp
            probs = F.softmax(cal_logits, dim=1).squeeze(0).cpu().numpy()

            dn_prob_vec = dn_mv.squeeze(0).cpu().numpy()
            ef_prob_vec = ef_mv.squeeze(0).cpu().numpy()
            rn_prob_vec = rn_mv.squeeze(0).cpu().numpy()

        top_cls = int(np.argmax(probs))
        conf = float(probs[top_cls])
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
        max_entropy = float(np.log(len(SHORT_NAMES)))

        cam_dn = compute_gradcam(dn_model, dn_model.backbone.features.norm5, x, top_cls)
        cam_ef = compute_gradcam(ef_model, ef_model.backbone.conv_head, x, top_cls)
        cam_syn = 0.55 * cam_dn + 0.45 * cam_ef
        cam_syn = (cam_syn - cam_syn.min()) / (cam_syn.max() - cam_syn.min() + 1e-8)

        cos_sim = float(np.dot(cam_dn.flatten(), cam_ef.flatten()) / (np.linalg.norm(cam_dn.flatten()) * np.linalg.norm(cam_ef.flatten()) + 1e-8))
        infer_ms = (time.time() - t0) * 1000.0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Predicted Classification</div><div class="kpi-value" style="color:#2ecc71;">{SHORT_NAMES[top_cls]}</div><div class="kpi-sub">File: {specimen_label[:26]}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Calibrated Confidence</div><div class="kpi-value">{conf*100:.2f}%</div><div class="kpi-sub">Temperature Scaling (T = {custom_temp:.2f})</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Predictive Entropy H(p)</div><div class="kpi-value">{entropy:.3f} <span style="font-size:0.9rem;color:#8b949e;">/ {max_entropy:.2f}</span></div><div class="kpi-sub">Threshold τ = {entropy_thresh:.2f}</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Spatial CAM Similarity</div><div class="kpi-value">{cos_sim:.3f}</div><div class="kpi-sub">Inference Time: {infer_ms:.0f} ms ({device.type.upper()})</div></div>', unsafe_allow_html=True)

        if entropy < entropy_thresh:
            st.markdown(f'<div class="triage-pass">✅ <b>AUTOMATED DIAGNOSIS ACCEPTED:</b> Predictive entropy (H = {entropy:.3f}) is below the referral threshold (τ = {entropy_thresh:.2f}). High-confidence prediction verified for <b>{SHORT_NAMES[top_cls]}</b>.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="triage-warn">⚠️ <b>UNCERTAINTY TRIAGE ALERT:</b> Elevated predictive entropy detected (H = {entropy:.3f} ≥ τ = {entropy_thresh:.2f}). Specimen flagged for manual agronomist inspection.</div>', unsafe_allow_html=True)

        st.markdown("#### 🔍 Multi-Backbone Grad-CAM Spatial Attribution")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.image(img_proc, caption=f"1. Input Image ({'Bilateral Filtered' if use_bilateral else 'Raw RGB'})", use_container_width=True)
        with v2:
            st.image(make_heatmap_overlay(img_resized, cam_dn), caption="2. DenseNet-121 Grad-CAM (Texture Stream)", use_container_width=True)
        with v3:
            st.image(make_heatmap_overlay(img_resized, cam_ef), caption="3. EfficientNet-B0 Grad-CAM (Morphology Stream)", use_container_width=True)
        with v4:
            st.image(make_heatmap_overlay(img_resized, cam_syn), caption=f"4. Combined Attribution Map (Cosine Sim: {cos_sim:.3f})", use_container_width=True)

        st.markdown("---")
        b_left, b_right = st.columns([1.15, 1])

        with b_left:
            st.markdown("#### 📈 Top-5 Calibrated Class Probabilities")
            top5_idx = np.argsort(probs)[::-1][:5]
            for rank_i, idx_t in enumerate(top5_idx, 1):
                p_val = float(probs[idx_t])
                st.progress(p_val, text=f"#{rank_i} {SHORT_NAMES[idx_t]} — {p_val*100:.2f}%")

        with b_right:
            st.markdown(f"#### 🧠 Per-Backbone Probability Breakdown (`{SHORT_NAMES[top_cls]}`)")
            stream_df = pd.DataFrame({
                "Model Component": [
                    "DenseNet-121 (Fine-Grained Texture Stream)",
                    "EfficientNet-B0 (Multi-Scale Morphology Stream)",
                    "ResNet-50 (Deep Residual Stream)",
                    "MS-CalibSynergy-Net (Calibrated Ensemble Output)"
                ],
                "Confidence (%)": [
                    round(float(dn_prob_vec[top_cls]) * 100, 2),
                    round(float(ef_prob_vec[top_cls]) * 100, 2),
                    round(float(rn_prob_vec[top_cls]) * 100, 2),
                    round(conf * 100, 2)
                ]
            })
            st.dataframe(stream_df, use_container_width=True, hide_index=True)

            report_payload = {
                "specimen": specimen_label,
                "predicted_class": SHORT_NAMES[top_cls],
                "calibrated_confidence_pct": round(conf * 100, 2),
                "shannon_entropy": round(entropy, 4),
                "spatial_consensus_cosine_sim": round(cos_sim, 4),
                "triage_decision": "ACCEPTED" if entropy < entropy_thresh else "MANUAL_REVIEW_REQUIRED",
                "top5_probabilities": {SHORT_NAMES[i]: round(float(probs[i]), 4) for i in top5_idx}
            }
            st.download_button(
                label="📥 Export Diagnostic Report (JSON)",
                data=json.dumps(report_payload, indent=4),
                file_name=f"diagnosis_{SHORT_NAMES[top_cls].replace(': ', '_').replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

with tab_benchmarks:
    st.markdown("### 📊 Model Ablation Leaderboard & Explainability Audit")
    csv_path = "research_artifacts/tables/definitive_6model_q1_leaderboard.csv"
    if os.path.exists(csv_path):
        st.markdown("#### Held-Out Test Set Evaluation Across 6 Architectures ($N = 526$)")
        df_lead = pd.read_csv(csv_path)
        st.dataframe(df_lead, use_container_width=True, hide_index=True)

    fig_gallery = [
        ("research_artifacts/figures/definitive_6model_q1_summary_clean.png", "1. Multi-Model Test Accuracy Ranking & Calibration Trade-off (Macro-F1 vs. ECE)"),
        ("research_artifacts/figures/q1_novelty_calibration_and_gradcam.png", "2. Reliability Calibration Curve, Selective Risk-Coverage Triage & Grad-CAM Attribution"),
        ("research_artifacts/figures/shap_global_and_local_summary_clean.png", "3. Global & Local SHAP Feature Attribution Across Ensemble Streams"),
        ("research_artifacts/figures/lime_proposed_model_multicrop_clean.png", "4. Multi-Crop LIME Superpixel Attribution Audit (Bell Pepper, Potato & Tomato)"),
        ("research_artifacts/figures/q1_generalization_and_statistical_rigor_clean.png", "5. Bootstrap Confidence Intervals, Noise Robustness & Hardware Latency Profiling"),
        ("research_artifacts/figures/preprocessing_executive_summary.png", "6. Dataset Quality Audit, 3-Sigma Outlier Removal & Group-Isolated Split Distribution")
    ]

    for fig_path, fig_title in fig_gallery:
        if os.path.exists(fig_path):
            st.markdown(f"#### {fig_title}")
            st.image(fig_path, use_container_width=True)
            st.markdown("---")

with tab_api:
    st.markdown("### ⚡ Containerized FastAPI Service & REST API Integration")
    st.markdown("""
    This repository includes a standalone **FastAPI REST service (`app.py`)** and **`Dockerfile`** for containerized inference and integration with external web or mobile clients.
    """)
    c_api1, c_api2 = st.columns(2)
    with c_api1:
        st.markdown("#### 🐳 1. Build & Run Docker Container")
        st.code("""docker build -t ms-calib-plant-api .
docker run -d -p 7860:7860 ms-calib-plant-api

# OpenAPI / Swagger UI Documentation:
# http://localhost:7860/docs""", language="bash")
    with c_api2:
        st.markdown("#### 🐍 2. Example Python Client Request (`POST /api/v1/predict`)")
        st.code("""import requests

url = "http://localhost:7860/api/v1/predict"
with open("sample_1_Pepper_Bacterial_spot.JPG", "rb") as f:
    res = requests.post(url, files={"file": f})

print(res.json())""", language="python")
