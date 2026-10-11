import os
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
from fpdf import FPDF

st.set_page_config(
    page_title="PhytoScan AI | Plant Pathology Diagnostic Portal",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px; }
    .app-header {
        background: linear-gradient(135deg, #0b192c 0%, #16325b 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .app-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 4px 0;
        letter-spacing: -0.3px;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #b0c4de;
        margin: 0;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(46, 204, 113, 0.07);
        border: 2px dashed #2ecc71;
        border-radius: 12px;
        padding: 12px 16px;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stFileUploader"]:hover {
        background: rgba(46, 204, 113, 0.12);
        border-color: #58d68d;
    }
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .kpi-label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.8px; color: #8b949e; margin-bottom: 4px; }
    .kpi-value { font-size: 1.35rem; font-weight: 800; color: #f0f6fc; }
    .kpi-sub { font-size: 0.74rem; color: #58a6ff; margin-top: 3px; }
    .triage-pass {
        background: rgba(35, 134, 54, 0.16);
        border-left: 4px solid #2ea043;
        padding: 12px 16px;
        border-radius: 8px;
        color: #aff5b4;
        font-weight: 600;
        margin: 14px 0;
    }
    .triage-warn {
        background: rgba(210, 153, 34, 0.16);
        border-left: 4px solid #d29922;
        padding: 12px 16px;
        border-radius: 8px;
        color: #f8e3a1;
        font-weight: 600;
        margin: 14px 0;
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
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_224, 0.52, heat, 0.48, 0)

def generate_pdf_report(specimen_name, pred_cls, conf_val, entropy_val, cos_sim, triage_status, top5_data, input_img_rgb, cam_overlay_rgb):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_fill_color(11, 25, 44)
    pdf.rect(0, 0, 210, 26, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(12, 6)
    pdf.cell(0, 7, "Plant Pathology Diagnostic Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(12)
    pdf.cell(0, 6, "Automated Foliar Disease Identification & Visual Attribution Analysis", new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(35, 35, 35)
    pdf.set_xy(12, 33)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "1. Diagnostic Summary", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9.5)
    info_lines = [
        f"Specimen Source: {specimen_name}",
        f"Identified Condition: {pred_cls}",
        f"Diagnostic Confidence: {conf_val * 100:.2f}%",
        f"Predictive Uncertainty (Entropy): {entropy_val:.3f}",
        f"Spatial Attribution Agreement: {cos_sim:.3f}",
        f"Clinical Status: {triage_status}"
    ]
    for line in info_lines:
        pdf.set_x(14)
        pdf.cell(0, 5.5, line, new_x="LMARGIN", new_y="NEXT")

    tmp_in = "temp_input_leaf.jpg"
    tmp_cam = "temp_cam_leaf.jpg"
    cv2.imwrite(tmp_in, cv2.cvtColor(input_img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(tmp_cam, cv2.cvtColor(cam_overlay_rgb, cv2.COLOR_RGB2BGR))

    pdf.set_xy(12, 74)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "2. Visual Evidence & Lesion Localization", new_x="LMARGIN", new_y="NEXT")

    pdf.image(tmp_in, x=20, y=82, w=75, h=75)
    pdf.image(tmp_cam, x=115, y=82, w=75, h=75)

    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_xy(20, 159)
    pdf.cell(75, 5, "Analyzed Foliar Specimen", align="C")
    pdf.set_xy(115, 159)
    pdf.cell(75, 5, "Grad-CAM Lesion Attribution Map", align="C")

    pdf.set_xy(12, 170)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "3. Top-5 Differential Probabilities", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(14, 178)
    pdf.cell(105, 6.5, "Pathology Classification", border=1)
    pdf.cell(55, 6.5, "Calibrated Probability", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for cls_name, p_val in top5_data:
        pdf.set_x(14)
        pdf.cell(105, 6, cls_name, border=1)
        pdf.cell(55, 6, f"{p_val * 100:.2f}%", border=1, new_x="LMARGIN", new_y="NEXT")

    for tmp_file in [tmp_in, tmp_cam]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    return bytes(pdf.output())

dn_model, ef_model, rn_model, meta_net, CFG, device = load_synergy_system()
SHORT_NAMES = CFG["short_names"]
MEAN = np.array(CFG["dataset_mean"], dtype=np.float32)
STD = np.array(CFG["dataset_std"], dtype=np.float32)

st.markdown("""
<div class="app-header">
    <div class="app-title">🌿 Plant Pathology Diagnostic & Visual Explainability Portal</div>
    <div class="app-subtitle">Multi-Scale Deep Learning Disease Identification, Lesion Localization & Uncertainty Triage</div>
</div>
""", unsafe_allow_html=True)

tab_studio, tab_benchmarks, tab_api = st.tabs([
    "🔬 Diagnostic Workspace",
    "📊 System Architecture & Validation",
    "⚡ REST API & Container Deployment"
])

with tab_studio:
    up_col, ref_col = st.columns([1.4, 1])

    sample_files = sorted([f for f in os.listdir(".") if f.startswith("sample_") and f.upper().endswith(".JPG")])

    with up_col:
        st.markdown("##### 📤 Upload Any Leaf Image for Instant Diagnosis")
        uploaded_file = st.file_uploader(
            "Drag and drop a leaf photo here, or click Browse Files (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )

    with ref_col:
        st.markdown("##### 🧪 Or Select a Reference Specimen & Configure Inference")
        selected_sample = st.selectbox(
            "Reference Specimen (Used when no custom image is uploaded):",
            sample_files if len(sample_files) > 0 else ["None"],
            index=0
        )
        with st.expander("⚙️ Advanced Inference & Triage Settings", expanded=False):
            use_bilateral = st.toggle("Bilateral Sensor-Noise Filter", value=True)
            use_4view_tta = st.toggle("4-View Test-Time Augmentation", value=True)
            custom_temp = st.slider("Calibration Temperature (T)", 0.20, 2.00, float(round(CFG["opt_temp"], 2)), 0.05)
            entropy_thresh = st.slider("Triage Uncertainty Threshold (τ)", 0.15, 1.20, float(CFG.get("entropy_threshold", 0.65)), 0.05)

    rgb_input = None
    specimen_label = "Custom Specimen"

    if uploaded_file is not None:
        rgb_input = np.array(Image.open(uploaded_file).convert("RGB"))
        specimen_label = f"Uploaded: {uploaded_file.name}"
    elif selected_sample != "None" and os.path.exists(selected_sample):
        rgb_input = np.array(Image.open(selected_sample).convert("RGB"))
        specimen_label = f"Reference: {selected_sample}"

    st.markdown("---")

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
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Diagnosed Pathology</div><div class="kpi-value" style="color:#2ecc71;">{SHORT_NAMES[top_cls]}</div><div class="kpi-sub">{specimen_label[:30]}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Diagnostic Confidence</div><div class="kpi-value">{conf*100:.2f}%</div><div class="kpi-sub">Calibrated Probability</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Uncertainty (Entropy)</div><div class="kpi-value">{entropy:.3f} <span style="font-size:0.85rem;color:#8b949e;">/ {max_entropy:.2f}</span></div><div class="kpi-sub">Triage Threshold τ = {entropy_thresh:.2f}</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Spatial Agreement</div><div class="kpi-value">{cos_sim:.3f}</div><div class="kpi-sub">Latency: {infer_ms:.0f} ms ({device.type.upper()})</div></div>', unsafe_allow_html=True)

        if entropy < entropy_thresh:
            triage_text = "AUTOMATED DIAGNOSIS VERIFIED"
            st.markdown(f'<div class="triage-pass">✅ <b>AUTOMATED DIAGNOSIS VERIFIED:</b> Predictive uncertainty (H = {entropy:.3f}) is within safe clinical bounds (τ = {entropy_thresh:.2f}). Condition identified as <b>{SHORT_NAMES[top_cls]}</b>.</div>', unsafe_allow_html=True)
        else:
            triage_text = "MANUAL AGRONOMIST REVIEW RECOMMENDED"
            st.markdown(f'<div class="triage-warn">⚠️ <b>MANUAL AGRONOMIST REVIEW RECOMMENDED:</b> Elevated diagnostic uncertainty detected (H = {entropy:.3f} ≥ τ = {entropy_thresh:.2f}). Please inspect lesion morphology manually.</div>', unsafe_allow_html=True)

        st.markdown("#### 🔍 Visual Lesion Localization (Grad-CAM Attribution)")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.image(img_proc, caption="1. Analyzed Specimen", use_container_width=True)
        with v2:
            st.image(make_heatmap_overlay(img_resized, cam_dn), caption="2. Fine Texture Stream (DenseNet-121)", use_container_width=True)
        with v3:
            st.image(make_heatmap_overlay(img_resized, cam_ef), caption="3. Morphology Stream (EfficientNet-B0)", use_container_width=True)
        with v4:
            cam_overlay_img = make_heatmap_overlay(img_resized, cam_syn)
            st.image(cam_overlay_img, caption="4. Combined Consensus Heatmap", use_container_width=True)

        st.markdown("---")
        b_left, b_right = st.columns([1.15, 1])

        with b_left:
            st.markdown("#### 📈 Differential Diagnosis (Top-5 Probabilities)")
            top5_idx = np.argsort(probs)[::-1][:5]
            top5_pairs = [(SHORT_NAMES[i], float(probs[i])) for i in top5_idx]
            for rank_i, (c_name, p_val) in enumerate(top5_pairs, 1):
                st.progress(p_val, text=f"#{rank_i} {c_name} — {p_val*100:.2f}%")

        with b_right:
            st.markdown(f"#### 🧠 Ensemble Stream Breakdown (`{SHORT_NAMES[top_cls]}`)")
            stream_df = pd.DataFrame({
                "Diagnostic Stream": [
                    "DenseNet-121 (Texture Stream)",
                    "EfficientNet-B0 (Morphology Stream)",
                    "ResNet-50 (Residual Stream)",
                    "Synergistic Calibrated Output"
                ],
                "Confidence (%)": [
                    round(float(dn_prob_vec[top_cls]) * 100, 2),
                    round(float(ef_prob_vec[top_cls]) * 100, 2),
                    round(float(rn_prob_vec[top_cls]) * 100, 2),
                    round(conf * 100, 2)
                ]
            })
            st.dataframe(stream_df, use_container_width=True, hide_index=True)

            pdf_report_bytes = generate_pdf_report(
                specimen_name=specimen_label,
                pred_cls=SHORT_NAMES[top_cls],
                conf_val=conf,
                entropy_val=entropy,
                cos_sim=cos_sim,
                triage_status=triage_text,
                top5_data=top5_pairs,
                input_img_rgb=img_proc,
                cam_overlay_rgb=cam_overlay_img
            )

            st.download_button(
                label="📄 Download Clinical Diagnostic Report (PDF)",
                data=pdf_report_bytes,
                file_name=f"Diagnostic_Report_{SHORT_NAMES[top_cls].replace(': ', '_').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

with tab_benchmarks:
    st.markdown("### 📊 System Validation & Explainability Artifacts")
    csv_path = "research_artifacts/tables/definitive_6model_q1_leaderboard.csv"
    if os.path.exists(csv_path):
        st.markdown("#### Multi-Architecture Evaluation Summary")
        df_lead = pd.read_csv(csv_path)
        st.dataframe(df_lead, use_container_width=True, hide_index=True)

    fig_gallery = [
        ("research_artifacts/figures/definitive_6model_q1_summary_clean.png", "1. Architecture Comparison & Calibration Analysis"),
        ("research_artifacts/figures/q1_novelty_calibration_and_gradcam.png", "2. Reliability Curve, Selective Triage & Grad-CAM Verification"),
        ("research_artifacts/figures/shap_global_and_local_summary_clean.png", "3. Global & Local SHAP Stream Attribution"),
        ("research_artifacts/figures/lime_proposed_model_multicrop_clean.png", "4. Multi-Crop LIME Superpixel Attribution"),
        ("research_artifacts/figures/q1_generalization_and_statistical_rigor_clean.png", "5. Bootstrap Confidence, Noise Tolerance & Latency Profile"),
        ("research_artifacts/figures/preprocessing_executive_summary.png", "6. Dataset Quality Assurance & Split Distribution")
    ]

    for fig_path, fig_title in fig_gallery:
        if os.path.exists(fig_path):
            st.markdown(f"#### {fig_title}")
            st.image(fig_path, use_container_width=True)
            st.markdown("---")

with tab_api:
    st.markdown("### ⚡ Containerized FastAPI Service & REST API Integration")
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
