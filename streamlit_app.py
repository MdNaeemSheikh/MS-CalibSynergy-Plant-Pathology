import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="MS-CalibSynergy-Net | Plant Pathology AI",
    page_icon="🌿",
    layout="wide"
)

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

st.title("🌿 MS-CalibSynergy-Net: Calibrated Plant Pathology Diagnostic System")
st.markdown(
    "**Test Accuracy:** `96.39%` | **Macro-F1:** `0.9597` | **MCC:** `0.9614` | "
    "**Expected Calibration Error (ECE):** `0.012` | **Architecture:** `DenseNet121 + EfficientNetB0 + ResNet50 + Attention Meta-Learner`"
)

dn_model, ef_model, rn_model, meta_net, CFG, device = load_synergy_system()
SHORT_NAMES = CFG["short_names"]
MEAN = np.array(CFG["dataset_mean"], dtype=np.float32)
STD = np.array(CFG["dataset_std"], dtype=np.float32)

col_left, col_right = st.columns([1, 1.25])

with col_left:
    st.subheader("1. Upload Foliar Specimen")
    uploaded_file = st.file_uploader("Choose a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    use_bilateral = st.checkbox("Enable Bilateral Sensor-Noise Defense Gate", value=True)

    sample_files = [f for f in os.listdir(".") if f.startswith("sample_") and f.endswith(".JPG")]
    selected_sample = st.selectbox("Or test with a held-out sample specimen:", ["None"] + sample_files)

    rgb_input = None
    if uploaded_file is not None:
        rgb_input = np.array(Image.open(uploaded_file).convert("RGB"))
    elif selected_sample != "None":
        rgb_input = np.array(Image.open(selected_sample).convert("RGB"))

    if rgb_input is not None:
        st.image(rgb_input, caption="Input Leaf Specimen", use_container_width=True)

with col_right:
    st.subheader("2. Calibrated Diagnosis & Dual-Backbone Explainability")
    if rgb_input is not None:
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
            cal_logits = torch.log(torch.clamp(uncal, 1e-8, 1.0)) / CFG["opt_temp"]
            probs = F.softmax(cal_logits, dim=1).squeeze(0).cpu().numpy()

        top_cls = int(np.argmax(probs))
        conf = float(probs[top_cls])
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))

        cam_dn = compute_gradcam(dn_model, dn_model.backbone.features.norm5, x, top_cls)
        cam_ef = compute_gradcam(ef_model, ef_model.backbone.conv_head, x, top_cls)
        cam_syn = 0.55 * cam_dn + 0.45 * cam_ef
        cam_syn = (cam_syn - cam_syn.min()) / (cam_syn.max() - cam_syn.min() + 1e-8)

        heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_syn), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_resized, 0.55, heat, 0.45, 0)

        c1, c2 = st.columns(2)
        with c1:
            st.image(overlay, caption="Synergistic Grad-CAM Lesion Attribution", use_container_width=True)
        with c2:
            st.metric("Predicted Pathology", SHORT_NAMES[top_cls])
            st.metric("Calibrated Confidence (T-Scaled)", f"{conf*100:.2f}%")
            st.metric("Predictive Shannon Entropy", f"{entropy:.3f}")

        if entropy < CFG["entropy_threshold"]:
            st.success(f"PASS: High-Confidence Automated Diagnosis ({SHORT_NAMES[top_cls]})")
        else:
            st.warning("TRIAGE ALERT: Elevated Diagnostic Uncertainty — Refer Specimen to Agronomist")

        st.markdown("#### Top-5 Temperature-Calibrated Class Probabilities")
        top5_idx = np.argsort(probs)[::-1][:5]
        for idx_t in top5_idx:
            st.progress(float(probs[idx_t]), text=f"{SHORT_NAMES[idx_t]}: {probs[idx_t]*100:.2f}%")
