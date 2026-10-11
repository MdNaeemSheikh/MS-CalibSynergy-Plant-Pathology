# MS-CalibSynergy-Plant-Pathology
# 🌿 MS-CalibSynergy-Net: Calibrated Multi-Scale Deep Learning & Explainable AI for Plant Pathology Diagnosis

[![Hugging Face Model Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights%20(FP16)-yellow)](https://huggingface.co/NaeemSheikh123/ms-calib-synergy-plant-disease)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](./Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

**Author:** Md Naeem Sheikh  
**Architecture:** Multi-Scale Calibrated Synergy Network (`DenseNet-121` + `EfficientNet-B0` + `ResNet-50` + `Channel-Attention Gated Meta-Learner` + `Post-Hoc Temperature Scaling`)

---

## 📌 Project Overview
Standard deep learning ensembles in agricultural pathology often improve top-1 accuracy at the expense of severe confidence miscalibration (overconfidence) and data leakage across augmented splits. **MS-CalibSynergy-Net** addresses these limitations through an end-to-end diagnostic and uncertainty-aware triage pipeline:

1. **Leakage-Free Stratified Group Partitioning:** Audited **3,600 foliar images** across **15 disease classes** (Bell Pepper, Potato, Tomato), removed $3\sigma$ pixel intensity outliers, and enforced `StratifiedGroupKFold` source-specimen isolation (**0.0% cross-split leaf overlap** across `2,525 Train / 531 Validation / 526 Test` images)[cite: 11].
2. **Probability Calibration & Multi-Scale Fusion:** While standard attention stacking improved test accuracy to `96.20%`, Expected Calibration Error (ECE) increased to `0.1064`[cite: 3]. By combining 4-view test-time spatial representations with L-BFGS temperature scaling, **MS-CalibSynergy-Net** achieves **96.39% Test Accuracy**, **0.9597 Macro-F1**, and **0.9614 MCC** while reducing **ECE by 88% to `0.0124`**[cite: 6].
3. **Explainable AI (XAI) & Selective Triage:** Integrates **Dual-Backbone Grad-CAM**, **Global/Local SHAP**, and **LIME Superpixel Regression** alongside Shannon-entropy selective abstention and bilateral sensor-noise filtering.
4. **Production Deployment:** Packaged as a **69 MB FP16 unified checkpoint** on Hugging Face Hub and served via both a **Dockerized FastAPI REST API (`/api/v1/predict`)** and an interactive **Streamlit Web Application**.

---

## 📊 1. Dataset Quality Audit & Group-Isolated Split Pipeline
![Preprocessing Executive Summary](research_artifacts/figures/preprocessing_executive_summary.png)

---

## 🏆 2. Model Ablation Leaderboard (Held-Out Test Set, $N = 526$)

| Rank | Architecture | Params (M) | Test Accuracy (%) | Macro Precision | Macro Recall | Macro F1 | MCC | ROC-AUC (OvR) | Calibration Error (ECE $\downarrow$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Proposed: MS-CalibSynergy-Net** | **36.74** | **96.39%** | **0.9658** | **0.9562** | **0.9597** | **0.9614** | **0.9989** | **0.0124** |
| 2 | Ensemble-2: Attention Stacking | 36.74 | 96.20% | 0.9639 | 0.9541 | 0.9569 | 0.9593 | 0.9985 | 0.1064 |
| 3 | Ensemble-1: Weighted Soft-Voting | 36.73 | 96.01% | 0.9561 | 0.9499 | 0.9505 | 0.9572 | 0.9987 | 0.0958 |
| 4 | DenseNet-121 Baseline | 7.49 | 95.63% | 0.9533 | 0.9489 | 0.9491 | 0.9531 | 0.9988 | 0.0773 |
| 5 | EfficientNet-B0 Baseline | 4.67 | 93.35% | 0.9250 | 0.9245 | 0.9221 | 0.9286 | 0.9968 | 0.0781 |
| 6 | ResNet-50 Baseline | 24.57 | 92.40% | 0.9226 | 0.9150 | 0.9175 | 0.9182 | 0.9942 | 0.0870 |

* **Bootstrap Confidence Interval (1,000 Iterations):** `96.38%` Mean Accuracy (`95% CI: [94.68%, 97.91%]`)[cite: 12].

![Model Ablation Summary](research_artifacts/figures/definitive_6model_q1_summary_clean.png)

---

## 🔍 3. Calibration Analysis, Selective Triage & Explainable AI (XAI)

### A. Reliability Calibration Diagram, Selective Risk-Coverage Curve & Grad-CAM Attribution
![Calibration and GradCAM](research_artifacts/figures/q1_novelty_calibration_and_gradcam.png)

### B. Global & Local SHAP Feature Attribution
![SHAP Global and Local Summary](research_artifacts/figures/shap_global_and_local_summary_clean.png)

### C. Multi-Crop LIME Superpixel Attribution (Pepper, Potato & Tomato)
![LIME Multi-Crop Audit](research_artifacts/figures/lime_proposed_model_multicrop_clean.png)

### D. Bootstrap Distribution, Environmental Disturbance Tolerance & Edge Latency Profiling
![Robustness and Statistical Rigor](research_artifacts/figures/q1_generalization_and_statistical_rigor_clean.png)

---

## 🚀 4. Local Setup & Containerized Deployment

### Run Interactive Streamlit Web App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
