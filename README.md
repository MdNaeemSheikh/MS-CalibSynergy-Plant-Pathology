<div align="center">

<!-- Animated Waving Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:1b4332,100:2d6a4f&height=210&section=header&text=%F0%9F%8C%BF%20MS-CalibSynergy-Net&fontSize=44&fontColor=ffffff&animation=twinkling&fontAlignY=36&desc=Calibrated%20Multi-Scale%20Plant%20Pathology%20Diagnosis%20%26%20Explainable%20AI%20System&descAlignY=58&descSize=17" width="100%" alt="Header Banner"/>

<!-- Animated Typing Subtitle -->
<a href="https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=18&duration=3200&pause=1000&color=2ECC71&center=true&vCenter=true&width=750&lines=%F0%9F%8C%B1+Live+Diagnostic+Portal+%7C+15+Foliar+Crop+Pathologies;%F0%9F%94%AC+Dual-Backbone+Grad-CAM%2C+SHAP+%26+LIME+Visual+Verification;%F0%9F%93%8A+96.39%25+Test+Accuracy+%7C+0.0124+Expected+Calibration+Error;%E2%9A%A1+69+MB+FP16+Weights+%7C+Streamlit+Cloud+%2B+FastAPI+%2B+Docker" alt="Typing Animation" />
</a>

<br/>

[![Live Web App](https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Web%20Portal-Streamlit%20Cloud-FF4B4B?style=for-the-badge)](https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/)
[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Model%20Hub-FP16%20Weights%20(69%20MB)-FFD21E?style=for-the-badge)](https://huggingface.co/NaeemSheikh123/ms-calib-synergy-plant-disease)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](./app.py)
[![Docker](https://img.shields.io/badge/Docker-Cloud%20Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](./Dockerfile)

</div>

---

## 🌐 Live Interactive Web Application
You can test the deployed model in real time with custom leaf photos or held-out test specimens, inspect multi-backbone Grad-CAM heatmaps, adjust calibration temperature and triage thresholds, and download clinical PDF reports:

👉 **Launch Live Diagnostic Portal:** [**https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/**](https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/)

* **Cloud Model Registry (FP16 Checkpoint):** [`NaeemSheikh123/ms-calib-synergy-plant-disease`](https://huggingface.co/NaeemSheikh123/ms-calib-synergy-plant-disease)
* **Complete Research Notebook:** [`01_MS_CalibSynergy_Net_End_to_End_Research_Pipeline.ipynb`](./01_MS_CalibSynergy_Net_End_to_End_Research_Pipeline.ipynb)

---

## 📌 Overview & Problem Statement
Deep learning classifiers trained on plant disease datasets frequently suffer from three practical issues when moving from research notebooks to real-world deployment:
1. **Specimen Data Leakage:** Standard random train/test splits often place multiple augmented or sequential photos of the exact same physical leaf into both training and test sets, artificially inflating reported accuracy.
2. **Ensemble Overconfidence (Calibration Paradox):** Combining multiple CNN backbones via soft-voting or stacking improves top-1 accuracy but often degrades probability calibration—producing overly confident predictions even when the model is wrong.
3. **Lack of Decision Transparency & Triage:** Farmers and agronomists need visual proof that the model is focusing on actual necrotic lesions rather than background soil/lighting artifacts, along with an automated way to flag uncertain predictions for human review.

**MS-CalibSynergy-Net** addresses these challenges by pairing a **zero-leakage group-isolated data pipeline** with a **multi-scale calibrated ensemble** (`DenseNet-121` + `EfficientNet-B0` + `ResNet-50` + `Attention Meta-Learner` + `Post-Hoc Temperature Scaling`), dual-paradigm explainability (**Grad-CAM, SHAP, and LIME**), and an end-to-end production deployment stack.

---

## 🗂️ Dataset Source & Zero-Leakage Preprocessing Pipeline
* **Dataset Source:** Derived from the **PlantVillage Foliar Pathology Benchmark** covering **15 distinct classes** across three major solanaceous crops:
  * **Bell Pepper (*Capsicum annuum*):** Bacterial Spot, Healthy
  * **Potato (*Solanum tuberosum*):** Early Blight, Late Blight, Healthy
  * **Tomato (*Solanum lycopersicum*):** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted Spider Mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy
* **Quality Audit & Outlier Filtering:** Audited **3,600 raw images**, verified 0 corrupted files, and removed **18 statistical $3\sigma$ brightness/contrast outliers**, yielding a clean dataset of **3,582 images** from **1,354 unique source specimens**.
* **Leakage-Free `StratifiedGroupKFold` Partitioning:** Extracted unique specimen identifiers from raw filenames and enforced group-isolated splitting across **Train (70.5%, 2,525 images)**, **Validation (14.8%, 531 images)**, and **Test (14.7%, 526 images)** with **0 overlapping source leaves (0.00% leakage)**.
* **Domain-Specific Preprocessing:** Computed training-set RGB channel statistics (`Mean: [0.457, 0.473, 0.414]`, `Std: [0.187, 0.165, 0.201]`), applied `CLAHE` local contrast enhancement, and added a bilateral edge-preserving denoising filter (`d=5, sigma=35`) to suppress sensor noise while keeping lesion borders sharp.

![Dataset Quality and Split Summary](research_artifacts/figures/preprocessing_executive_summary.png)

---

## 🏗️ System Architecture & Key Technical Contributions
1. **Complementary Multi-Backbone Feature Extraction:**
   * **`DenseNet-121` (7.49M params):** Captures fine-grained, localized necrotic spots and chlorotic halo textures via dense feature reuse.
   * **`EfficientNet-B0` (4.67M params):** Captures multi-scale leaf morphology and structural venation patterns via compound scaling.
   * **`ResNet-50` (24.57M params):** Provides deep residual semantic representations.
2. **Channel-Attention Gated Meta-Learner & Multi-View Synergy:**
   * Fuses log-probability representations across spatial test-time views using a residual channel-attention meta-network trained with out-of-fold validation predictions.
3. **Post-Hoc L-BFGS Temperature Scaling & Selective Triage:**
   * Optimizes a single calibration temperature parameter ($T$) on the validation set to align predicted probabilities with true empirical accuracy, cutting Expected Calibration Error (ECE) from `0.1064` (uncalibrated stacking) down to `0.0124`.
   * Computes predictive Shannon entropy $H(p)$ for every specimen, automatically flagging high-uncertainty cases ($H \ge \tau$) for manual agronomist inspection.

---

## 📊 Experimental Results & Model Ablation (Held-Out Test Set, $N = 526$)

| Rank | Model Architecture | Params (M) | Test Accuracy (%) | Macro Precision | Macro Recall | Macro F1 | MCC | ROC-AUC (OvR) | Calibration Error (ECE $\downarrow$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **MS-CalibSynergy-Net (Proposed)** | **36.74** | **96.39%** | **0.9658** | **0.9562** | **0.9597** | **0.9614** | **0.9989** | **0.0124** |
| 2 | Ensemble-2: Attention Stacking | 36.74 | 96.20% | 0.9639 | 0.9541 | 0.9569 | 0.9593 | 0.9985 | 0.1064 |
| 3 | Ensemble-1: Weighted Soft-Voting | 36.73 | 96.01% | 0.9561 | 0.9499 | 0.9505 | 0.9572 | 0.9987 | 0.0958 |
| 4 | DenseNet-121 Baseline | 7.49 | 95.63% | 0.9533 | 0.9489 | 0.9491 | 0.9531 | 0.9988 | 0.0773 |
| 5 | EfficientNet-B0 Baseline | 4.67 | 93.35% | 0.9250 | 0.9245 | 0.9221 | 0.9286 | 0.9968 | 0.0781 |
| 6 | ResNet-50 Baseline | 24.57 | 92.40% | 0.9226 | 0.9150 | 0.9175 | 0.9182 | 0.9942 | 0.0870 |

* **Bootstrap Statistical Confidence (1,000 Iterations):** Mean Test Accuracy of `96.38%` with a **95% Confidence Interval of `[94.68%, 97.91%]`**.

![Model Comparison and Calibration Tradeoff](research_artifacts/figures/definitive_6model_q1_summary_clean.png)

---

## 🔍 Explainable AI (XAI) & Robustness Verification

### 1. Reliability Calibration, Risk-Coverage Triage & Dual-Backbone Grad-CAM
Demonstrates probability alignment along the ideal diagonal, selective abstention accuracy gains (`>99.2%` accuracy at 95% coverage), and spatial agreement between `DenseNet-121` and `EfficientNet-B0` activation maps.
![Calibration and GradCAM](research_artifacts/figures/q1_novelty_calibration_and_gradcam.png)

### 2. Global & Local SHAP Feature Attribution
Quantifies how each backbone stream contributes to specific crop disease predictions across the meta-learner decision boundary.
![SHAP Attribution Summary](research_artifacts/figures/shap_global_and_local_summary_clean.png)

### 3. Multi-Crop LIME Superpixel Ridge Regression Audit
Isolates positive lesion superpixels (green) vs. healthy tissue regions (red) across Bell Pepper Bacterial Spot, Potato Late Blight, and Tomato Early Blight.
![LIME Multi-Crop Audit](research_artifacts/figures/lime_proposed_model_multicrop_clean.png)

### 4. Bootstrap Distribution, Field Disturbance Tolerance & FP16 Edge Profiling
Evaluates model resilience under illumination shifts (`75%` and `+25%` brightness), Gaussian sensor noise with bilateral filtering, low-resolution restoration, and FP16 half-precision weight compression (`69 MB`).
![Robustness and Hardware Profiling](research_artifacts/figures/q1_generalization_and_statistical_rigor_clean.png)

---

## 📁 Repository Structure
```text
├── 01_MS_CalibSynergy_Net_End_to_End_Research_Pipeline.ipynb  # Complete training, evaluation & XAI notebook
├── streamlit_app.py                                           # Production Streamlit web application + PDF report generator
├── app.py                                                     # Standalone FastAPI REST microservice (/api/v1/predict) + Gradio UI
├── Dockerfile                                                 # Container specification for cloud/edge microservice deployment
├── requirements.txt                                           # Python package dependencies
├── sample_1_Pepper_Bacterial_spot.JPG                         # Held-out test reference specimen (Bell Pepper)
├── sample_2_Potato_Late_blight.JPG                            # Held-out test reference specimen (Potato)
├── sample_3_Tomato_Early_blight.JPG                           # Held-out test reference specimen (Tomato)
└── research_artifacts/
    ├── figures/                                               # High-resolution 300-DPI evaluation & XAI figures (.png)
    └── tables/                                                # Benchmark leaderboards and metric exports (.csv)
