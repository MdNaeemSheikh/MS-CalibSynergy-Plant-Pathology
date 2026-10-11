<div align="center">

<a href="https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=800&size=34&duration=4000&pause=1500&color=2ECC71&center=true&vCenter=true&width=850&height=60&lines=%F0%9F%8C%BF+MS-CalibSynergy-Net+%7C+Plant+Pathology+AI;%F0%9F%8D%83+Calibrated+Multi-Scale+Diagnostic+System" alt="MS-CalibSynergy-Net Title" />
</a>

<a href="https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=17&duration=3200&pause=1000&color=58A6FF&center=true&vCenter=true&width=780&height=35&lines=15+Foliar+Crop+Pathologies+%7C+Bell+Pepper%2C+Potato+%26+Tomato;Dual-Backbone+Grad-CAM%2C+SHAP+%26+LIME+Visual+Explainability;96.39%25+Test+Accuracy+%7C+0.0124+Expected+Calibration+Error;Live+Streamlit+Web+Portal+%2B+Clinical+PDF+Report+Generator" alt="Subtitle Animation" />
</a>

<br/><br/>

[![Live Web App](https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Web%20Portal-Streamlit%20Cloud-FF4B4B?style=for-the-badge)](https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/)
[![Hugging Face Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Model%20Registry-Hugging%20Face%20Hub-FFD21E?style=for-the-badge)](https://huggingface.co/NaeemSheikh123/ms-calib-synergy-plant-disease)
[![Research Notebook](https://img.shields.io/badge/%F0%9F%93%93%20Jupyter%20Notebook-Research%20Pipeline-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](./End_to_End_Research_Pipeline.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Container%20Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](./Dockerfile)

</div>

<br/>

---

## 🌐 Live Diagnostic Web Portal

Test the deployed system in real time with custom leaf images or reference specimens. The interface provides calibrated class probabilities, multi-backbone Grad-CAM lesion localization, uncertainty triage alerts, and an automated **Clinical Diagnostic Report (PDF)** generator.

👉 **Launch Web Application:** [**https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/**](https://ms-calibsynergy-plant-pathology-ezhohi4fcx94mzkkbfyka3.streamlit.app/)

* **Cloud Model Registry:** [`NaeemSheikh123/ms-calib-synergy-plant-disease`](https://huggingface.co/NaeemSheikh123/ms-calib-synergy-plant-disease)
* **End-to-End Research Notebook:** [`End_to_End_Research_Pipeline.ipynb`](./End_to_End_Research_Pipeline.ipynb)

<br/>

---

## 📌 Project Overview & Motivation

Moving plant disease classifiers from experimental notebooks into practical field tools requires solving three common challenges:

1. **Specimen Data Leakage:** Standard random train/test splits often place multiple photos of the same physical leaf into both training and evaluation sets, inflating test accuracy.
2. **Ensemble Overconfidence:** Combining multiple CNN architectures via soft-voting or stacking improves accuracy but frequently degrades probability calibration—producing overly confident predictions even when the model is wrong.
3. **Decision Transparency & Triage:** Agronomists need visual proof that a model is attending to actual disease lesions rather than background noise, alongside an automated mechanism to flag uncertain cases for manual inspection.

**MS-CalibSynergy-Net** addresses these issues through a **group-isolated data pipeline (0% specimen overlap)**, a **multi-scale calibrated ensemble** (`DenseNet-121` + `EfficientNet-B0` + `ResNet-50` + `Attention Meta-Learner` + `Temperature Scaling`), multi-paradigm explainability (**Grad-CAM, SHAP, and LIME**), and an interactive web portal.

<br/>

---

## 🗂️ Dataset Source & Preprocessing Pipeline

* **Dataset Source:** Built from the **PlantVillage Foliar Pathology Benchmark**, covering **15 pathology classes** across three solanaceous crops:
  * **Bell Pepper (*Capsicum annuum*):** Bacterial Spot, Healthy
  * **Potato (*Solanum tuberosum*):** Early Blight, Late Blight, Healthy
  * **Tomato (*Solanum lycopersicum*):** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy

* **Statistical Quality Audit:** Audited **3,600 raw images**, verified zero corrupted files, and filtered **18 statistical $3\sigma$ brightness/contrast outliers**, yielding **3,582 clean images** across **1,354 unique source specimens**.

* **Leakage-Free `StratifiedGroupKFold` Partitioning:** Extracted unique specimen identifiers from filenames to enforce strict group isolation across **Train (70.5%, 2,525 images)**, **Validation (14.8%, 531 images)**, and **Test (14.7%, 526 images)** splits with **0 shared source leaves**.

* **Image Enhancement & Denoising:** Normalized inputs using training-set RGB channel statistics (`Mean: [0.457, 0.473, 0.414]`, `Std: [0.187, 0.165, 0.201]`), applied `CLAHE` local contrast enhancement, and integrated an edge-preserving bilateral filter (`d=5, sigma=35`) to suppress sensor noise without blurring lesion margins.

<br/>

![Dataset Quality and Split Summary](research_artifacts/figures/preprocessing_executive_summary.png)

<br/>

---

## 🏗️ System Architecture

1. **Complementary Backbone Streams:**
   * **`DenseNet-121`:** Captures fine-grained necrotic spots and chlorotic textures through dense feature reuse.
   * **`EfficientNet-B0`:** Captures multi-scale leaf morphology and structural venation patterns.
   * **`ResNet-50`:** Extracts deep residual semantic representations.

2. **Channel-Attention Gated Meta-Learner:**
   * Fuses log-probability representations across spatial views using a residual channel-attention meta-network trained on out-of-fold validation predictions.

3. **Post-Hoc Temperature Scaling & Selective Triage:**
   * Optimizes a calibration temperature parameter ($T$) via L-BFGS on the validation split, reducing Expected Calibration Error (ECE) by 88% (`0.1064` $\rightarrow$ `0.0124`).
   * Computes Shannon entropy $H(p)$ for each prediction and triggers a triage alert whenever uncertainty exceeds the clinical threshold ($H \ge \tau$).

<br/>

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

<br/>

* **Bootstrap Confidence Interval (1,000 Iterations):** Mean Test Accuracy of `96.38%` (`95% CI: [94.68%, 97.91%]`).

<br/>

![Model Comparison and Calibration Tradeoff](research_artifacts/figures/definitive_6model_q1_summary_clean.png)

<br/>

---

## 🔍 Explainable AI (XAI) & Robustness Audit

### 1. Reliability Calibration, Risk-Coverage Triage & Dual-Backbone Grad-CAM
![Calibration and GradCAM](research_artifacts/figures/q1_novelty_calibration_and_gradcam.png)

<br/>

### 2. Global & Local SHAP Feature Attribution
![SHAP Attribution Summary](research_artifacts/figures/shap_global_and_local_summary_clean.png)

<br/>

### 3. Multi-Crop LIME Superpixel Attribution (Bell Pepper, Potato & Tomato)
![LIME Multi-Crop Audit](research_artifacts/figures/lime_proposed_model_multicrop_clean.png)

<br/>

### 4. Bootstrap Confidence Distribution, Disturbance Tolerance & Latency Profiling
![Robustness and Hardware Profiling](research_artifacts/figures/q1_generalization_and_statistical_rigor_clean.png)

<br/>

---

## 📁 Repository Structure

```text
├── End_to_End_Research_Pipeline.ipynb      # Complete data audit, training, calibration & XAI notebook
├── streamlit_app.py                        # Production Streamlit web portal + PDF report generator
├── Dockerfile                              # Container specification for cloud deployment
├── requirements.txt                        # Python package dependencies
├── sample_1_Pepper_Bacterial_spot.JPG      # Held-out test reference specimen (Bell Pepper)
├── sample_2_Potato_Late_blight.JPG         # Held-out test reference specimen (Potato)
├── sample_3_Tomato_Early_blight.JPG        # Held-out test reference specimen (Tomato)
├── LICENSE                                 # MIT License
└── research_artifacts/
    ├── figures/                            # High-resolution evaluation & explainability plots (.png)
    └── tables/                             # Benchmark leaderboards and metric tables (.csv)
