# ML-Project-Part-2

# Advanced Classification Systems: Risk Prediction & Travel Behavior Insights

**Team:** Chicken Biryani  
**Members:** - Areen Patil (IMT2023013)  
- Prakrititz Borah (IMT2023547)  
- Unnath Chittimalla (IMT2023620)

---

## 📖 Overview
This repository contains two distinct machine learning pipelines developed to solve complex classification challenges: **Financial Risk Prediction** (Binary) and **Traveler Spending Classification** (Multiclass). 

Our approach prioritizes data-centric AI over model-centric tuning. We utilized advanced preprocessing (Gaussian Quantile Transformation), domain-specific feature engineering, and hybrid architectures (PyTorch MLPs stacked with SVMs).

---

## 🚀 Project 1: Risk Prediction System
**Goal:** Predict the `RiskFlag` of user profiles based on high-dimensional behavioral and financial data.

### 🧠 Key Methodologies
* **Gaussian Quantile Transformation:** We addressed skewed numerical distributions (e.g., `ApplicantYears`, `AnnualEarnings`) by forcing them into a Normal Distribution via Quantile Transformation. This stabilized gradients for the Neural Network.
* **Split-Stream Categorical Encoding:**
    * **Low Cardinality (<10):** One-Hot Encoding.
    * **High Cardinality:** Target Encoding to compress sparse data into dense "risk probability" features.
* **Bagged SVM Optimization:** To solve the $O(N^3)$ complexity of SVMs, we implemented a **Bagging Ensemble** where 10 independent SVMs trained on random 15% subsets of the data, achieving non-linear separation without the computational bottleneck.

### 🏗️ Model Architectures
1.  **Deep Learning (PyTorch MLP):** * Architecture: Funnel structure (512 $\to$ 256 $\to$ 128).
    * Activation: **GELU** (Gaussian Error Linear Unit) to prevent "dying ReLU".
    * Scheduler: **OneCycleLR** for super-convergence.
2.  **Logistic Regression:** Uses the **SAGA** solver for fast convergence on L1/L2 regularized problems.

### 📊 Performance
| Model | Accuracy | Key Configuration |
| :--- | :--- | :--- |
| **Deep Learning (MLP)** | **0.888** | OneCycleLR, GELU, Quantile Transform |
| Logistic Regression | 0.887 | SAGA Solver, Elastic Net |
| Bagged SVM | 0.884 | 10 Estimators, RBF Kernel |

---

## ✈️ Project 2: Travel Behavior Insights
**Goal:** Predict traveler `Spend_Category` (High/Medium/Low) using profile and trip details.

### 🧠 Feature Engineering "Secret Sauce"
Instead of raw columns, we engineered features based on domain logic:
* **Economic Tiering:** Mapped raw country names to GDP tiers (e.g., USA $\to$ High, Kenya $\to$ Low).
* **Package Inclusion Score:** A summation of binary flags (Food + Transport + Guide) to quantify luxury level.
* **The "Free Loader" Flag:** A specific interaction feature identifying users who visit friends, travel independently, and pay \$0 for accommodation.
* **Log Normalization:** Applied `np.log1p` to stay durations to handle power-law distributions.

### 🏗️ Hybrid Stacking Architecture
We employed a **Stacking Classifier** with a "Passthrough" strategy:
1.  **Level 0 (Base Models):** * *PyTorch MLP:* Captures complex non-linear patterns.
    * *Linear SVC:* Captures high-dimensional sparse boundaries.
2.  **Level 1 (Meta-Learner):** * *Logistic Regression:* Receives predictions from Level 0 **AND** the original raw features to make the final decision.

### 📊 Performance
* **Validation Accuracy:** 71.85%
* **Macro F1 Score:** 0.67
* **Explainability:** SHAP analysis confirmed `log_total_stay_nights` and `Package_Inclusion_Score` as the top predictors.

---

## 💻 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn torch category_encoders matplotlib seaborn tqdm
