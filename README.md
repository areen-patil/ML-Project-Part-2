      # Advanced Classification Systems: Risk Prediction & Travel Behavior Insights

**Team:** Chicken Biryani  
**Members:** 
- Areen Patil (IMT2023013)  
- Prakrititz Borah (IMT2023547)  
- Unnath Chittimalla (IMT2023620)

---

## 📖 Overview
This repository contains two distinct machine learning pipelines developed to solve complex classification challenges: **Financial Risk Prediction** (Binary) and **Traveler Spending Classification** (Multiclass). 

Our approach prioritizes data-centric AI over model-centric tuning. We utilized advanced preprocessing (Gaussian Quantile Transformation), domain-specific feature engineering, and hybrid architectures (PyTorch MLPs stacked with SVMs).

---

## 🚀 Project 1: Risk Prediction System
[cite_start]**Goal:** Predict the `RiskFlag` of user profiles based on high-dimensional behavioral and financial data[cite: 307].

### 📊 Exploratory Analysis & Method
We faced a significant class imbalance and skewed numerical features.
![Risk Target Distribution](assets/dist.png)
[cite_start]*Figure 1: Target Variable Distribution showing the imbalance between "No Risk" (0) and "Risk" (1)[cite: 324, 327].*

**The Gaussian Fix:**
Standard scaling failed because features like `ApplicantYears` followed a Power Law. [cite_start]We used **Quantile Transformation** to force these features into a Gaussian (Bell Curve) distribution, stabilizing the Neural Network gradients[cite: 364, 367].

![Quantile Transformation](assets/quantile.png)
[cite_start]*Figure 2: Transformation of 'ApplicantYears' from raw skewed data (Left) to Gaussian distribution (Right)[cite: 368, 389].*

### 🧠 Key Architectures
* [cite_start]**Deep Learning (PyTorch MLP):** Uses a funnel architecture (512 $\to$ 256 $\to$ 128) with **GELU** activation and **OneCycleLR** scheduler[cite: 416, 420].
* [cite_start]**Bagged SVM:** To solve the $O(N^3)$ complexity of SVMs, we implemented a **Bagging Ensemble** where 10 independent SVMs trained on random 15% subsets of the data[cite: 435, 436].

### 📈 Performance
| Model | Accuracy | Key Configuration |
| :--- | :--- | :--- |
| **Deep Learning (MLP)** | **0.888** | [cite_start]OneCycleLR, GELU, Quantile Transform [cite: 442] |
| Logistic Regression | 0.887 | [cite_start]SAGA Solver, Elastic Net [cite: 442] |
| Bagged SVM | 0.884 | [cite_start]10 Estimators, RBF Kernel [cite: 442] |

---

## ✈️ Project 2: Travel Behavior Insights
[cite_start]**Goal:** Predict traveler `Spend_Category` (High/Medium/Low) using profile and trip details[cite: 7].

### 🧠 Feature Engineering "Secret Sauce"
[cite_start]We engineered features based on domain logic rather than raw columns[cite: 85]:
* [cite_start]**Economic Tiering:** Mapped raw country names to GDP tiers (e.g., USA $\to$ High, Kenya $\to$ Low)[cite: 88].
* [cite_start]**Package Inclusion Score:** A summation of binary flags (Food + Transport + Guide) to quantify luxury level[cite: 90].
* [cite_start]**Log Normalization:** Applied `np.log1p` to stay durations to handle power-law distributions[cite: 81].

![Feature Correlation](assets/corr-2.png)
[cite_start]*Figure 3: Correlation Matrix confirming 'Package_Inclusion_Score' as a strong predictor[cite: 245].*

### 🏗️ Hybrid Stacking Architecture
We employed a **Stacking Classifier** with a "Passthrough" strategy:
1.  [cite_start]**Level 0 (Base Models):** * *PyTorch MLP:* Captures complex non-linear patterns[cite: 250].
    * [cite_start]*Linear SVM:* Captures high-dimensional sparse boundaries[cite: 259].
2.  [cite_start]**Level 1 (Meta-Learner):** * *Logistic Regression:* Receives predictions from Level 0 **AND** the original raw features to make the final decision[cite: 263].

### 🔍 Explainability (SHAP)
We validated our model using SHAP values. [cite_start]The engineered feature `log_total_stay_nights` was identified as the #1 predictor of spending[cite: 270].

![SHAP Summary Plot](assets/shap.png)
[cite_start]*Figure 4: SHAP summary plot showing feature impact on model output[cite: 302].*

### 📊 Performance
* [cite_start]**Validation Accuracy:** 71.85% [cite: 267]
* [cite_start]**Macro F1 Score:** 0.67 [cite: 267]

---

## 💻 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn torch category_encoders matplotlib seaborn tqdm
