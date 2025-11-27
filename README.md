# ML-Project-Part-2

Machine Learning Project: Risk Prediction & Travel Behavior Insights

Team Name: Team Chicken Biryani

Team Members:

Unnath Chittimalla (IMT2023620)

Prakrititz Borah (IMT2023547)

Areen Patil (IMT2023013)

🟢 Section 1: Binary Classification (Risk Prediction)

🎯 Objective

To predict the RiskFlag (0 or 1) of user profiles based on high-dimensional categorical data and behavioral statistics. The focus was on optimizing for a dataset with non-linear feature interactions and establishing a robust pipeline that exceeds standard linear baselines.

🛠️ Data Pipeline & Feature Engineering

Quantile Transformation: Used a Gaussian Output distribution to map skewed numeric features into a Normal (Bell Curve) distribution. This was critical for the convergence of the Deep Neural Network and RBF Kernel SVM.

Row-Wise Statistics: Engineered features capturing user volatility (Mean, Std, Min, Max) across numeric attributes.

Hybrid Encoding:

Low Cardinality: One-Hot Encoding for categories with $\le$ 10 levels.

High Cardinality: Target Encoding for sparse categories to capture risk probability density.

🧠 Model Architectures

Deep Learning (MLP) - Best Performer

Architecture: Tapered Funnel (512 $\to$ 256 $\to$ 128).

Activation: GELU (Gaussian Error Linear Unit) for smooth probabilistic non-linearity.

Optimization: OneCycleLR scheduler to escape local minima.

Bagged SVM

Solved the $O(N^3)$ complexity of SVMs by training an ensemble of 10 independent estimators, each on a 15% subsample of the data.

Logistic Regression

Baseline linear model using the SAGA solver for fast convergence on high-dimensional data.

🏆 Results (Stratified 5-Fold CV)

Model

Accuracy

Key Technique

Deep Learning (MLP)

0.888

Quantile Transform + OneCycleLR

Logistic Regression

0.887

SAGA Solver

Bagged SVM

0.884

Bagging Ensemble (10 Est.)

🔵 Section 2: Multi-Class Classification (Travel Behavior)

🎯 Objective

To classify travelers into three spending tiers (High, Medium, Low) based on trip details and demographics. The challenge involved handling "Power Law" distributions in stay durations and extracting patterns from raw text data like Country and Activity.

🛠️ Data Pipeline & Feature Engineering

Log Normalization: Applied np.log1p to duration columns (total_stay_nights) to fix massive right-skewness and prevent gradient explosion.

Economic Tiering: Manually mapped countries to GDP tiers (e.g., 'USA' $\to$ 'High', 'Kenya' $\to$ 'Low') to reduce noise.

"Package Score": A composite score derived from summing binary flags (Food + Transport + Guide) to quantify luxury level.

"Free Loader" Flag: Interaction feature identifying users who visit friends and pay 0 for accommodation.

🧠 Model Architecture: Hybrid Stacking Ensemble

We implemented a Passthrough Stacking architecture that combines the strengths of linear and non-linear models.

Base Learners:

PyTorch MLP: Captures smooth, complex patterns (e.g., Duration vs. Cost).

Linear SVM: Finds hard decision boundaries in sparse data (e.g., Country/Activity).

Meta-Learner: Logistic Regression.

Innovation: Unlike standard stacking, we fed the Meta-Learner both the predictions of the base models AND the original raw features ("Passthrough").

🏆 Results (Stratified 3-Fold CV)

Metric

Score

Interpretation

Accuracy

0.7185

Correctly classifies ~72% of travelers.

F1 Score (Macro)

0.6703

Balances precision/recall equally across all 3 classes.

🔍 Explainability (SHAP)

SHAP analysis confirmed that log_total_stay_nights was the #1 predictor of spending, followed by tour_type (Package vs. Independent).

📂 Repository Structure

├── Risk_Prediction/
│   ├── train_updated.csv
│   ├── test_updated.csv
│   ├── risk_model_training.py  # GPU MLP & Bagged SVM Code
│   └── Risk_Report.pdf
├── Travel_Behavior/
│   ├── travel_data.csv
│   ├── travel_stacking.py      # Stacking Ensemble Code
│   └── Travel_Report.pdf
└── README.md
