import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- 1. Configuration & Data Loading (Same as before) ---
TRAIN_FILE = "travel-behavior-insights/train.csv"
TEST_FILE = "travel-behavior-insights/test.csv"
RANDOM_SEED = 42
N_SPLITS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Error: train.csv or test.csv not found. Please check file paths.")
    exit()

X = train_df.drop("spend_category", axis=1)
y = train_df["spend_category"]
valid_indices = y.dropna().index
X = X.loc[valid_indices]
y = y.loc[valid_indices].astype(int)
NUM_CLASSES = y.nunique()

X_test = test_df.copy()
combined_df = pd.concat([X, X_test], ignore_index=True)

# --- 2. Custom Feature Engineering Transformer (Same as before) ---
# NOTE: The CustomFeatureEngineer class definition should be copied here exactly
# from the previous code block to ensure the data preparation is correct.
# (Skipping the definition here for brevity in the response, assume it's included)
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gdp_map = {
            'USA': 'High', 'CANADA': 'High', 'UK': 'High', 'GERMANY': 'High', 
            'FRANCE': 'High', 'JAPAN': 'High', 'SWITZERLAND': 'High', 
            'AUSTRALIA': 'High', 'SWEDEN': 'High', 'SPAIN': 'High', 
            'ITALY': 'High', 'MEXICO': 'Upper_Mid', 'INDIA': 'Mid', 
            'CHINA': 'Mid', 'BRAZIL': 'Mid', 'CONGO': 'Low', 
            'KENYA': 'Low', 'ZAMBIA': 'Low'
        }
        self.activity_cost_map = {
            'Hunting Tourism': 'Luxury', 'Mountain Climbing': 'Luxury', 
            'Diving and Sport Fishing': 'Luxury', 'Cultural Tourism': 'Premium', 
            'Wildlife Tourism': 'Premium', 'Widlife Tourism': 'Premium',
            'Conference Tourism': 'Premium', 'Business': 'Premium',
            'Beach Tourism': 'Standard', 'Bird Tourism': 'Standard',
        }
        self.pkg_cols = ['intl_transport_included', 'accomodation_included', 'food_included', 
                         'domestic_transport_included', 'sightseeing_included', 
                         'guide_included', 'insurance_included']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_c = X.copy()
        
        num_cols = ['mainland_stay_nights', 'island_stay_nights', 'num_females', 'num_males']
        for c in num_cols:
            X_c[c] = pd.to_numeric(X_c[c], errors='coerce').fillna(0)

        X_c['total_stay_nights'] = X_c['mainland_stay_nights'] + X_c['island_stay_nights']
        X_c['total_travelers'] = X_c['num_females'] + X_c['num_males']
        
        pkg_bin = X_c[self.pkg_cols].replace({'Yes': 1, 'No': 0}).fillna(0)
        X_c['Package_Inclusion_Score'] = pkg_bin.sum(axis=1)

        X_c['Gender_Ratio'] = np.where(X_c['total_travelers'] > 0, 
                                       X_c['num_females'] / X_c['total_travelers'], 0)
        X_c['inclusions_per_day'] = np.where(X_c['total_stay_nights'] > 0,
                                             X_c['Package_Inclusion_Score'] / X_c['total_stay_nights'], 0)

        for c in ['mainland_stay_nights', 'island_stay_nights', 'total_stay_nights']:
            X_c[f'log_{c}'] = np.log1p(X_c[c])

        X_c['country_economic_tier'] = X_c['country'].map(self.gdp_map).fillna('Other')
        X_c['activity_cost_tier'] = X_c['main_activity'].map(self.activity_cost_map).fillna('Standard')

        X_c['is_free_loader'] = ((X_c['visit_purpose']=='Visiting Friends and Relatives') & 
                                 (X_c['tour_type']=='Independent') & 
                                 (X_c['accomodation_included']=='No')).astype(int)
        
        high_spend = ['Business', 'Meetings and Conference', 'Scientific and Academic']
        X_c['is_high_spend_purpose'] = X_c['visit_purpose'].isin(high_spend).astype(int)
        
        return X_c

combined_df = CustomFeatureEngineer().fit_transform(combined_df)

# Define features for Preprocessing
numerical_features = ['log_mainland_stay_nights', 'log_island_stay_nights', 
                      'log_total_stay_nights', 'num_females', 'num_males', 
                      'total_travelers', 'Package_Inclusion_Score', 
                      'Gender_Ratio', 'inclusions_per_day']
binary_features = ['is_free_loader', 'is_high_spend_purpose']
categorical_features = [c for c in combined_df.columns 
                        if c not in numerical_features + binary_features + ['trip_id', 'spend_category']]

# --- 3. Preprocessing Pipeline Setup (Same as before) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Missing')), 
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
        ('bin', 'passthrough', binary_features)
    ],
    remainder='drop'
)

X_train = combined_df.iloc[:len(X)].drop(columns=['trip_id'])
X_test_final = combined_df.iloc[len(X):].drop(columns=['trip_id'])

# Apply Preprocessor to determine input size
# We need to transform the training data once to get the final feature count
X_train_processed = preprocessor.fit_transform(X_train)
INPUT_SIZE = X_train_processed.shape[1]

# --- 4. PyTorch MLP Model Definition (GPU Accelerated) ---

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.layer_1 = nn.Linear(input_size, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.output(x)
        return x

# --- 5. PyTorch Wrapper for Scikit-learn Compatibility ---

class PyTorchMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES, 
                 n_epochs=20, lr=0.005, batch_size=256, device=DEVICE, random_state=RANDOM_SEED):
        self.input_size = input_size
        self.num_classes = num_classes
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        # 1. ADD THIS LINE: Define the classes_ attribute
        self.classes_ = np.unique(y) 
        
        torch.manual_seed(self.random_state)
        self.model = SimpleMLP(self.input_size, self.num_classes).to(self.device)
        
        # Data preparation (X is a sparse matrix from preprocessor)
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.from_numpy(X.values).float().to(self.device)
        else:
            # Handle sparse matrix case correctly by converting to dense array
            X_tensor = torch.from_numpy(X.toarray()).float().to(self.device)
        
        # Ensure y is a NumPy array for unique() and then a tensor
        y_tensor = torch.from_numpy(y).long().to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        for epoch in tqdm(range(self.n_epochs), desc="PyTorch Training"):
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X_tensor = torch.from_numpy(X.values).float().to(self.device)
            else:
                X_tensor = torch.from_numpy(X.toarray()).float().to(self.device)
            
            outputs = self.model(X_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        return probabilities

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# --- 6. Stacking Classifier Definition ---

# Calculate class weights for SVM (LinearSVC)
class_counts = y.value_counts()
class_weights_dict = {i: len(y) / (len(class_counts) * count) for i, count in class_counts.items()}

estimators = [
    # 1. PyTorch Neural Network (GPU Accelerated)
    ('nn', PyTorchMLPWrapper(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, n_epochs=50)),

    # 2. Linear SVC (Fastest SVM variant, wrapped for probability)
    ('linear_svc', CalibratedClassifierCV(
        estimator=OneVsRestClassifier(LinearSVC(
            random_state=RANDOM_SEED, 
            class_weight=class_weights_dict, 
            dual='auto', 
            max_iter=5000 
        )),
        cv=3, 
        method='isotonic', # Better calibration for multiclass
        n_jobs=-1 # Parallelize internal CV for faster calibration
    ))
]

# Meta-Model (Level 1)
meta_model = LogisticRegression(
    random_state=RANDOM_SEED, 
    solver='lbfgs', 
    multi_class='multinomial', 
    class_weight=class_weights_dict, 
    n_jobs=-1
)

# Define the Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
    n_jobs=1 # Use n_jobs=1 to avoid PyTorch GPU conflicts with parallel training processes
)

# Full Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_classifier)
])

# --- 7. Execution ---

print(f"--- Starting PyTorch/LinearSVC Hybrid Training on {DEVICE} ---")
print(f"Input Feature Size for PyTorch/LinearSVC: {INPUT_SIZE}")

# Fit the full pipeline
# The PyTorch wrapper will automatically handle moving data and training on the GPU (if available).
pipeline.fit(X_train, y)

print("--- Training Complete. Making Predictions ---")

# Make predictions on the test set
preds = pipeline.predict(X_test_final)

# Create submission
sub = pd.DataFrame({'trip_id': test_df['trip_id'], 'spend_category': preds})
sub.to_csv("submission_pytorch_svm_stack.csv", index=False)
print("\nSubmission file 'submission_pytorch_svm_stack.csv' created.")