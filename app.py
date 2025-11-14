import streamlit as st

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle
from datetime import datetime
import os

print("="*80)
print("TRAINING SEVERITY CLASSIFIER FOR PARKINSON'S DISEASE")
print("="*80)

# ============================================================================
# STEP 1: LOAD YOUR PREPROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

try:
    # Load your preprocessed training data
    X_train = pd.read_csv('X_train_preprocessed.csv')
    X_test = pd.read_csv('X_test_preprocessed.csv')

    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ X_test shape: {X_test.shape}")

except FileNotFoundError:
    print("❌ ERROR: Preprocessed data not found!")
    print("Expected files:")
    print("  - X_train_preprocessed.csv")
    print("  - X_test_preprocessed.csv")
    print("\nPlease run your data preprocessing notebook first!")
    raise

# ============================================================================
# STEP 2: LOAD UPDRS SCORES (For creating severity labels)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: LOADING UPDRS SCORES")
print("="*80)

try:
    # Try different possible filenames
    try:
        y_train_updrs = pd.read_csv('y_train_updrs.csv')
    except FileNotFoundError:
        y_train_updrs = pd.read_csv('y_train_severity.csv')

    try:
        y_test_updrs = pd.read_csv('y_test_updrs.csv')
    except FileNotFoundError:
        y_test_updrs = pd.read_csv('y_test_severity.csv')

    # Extract UPDRS values
    if 'UPDRS' in y_train_updrs.columns:
        y_train_updrs_values = y_train_updrs['UPDRS'].values
    else:
        # Try first column if named differently
        y_train_updrs_values = y_train_updrs.iloc[:, 0].values

    if 'UPDRS' in y_test_updrs.columns:
        y_test_updrs_values = y_test_updrs['UPDRS'].values
    else:
        y_test_updrs_values = y_test_updrs.iloc[:, 0].values

    print(f"✓ Training UPDRS scores: {len(y_train_updrs_values)} samples")
    print(f"✓ Test UPDRS scores: {len(y_test_updrs_values)} samples")
    print(f"✓ UPDRS range: {y_train_updrs_values.min():.1f} - {y_train_updrs_values.max():.1f}")

except FileNotFoundError as e:
    print(f"❌ ERROR: UPDRS data not found!")
    print("Expected files:")
    print("  - y_train_updrs.csv or y_train_severity.csv")
    print("  - y_test_updrs.csv or y_test_severity.csv")
    print("\nThese files should contain UPDRS scores (0-199)")
    raise

# ============================================================================
# STEP 3: CREATE SEVERITY LABELS FROM UPDRS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CREATING SEVERITY LABELS FROM UPDRS")
print("="*80)

def updrs_to_severity(updrs_scores):
    """
    Convert UPDRS scores to severity categories

    UPDRS Range → Severity Class:
    0-32    → 0 (Minimal)
    33-58   → 1 (Mild)
    59-95   → 2 (Moderate)
    96-199  → 3 (Severe)
    """
    severity = []
    for updrs in updrs_scores:
        if updrs <= 32:
            severity.append(0)  # Minimal
        elif updrs <= 58:
            severity.append(1)  # Mild
        elif updrs <= 95:
            severity.append(2)  # Moderate
        else:
            severity.append(3)  # Severe
    return np.array(severity)

# Create severity labels
y_train_severity = updrs_to_severity(y_train_updrs_values)
y_test_severity = updrs_to_severity(y_test_updrs_values)

print("\n✓ Severity labels created")

# Show distribution
print("\nTRAINING DATA - Severity Distribution:")
severity_names = ['Minimal (0-32)', 'Mild (33-58)', 'Moderate (59-95)', 'Severe (96-199)']
for i, name in enumerate(severity_names):
    count = sum(y_train_severity == i)
    pct = count/len(y_train_severity)*100
    print(f"  {i}. {name:20s}: {count:5d} samples ({pct:6.2f}%)")

print("\nTEST DATA - Severity Distribution:")
for i, name in enumerate(severity_names):
    count = sum(y_test_severity == i)
    pct = count/len(y_test_severity)*100
    print(f"  {i}. {name:20s}: {count:5d} samples ({pct:6.2f}%)")

# ============================================================================
# STEP 4: TRAIN SEVERITY CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TRAINING GRADIENT BOOSTING CLASSIFIER")
print("="*80)

print("\nInitializing Gradient Boosting Classifier...")
severity_model = GradientBoostingClassifier(
    n_estimators=100,           # Number of boosting stages
    learning_rate=0.1,          # Learning rate (0.01-0.2)
    max_depth=5,                # Maximum depth of trees
    min_samples_split=5,        # Minimum samples to split
    min_samples_leaf=2,         # Minimum samples per leaf
    random_state=42,            # For reproducibility
    verbose=1                   # Show progress
)

print("Training on preprocessed features...")
print(f"Features: {X_train.shape[1]}")
print(f"Samples: {X_train.shape[0]}")

severity_model.fit(X_train, y_train_severity)

print("\n✓ Training completed!")

# ============================================================================
# STEP 5: EVALUATE SEVERITY CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("STEP 5: EVALUATING MODEL PERFORMANCE")
print("="*80)

# Predictions on training set
y_pred_train = severity_model.predict(X_train)
train_accuracy = accuracy_score(y_train_severity, y_pred_train)

# Predictions on test set
y_pred_test = severity_model.predict(X_test)
test_accuracy = accuracy_score(y_test_severity, y_pred_test)

print(f"\n📊 ACCURACY METRICS:")
print(f"  Training Accuracy: {train_accuracy*100:.2f}%")
print(f"  Test Accuracy:     {test_accuracy*100:.2f}%")

if abs(train_accuracy - test_accuracy) > 0.10:
    print(f"  ⚠️  Warning: Possible overfitting detected (diff > 10%)")
else:
    print(f"  ✓ Model well-balanced (diff < 10%)")

# Classification report
print(f"\n📋 CLASSIFICATION REPORT (Test Set):")
print(classification_report(
    y_test_severity,
    y_pred_test,
    target_names=severity_names
))

# Confusion matrix
print(f"\n🔢 CONFUSION MATRIX (Test Set):")
cm = confusion_matrix(y_test_severity, y_pred_test)
print("\nPredicted →")
print("Actual ↓        Minimal    Mild  Moderate   Severe")
for i, row in enumerate(cm):
    print(f"{severity_names[i]:14s} {row[0]:8d} {row[1]:7d} {row[2]:9d} {row[3]:7d}")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': severity_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 TOP 20 MOST IMPORTANT FEATURES FOR SEVERITY PREDICTION:")
print("\nRank | Feature Name                       | Importance Score")
print("-" * 65)
for idx, (feat, imp) in enumerate(feature_importance.head(20).values, 1):
    print(f"{idx:2d}   | {feat:33s} | {imp:.6f}")

# ============================================================================
# STEP 7: SAVE SEVERITY MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 7: SAVING MODEL AND METADATA")
print("="*80)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model
model_path = 'models/severity_classifier.pkl'
joblib.dump(severity_model, model_path)
print(f"✓ Model saved: {model_path}")

# Save metadata
metadata = {
    'model_type': 'GradientBoostingClassifier',
    'severity_classes': ['Minimal', 'Mild', 'Moderate', 'Severe'],
    'severity_updrs_ranges': {
        'Minimal': '0-32',
        'Mild': '33-58',
        'Moderate': '59-95',
        'Severe': '96-199'
    },
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'n_features': X_train.shape[1],
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test),
    'feature_names': list(X_train.columns),
    'top_10_features': list(feature_importance.head(10)['feature'].values),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'hyperparameters': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
}

metadata_path = 'models/severity_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Metadata saved: {metadata_path}")

# ============================================================================
# STEP 8: SUMMARY AND VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
✅ SEVERITY CLASSIFIER TRAINING COMPLETE!

📊 MODEL PERFORMANCE:
   - Training Accuracy: {train_accuracy*100:.2f}%
   - Test Accuracy: {test_accuracy*100:.2f}%
   - Model Type: Gradient Boosting Classifier

📦 FILES CREATED:
   ✓ models/severity_classifier.pkl
   ✓ models/severity_metadata.pkl

📋 MODEL INFO:
   - Input Features: {X_train.shape[1]}
   - Training Samples: {len(X_train):,}
   - Test Samples: {len(X_test):,}
   - Severity Classes: 4
     * 0 = Minimal (UPDRS 0-32)
     * 1 = Mild (UPDRS 33-58)
     * 2 = Moderate (UPDRS 59-95)
     * 3 = Severe (UPDRS 96-199)

🎯 TOP 3 IMPORTANT FEATURES:
   1. {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.6f})
   2. {feature_importance.iloc[1]['feature']} ({feature_importance.iloc[1]['importance']:.6f})
   3. {feature_importance.iloc[2]['feature']} ({feature_importance.iloc[2]['importance']:.6f})

🚀 NEXT STEPS:
   1. Verify files exist in 'models/' directory
   2. Update your Streamlit app with severity model loading
   3. Run: streamlit run app.py
   4. Test with sample patient data
""")

# ============================================================================
# STEP 9: VERIFY FILES
# ============================================================================

print("="*80)
print("STEP 9: VERIFICATION")
print("="*80)

import os

files_to_check = [
    'models/severity_classifier.pkl',
    'models/severity_metadata.pkl'
]

print("\n✓ Checking saved files:")
for file_path in files_to_check:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024*1024)  # Convert to MB
        print(f"  ✓ {file_path:40s} ({file_size:.2f} MB)")
    else:
        print(f"  ❌ {file_path:40s} NOT FOUND")

print("\n" + "="*80)
print("✅ SEVERITY CLASSIFIER TRAINING COMPLETE!")
print("="*80)

# ============================================================================
# STEP 10: TEST THE MODEL (Optional)
# ============================================================================

print("\n" + "="*80)
print("STEP 10: TESTING MODEL (Sample Predictions)")
print("="*80)

print("\nTesting on first 5 test samples:")
print("-" * 80)

for i in range(min(5, len(X_test))):
    sample = X_test.iloc[i:i+1]
    pred = severity_model.predict(sample)[0]
    proba = severity_model.predict_proba(sample)[0]
    actual = y_test_severity[i]

    pred_name = severity_names[pred].split('(')[0].strip()
    actual_name = severity_names[actual].split('(')[0].strip()

    print(f"\nSample {i+1}:")
    print(f"  Actual:    {actual_name}")
    print(f"  Predicted: {pred_name} (Confidence: {proba[pred]*100:.1f}%)")
    print(f"  Probabilities: Minimal={proba[0]*100:.1f}%, Mild={proba[1]*100:.1f}%, " +
          f"Moderate={proba[2]*100:.1f}%, Severe={proba[3]*100:.1f}%")
    print(f"  Match: {'✓' if pred == actual else '❌'}")

print("\n" + "="*80)
print("Training script completed successfully!")
print("="*80)

