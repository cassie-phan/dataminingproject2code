import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Read the training data
train_data = pd.read_csv('binary_train.csv')

# Display data info
print("Binary Classification Dataset Info:")
print(train_data.head())
print("\n", train_data.info())

# Separate features and target
# Assuming the last column is the target (adjust if needed)
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

print(f"\nTarget variable: {train_data.columns[-1]}")
print(f"Class distribution:\n{y_train.value_counts()}")

# Hyperparameters to test
dt_max_depths = [3, 5, 7, 10, 15, 20]
rf_n_estimators = [10, 50, 100, 200, 300]

# Number of folds for cross-validation
k_folds = 5

print("\n" + "="*70)
print("Starting Decision Tree cross-validation...")
print("="*70)
# Decision Tree Cross-Validation
dt_mean_scores = []
dt_std_scores = []

for depth in dt_max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(dt, X_train, y_train, cv=k_folds, scoring='accuracy')
    dt_mean_scores.append(scores.mean())
    dt_std_scores.append(scores.std())
    print(f"max_depth={depth}: Mean Accuracy={scores.mean():.4f}, Std={scores.std():.4f}")

print("\n" + "="*70)
print("Starting Random Forest cross-validation...")
print("="*70)
# Random Forest Cross-Validation
rf_mean_scores = []
rf_std_scores = []

for n_est in rf_n_estimators:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=k_folds, scoring='accuracy')
    rf_mean_scores.append(scores.mean())
    rf_std_scores.append(scores.std())
    print(f"n_estimators={n_est}: Mean Accuracy={scores.mean():.4f}, Std={scores.std():.4f}")

# Create visualizations - Decision Tree
plt.figure(figsize=(10, 6))
plt.plot(dt_max_depths, dt_mean_scores, marker='o', linestyle='-', 
         linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('max_depth', fontsize=12)
plt.ylabel('CV accuracy', fontsize=12)
plt.title('Decision Tree cross-validation (binary)', fontsize=14)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('binary_dt_cv_results.png', dpi=300, bbox_inches='tight')
print("\nDecision Tree visualization saved as 'binary_dt_cv_results.png'")
plt.close()

# Create visualizations - Random Forest
plt.figure(figsize=(10, 6))
plt.plot(rf_n_estimators, rf_mean_scores, marker='o', linestyle='-', 
         linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('n_estimators', fontsize=12)
plt.ylabel('CV accuracy', fontsize=12)
plt.title('Random Forest cross-validation (binary)', fontsize=14)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('binary_rf_cv_results.png', dpi=300, bbox_inches='tight')
print("Random Forest visualization saved as 'binary_rf_cv_results.png'")
plt.close()

# Print summary
best_dt_idx = np.argmax(dt_mean_scores)
best_rf_idx = np.argmax(rf_mean_scores)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nDecision Tree:")
print(f"  Best max_depth: {dt_max_depths[best_dt_idx]}")
print(f"  Best CV Accuracy: {dt_mean_scores[best_dt_idx]:.4f} ± {dt_std_scores[best_dt_idx]:.4f}")

print(f"\nRandom Forest:")
print(f"  Best n_estimators: {rf_n_estimators[best_rf_idx]}")
print(f"  Best CV Accuracy: {rf_mean_scores[best_rf_idx]:.4f} ± {rf_std_scores[best_rf_idx]:.4f}")