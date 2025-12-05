import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_data = pd.read_csv('multiclass_train.csv')
val_data = pd.read_csv('multiclass_validation.csv')

X_train = train_data.drop('Diabetes_012', axis=1)
y_train = train_data['Diabetes_012']
X_val = val_data.drop('Diabetes_012', axis=1)
y_val = val_data['Diabetes_012']

print("="*70)
print("ORIGINAL DATASET")
print("="*70)
print(f"Number of features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

print("\n" + "="*70)
print("APPLYING PCA DIMENSION REDUCTION")
print("="*70)

n_components = X_train.shape[1] // 2  # 50% of features
print(f"Reducing from {X_train.shape[1]} features to {n_components} features")

pca = PCA(n_components=n_components)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_val_pca = pca.transform(X_val)

print(f"\nVariance explained by {n_components} components: {pca.explained_variance_ratio_.sum():.4f}")
print(f"New training shape: {X_train_pca.shape}")
print(f"New validation shape: {X_val_pca.shape}")

train_pca_df = pd.DataFrame(X_train_pca)
train_pca_df['Diabetes_012'] = y_train.values
train_pca_df.to_csv('multiclass_train_pca.csv', index=False)

val_pca_df = pd.DataFrame(X_val_pca)
val_pca_df['Diabetes_012'] = y_val.values
val_pca_df.to_csv('multiclass_validation_pca.csv', index=False)

print("\nPCA-reduced datasets saved:")
print("  - multiclass_train_pca.csv")
print("  - multiclass_validation_pca.csv")

print("\n" + "="*70)
print("CROSS-VALIDATION WITH REDUCED FEATURES")
print("="*70)

dt_max_depths = [3, 5, 7, 10, 15, 20]
rf_n_estimators = [10, 50, 100, 200, 300]
k_folds = 5

print("\nDecision Tree CV results:")
dt_mean_scores = []
dt_std_scores = []

for depth in dt_max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(dt, X_train_pca, y_train, cv=k_folds, scoring='accuracy')
    dt_mean_scores.append(scores.mean())
    dt_std_scores.append(scores.std())
    print(f"max_depth={depth}: Mean Accuracy={scores.mean():.4f}, Std={scores.std():.4f}")

best_dt_depth = dt_max_depths[np.argmax(dt_mean_scores)]
print(f"\nBest Decision Tree depth: {best_dt_depth}")

print("\nRandom Forest CV results:")
rf_mean_scores = []
rf_std_scores = []

for n_est in rf_n_estimators:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    scores = cross_val_score(rf, X_train_pca, y_train, cv=k_folds, scoring='accuracy')
    rf_mean_scores.append(scores.mean())
    rf_std_scores.append(scores.std())
    print(f"n_estimators={n_est}: Mean Accuracy={scores.mean():.4f}, Std={scores.std():.4f}")

best_rf_estimators = rf_n_estimators[np.argmax(rf_mean_scores)]
print(f"\nBest Random Forest n_estimators: {best_rf_estimators}")

plt.figure(figsize=(10, 6))
plt.plot(dt_max_depths, dt_mean_scores, marker='o', linestyle='-', 
         linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('max_depth', fontsize=12)
plt.ylabel('CV accuracy', fontsize=12)
plt.title('Decision Tree cross-validation (multiclass, PCA reduced)', fontsize=14)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('multiclass_dt_cv_pca_results.png', dpi=300, bbox_inches='tight')
print("\nDecision Tree CV plot saved as 'multiclass_dt_cv_pca_results.png'")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(rf_n_estimators, rf_mean_scores, marker='o', linestyle='-', 
         linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('n_estimators', fontsize=12)
plt.ylabel('CV accuracy', fontsize=12)
plt.title('Random Forest cross-validation (multiclass, PCA reduced)', fontsize=14)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('multiclass_rf_cv_pca_results.png', dpi=300, bbox_inches='tight')
print("Random Forest CV plot saved as 'multiclass_rf_cv_pca_results.png'")
plt.close()

# ============================================================================
# Train final models with best hyperparameters
# ============================================================================
print("\n" + "="*70)
print("TRAINING FINAL MODELS WITH PCA-REDUCED FEATURES")
print("="*70)

# Decision Tree
print(f"\nDecision Tree (max_depth={best_dt_depth}):")
start_time = time.time()
dt_model = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
dt_model.fit(X_train_pca, y_train)
dt_train_time = time.time() - start_time

start_pred_time = time.time()
y_val_pred_dt = dt_model.predict(X_val_pca)
dt_pred_time = time.time() - start_pred_time

y_train_pred_dt = dt_model.predict(X_train_pca)
dt_train_accuracy = accuracy_score(y_train, y_train_pred_dt)
dt_val_accuracy = accuracy_score(y_val, y_val_pred_dt)

print(f"Training time: {dt_train_time:.4f} seconds")
print(f"Prediction time: {dt_pred_time:.4f} seconds")
print(f"Training accuracy: {dt_train_accuracy:.4f}")
print(f"Validation accuracy: {dt_val_accuracy:.4f}")

# Random Forest
print(f"\nRandom Forest (n_estimators={best_rf_estimators}):")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=best_rf_estimators, random_state=42)
rf_model.fit(X_train_pca, y_train)
rf_train_time = time.time() - start_time

start_pred_time = time.time()
y_val_pred_rf = rf_model.predict(X_val_pca)
rf_pred_time = time.time() - start_pred_time

y_train_pred_rf = rf_model.predict(X_train_pca)
rf_train_accuracy = accuracy_score(y_train, y_train_pred_rf)
rf_val_accuracy = accuracy_score(y_val, y_val_pred_rf)

print(f"Training time: {rf_train_time:.4f} seconds")
print(f"Prediction time: {rf_pred_time:.4f} seconds")
print(f"Training accuracy: {rf_train_accuracy:.4f}")
print(f"Validation accuracy: {rf_val_accuracy:.4f}")

# Summary
print("\n" + "="*70)
print("SUMMARY - PCA REDUCED FEATURES")
print("="*70)
print(f"\n{'Model':<20} {'Train Time':<12} {'Pred Time':<12} {'Train Acc':<12} {'Val Acc':<12}")
print("-"*70)
print(f"{'Decision Tree':<20} {dt_train_time:<12.4f} {dt_pred_time:<12.4f} {dt_train_accuracy:<12.4f} {dt_val_accuracy:<12.4f}")
print(f"{'Random Forest':<20} {rf_train_time:<12.4f} {rf_pred_time:<12.4f} {rf_train_accuracy:<12.4f} {rf_val_accuracy:<12.4f}")
print("="*70)