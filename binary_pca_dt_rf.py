import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('binary_train.csv')
val_data = pd.read_csv('binary_validation.csv')

# Separate features and target
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_val = val_data.iloc[:, :-1]
y_val = val_data.iloc[:, -1]

print("="*70)
print("ORIGINAL BINARY DATASET")
print("="*70)
print(f"Number of features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

print("\n" + "="*70)
print("APPLYING PCA DIMENSION REDUCTION (Binary)")
print("="*70)

n_components = X_train.shape[1] // 2
print(f"Reducing from {X_train.shape[1]} to {n_components} features")

pca = PCA(n_components=n_components)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_val_pca = pca.transform(X_val)

print(f"Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
print(f"New train shape: {X_train_pca.shape}")
print(f"New val shape: {X_val_pca.shape}")

# Save PCA datasets
train_pca_df = pd.DataFrame(X_train_pca)
train_pca_df['Diabetes_binary'] = y_train.values
train_pca_df.to_csv('binary_train_pca.csv', index=False)

val_pca_df = pd.DataFrame(X_val_pca)
val_pca_df['Diabetes_binary'] = y_val.values
val_pca_df.to_csv('binary_validation_pca.csv', index=False)

print("\nSaved:")
print(" - binary_train_pca.csv")
print(" - binary_validation_pca.csv")

print("\n" + "="*70)
print("CROSS-VALIDATION ON PCA-REDUCED BINARY DATA")
print("="*70)

dt_max_depths = [3, 5, 7, 10, 15, 20]
rf_n_estimators = [10, 50, 100, 200, 300]
k_folds = 5

print("\nDecision Tree CV:")
dt_mean_scores = []
dt_std_scores = []

for depth in dt_max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(dt, X_train_pca, y_train, cv=k_folds, scoring='accuracy')
    dt_mean_scores.append(scores.mean())
    dt_std_scores.append(scores.std())
    print(f"max_depth={depth}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

best_dt_depth = dt_max_depths[np.argmax(dt_mean_scores)]
print(f"\nBest DT depth: {best_dt_depth}")

print("\nRandom Forest CV:")
rf_mean_scores = []
rf_std_scores = []

for n_est in rf_n_estimators:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    scores = cross_val_score(rf, X_train_pca, y_train, cv=k_folds, scoring='accuracy')
    rf_mean_scores.append(scores.mean())
    rf_std_scores.append(scores.std())
    print(f"n_estimators={n_est}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

best_rf_estimators = rf_n_estimators[np.argmax(rf_mean_scores)]
print(f"\nBest RF estimators: {best_rf_estimators}")

plt.figure(figsize=(10, 6))
plt.plot(dt_max_depths, dt_mean_scores, marker='o', linestyle='-', 
         linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('max_depth', fontsize=12)
plt.ylabel('Mean CV Accuracy', fontsize=12)
plt.title('Decision Tree Cross-Validation (Binary PCA)', fontsize=14)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('binary_dt_cv_pca_results.png', dpi=300, bbox_inches='tight')
print("\nSaved: binary_dt_cv_pca_results.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(rf_n_estimators, rf_mean_scores, marker='o', linestyle='-', 
         linewidth=2, markersize=10, color='#1f77b4')
plt.xlabel('n_estimators', fontsize=12)
plt.ylabel('Mean CV Accuracy', fontsize=12)
plt.title('Random Forest Cross-Validation (Binary PCA)', fontsize=14)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('binary_rf_cv_pca_results.png', dpi=300, bbox_inches='tight')
print("Saved: binary_rf_cv_pca_results.png")
plt.close()

print("\n" + "="*70)
print("TRAINING FINAL MODELS (Binary, PCA)")
print("="*70)

print(f"\nDecision Tree (max_depth={best_dt_depth})")
start = time.time()
dt_model = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
dt_model.fit(X_train_pca, y_train)
dt_train_time = time.time() - start

start = time.time()
y_val_pred_dt = dt_model.predict(X_val_pca)
dt_pred_time = time.time() - start

dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train_pca))
dt_val_acc = accuracy_score(y_val, y_val_pred_dt)

print(f"\nRandom Forest (n_estimators={best_rf_estimators})")
start = time.time()
rf_model = RandomForestClassifier(n_estimators=best_rf_estimators, random_state=42)
rf_model.fit(X_train_pca, y_train)
rf_train_time = time.time() - start

start = time.time()
y_val_pred_rf = rf_model.predict(X_val_pca)
rf_pred_time = time.time() - start

rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train_pca))
rf_val_acc = accuracy_score(y_val, y_val_pred_rf)

dt_results = pd.DataFrame({
    "Model": ["Decision Tree"],
    "Best_Param": [best_dt_depth],
    "Train_Time": [dt_train_time],
    "Pred_Time": [dt_pred_time],
    "Train_Accuracy": [dt_train_acc],
    "Validation_Accuracy": [dt_val_acc]
})

rf_results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Best_Param": [best_rf_estimators],
    "Train_Time": [rf_train_time],
    "Pred_Time": [rf_pred_time],
    "Train_Accuracy": [rf_train_acc],
    "Validation_Accuracy": [rf_val_acc]
})

dt_results.to_csv("binary_dt_results.csv", index=False)
rf_results.to_csv("binary_rf_results.csv", index=False)

print("\nSaved result CSV files:")
print(" - binary_dt_results.csv")
print(" - binary_rf_results.csv")

print("\n" + "="*70)
print("SUMMARY (Binary PCA)")
print("="*70)
print(f"{'Model':<20} {'Train Time':<12} {'Pred Time':<12} {'Train Acc':<12} {'Val Acc':<12}")
print("-"*70)
print(f"{'Decision Tree':<20} {dt_train_time:<12.4f} {dt_pred_time:<12.4f} {dt_train_acc:<12.4f} {dt_val_acc:<12.4f}")
print(f"{'Random Forest':<20} {rf_train_time:<12.4f} {rf_pred_time:<12.4f} {rf_train_acc:<12.4f} {rf_val_acc:<12.4f}")
print("="*70)