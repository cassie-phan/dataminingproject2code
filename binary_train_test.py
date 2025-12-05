import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('binary_train.csv')
val_data = pd.read_csv('binary_validation.csv')

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_val = val_data.iloc[:, :-1]
y_val = val_data.iloc[:, -1]

print("="*70)
print("TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS - BINARY CLASSIFICATION")
print("="*70)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")
print(f"\nTarget variable: {train_data.columns[-1]}")
print(f"Training class distribution:\n{y_train.value_counts()}")

best_dt_depth = 7

print("\n" + "-"*70)
print(f"DECISION TREE (max_depth={best_dt_depth})")
print("-"*70)

start_time = time.time()
dt_model = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_time = time.time() - start_time

start_pred_time = time.time()
y_val_pred_dt = dt_model.predict(X_val)
dt_pred_time = time.time() - start_pred_time

y_train_pred_dt = dt_model.predict(X_train)

dt_train_accuracy = accuracy_score(y_train, y_train_pred_dt)
dt_val_accuracy = accuracy_score(y_val, y_val_pred_dt)

print(f"\nTraining time: {dt_train_time:.4f} seconds")
print(f"Prediction time: {dt_pred_time:.4f} seconds")
print(f"Training accuracy: {dt_train_accuracy:.4f}")
print(f"Validation accuracy: {dt_val_accuracy:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred_dt))

best_rf_estimators = 300

print("\n" + "-"*70)
print(f"RANDOM FOREST (n_estimators={best_rf_estimators})")
print("-"*70)

start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=best_rf_estimators, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

start_pred_time = time.time()
y_val_pred_rf = rf_model.predict(X_val)
rf_pred_time = time.time() - start_pred_time

y_train_pred_rf = rf_model.predict(X_train)

rf_train_accuracy = accuracy_score(y_train, y_train_pred_rf)
rf_val_accuracy = accuracy_score(y_val, y_val_pred_rf)

print(f"\nTraining time: {rf_train_time:.4f} seconds")
print(f"Prediction time: {rf_pred_time:.4f} seconds")
print(f"Training accuracy: {rf_train_accuracy:.4f}")
print(f"Validation accuracy: {rf_val_accuracy:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred_rf))

print("\n" + "="*70)
print("SUMMARY COMPARISON - BINARY CLASSIFICATION")
print("="*70)
print(f"\n{'Model':<20} {'Train Time':<12} {'Pred Time':<12} {'Train Acc':<12} {'Val Acc':<12}")
print("-"*70)
print(f"{'Decision Tree':<20} {dt_train_time:<12.4f} {dt_pred_time:<12.4f} {dt_train_accuracy:<12.4f} {dt_val_accuracy:<12.4f}")
print(f"{'Random Forest':<20} {rf_train_time:<12.4f} {rf_pred_time:<12.4f} {rf_train_accuracy:<12.4f} {rf_val_accuracy:<12.4f}")
print("="*70)

print("\n" + "="*70)
print("FORMATTED RESULTS FOR REPORT")
print("="*70)
print(f"\nDecision Tree: The trained DT model achieved a validation accuracy of {dt_val_accuracy:.4f} / {dt_val_accuracy*100:.0f}% on the validation data.")
print(f"  - Training time: {dt_train_time:.4f} seconds")
print(f"  - Prediction time: {dt_pred_time:.4f} seconds")

print(f"\nRandom Forest: The trained RF model achieved a validation accuracy of {rf_val_accuracy:.4f} / {rf_val_accuracy*100:.0f}% on the validation data.")
print(f"  - Training time: {rf_train_time:.4f} seconds")
print(f"  - Prediction time: {rf_pred_time:.4f} seconds")