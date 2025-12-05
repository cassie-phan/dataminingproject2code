import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# read training and validation data
train_data = pd.read_csv('multiclass_train.csv')
val_data = pd.read_csv('multiclass_validation.csv')

# separate features and target
X_train = train_data.drop('Diabetes_012', axis=1)
y_train = train_data['Diabetes_012']
X_val = val_data.drop('Diabetes_012', axis=1)
y_val = val_data['Diabetes_012']

print("="*70)
print("TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
print("="*70)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")

# Decision Tree with best hyperparameter (max_depth=7)
print("\n" + "-"*70)
print("DECISION TREE (max_depth=7)")
print("-"*70)

start_time = time.time()
dt_model = DecisionTreeClassifier(max_depth=7, random_state=42)
dt_model.fit(X_train, y_train)
start_pred_time = time.time()
y_val_pred_dt = dt_model.predict(X_val)
dt_pred_time = time.time() - start_pred_time
print(f"Prediction time: {dt_pred_time:.4f} seconds")
dt_train_time = time.time() - start_time

# prections
y_train_pred_dt = dt_model.predict(X_train)
y_val_pred_dt = dt_model.predict(X_val)

# calc accuracies
dt_train_accuracy = accuracy_score(y_train, y_train_pred_dt)
dt_val_accuracy = accuracy_score(y_val, y_val_pred_dt)

print(f"\nTraining time: {dt_train_time:.4f} seconds")
print(f"Training accuracy: {dt_train_accuracy:.4f}")
print(f"Validation accuracy: {dt_val_accuracy:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred_dt, target_names=['No Diabetes', 'Prediabetes', 'Diabetes']))

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_dt))

# Random Forest with best hyperparameter (n_estimators=300)
print("\n" + "-"*70)
print("RANDOM FOREST (n_estimators=300)")
print("-"*70)

start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)
start_pred_time = time.time()
y_val_pred_rf = rf_model.predict(X_val)
rf_pred_time = time.time() - start_pred_time
print(f"Prediction time: {rf_pred_time:.4f} seconds")
rf_train_time = time.time() - start_time

y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)

rf_train_accuracy = accuracy_score(y_train, y_train_pred_rf)
rf_val_accuracy = accuracy_score(y_val, y_val_pred_rf)

print(f"\nTraining time: {rf_train_time:.4f} seconds")
print(f"Training accuracy: {rf_train_accuracy:.4f}")
print(f"Validation accuracy: {rf_val_accuracy:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred_rf, target_names=['No Diabetes', 'Prediabetes', 'Diabetes']))

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_rf))

print("\n" + "="*70)
print("Summary Comparisons")
print("="*70)
print(f"\n{'Model':<20} {'Training Time':<15} {'Train Acc':<12} {'Val Acc':<12}")
print("-"*70)
print(f"{'Decision Tree':<20} {dt_train_time:<15.4f} {dt_train_accuracy:<12.4f} {dt_val_accuracy:<12.4f}")
print(f"{'Random Forest':<20} {rf_train_time:<15.4f} {rf_train_accuracy:<12.4f} {rf_val_accuracy:<12.4f}")
print("="*70)