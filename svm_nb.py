import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings("ignore") 
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
BASE_PLOT_DIR = "plots_nb_svm"

class DiabetesClassifier:
    def __init__(self, problem_type="binary"):
        self.problem_type = problem_type
        self.scaler = StandardScaler()
        self.pca = None

        # Raw data
        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_subset = self.y_subset = None

        # Scaled data
        self.X_train_scaled = self.X_val_scaled = self.X_subset_scaled = None

        # PCA-transformed data
        self.X_train_pca = self.X_val_pca = self.X_subset_pca = None

        # Models and results
        self.models = {}
        self.results = {}
        self.cv_results = {}
        self.target_col = None

    def load_data(self, train_path, val_path, subset_path):
        print(f"\nLoading {self.problem_type} data...")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        subset_df = pd.read_csv(subset_path)

        self.target_col = (
            "Diabetes_binary" if self.problem_type == "binary" else "Diabetes_012"
        )

        self.X_train = train_df.drop(columns=[self.target_col]).values
        self.y_train = train_df[self.target_col].values

        self.X_val = val_df.drop(columns=[self.target_col]).values
        self.y_val = val_df[self.target_col].values

        self.X_subset = subset_df.drop(columns=[self.target_col]).values
        self.y_subset = subset_df[self.target_col].values

        print(f"  Training samples:   {len(self.X_train)}")
        print(f"  Subset samples:     {len(self.X_subset)}")
        print(f"  Validation samples: {len(self.X_val)}")
        print(f"  Features:           {self.X_train.shape[1]}")
        print(f"  Classes:            {np.unique(self.y_train)}\n")

    def scale_data(self):
        print("Scaling data")
        self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_subset_scaled = self.scaler.transform(self.X_subset)

    #PCA dataset
    def apply_pca(self, n_components=None):
        #Apply PCA to scaled data --> reduce to half the number of features
      
        print(f"Applying PCA ({self.problem_type})")

        n_features = self.X_train_scaled.shape[1]
        if n_components is None:
            n_components = n_features // 2

        print(f"Reducing from {n_features} -> {n_components} components")

        self.pca = PCA(n_components=n_components, random_state=42)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_val_pca = self.pca.transform(self.X_val_scaled)
        self.X_subset_pca = self.pca.transform(self.X_subset_scaled)

        explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance retained: {explained:.4f} ({explained * 100:.2f}%)\n")

        return explained

    def save_pca_data(self, out_dir):
    
        # Save PCA-transformed datasets (train, val, subset) to CSV, including label column.
        # Columns are named 0,1,2,...,k-1,<label>.
        self._ensure_dir(out_dir)

        # number of PCA components
        n_components = self.X_train_pca.shape[1]
        feature_cols = [str(i) for i in range(n_components)]

        prefix = "binary" if self.problem_type == "binary" else "multi"

        # Train
        train_df_pca = pd.DataFrame(self.X_train_pca, columns=feature_cols)
        train_df_pca[self.target_col] = self.y_train
        train_path = os.path.join(out_dir, f"{prefix}_train_pca.csv")
        train_df_pca.to_csv(train_path, index=False)

        # Validation
        val_df_pca = pd.DataFrame(self.X_val_pca, columns=feature_cols)
        val_df_pca[self.target_col] = self.y_val
        val_path = os.path.join(out_dir, f"{prefix}_val_pca.csv")
        val_df_pca.to_csv(val_path, index=False)

        # Subset
        subset_df_pca = pd.DataFrame(self.X_subset_pca, columns=feature_cols)
        subset_df_pca[self.target_col] = self.y_subset
        subset_path = os.path.join(out_dir, f"{prefix}_subset_pca.csv")
        subset_df_pca.to_csv(subset_path, index=False)

        print("PCA data saved:")
        print(f"  {train_path}")
        print(f"  {val_path}")
        print(f"  {subset_path}\n")

    #METRICS

    def _calc_metrics(self, y_true, y_pred, y_train, y_train_pred,
                      train_time, pred_time, cv_time):
        """
        y_true       : validation labels
        y_pred       : validation predictions
        y_train      : training labels
        y_train_pred : training predictions
        """
        avg = "binary" if self.problem_type == "binary" else "macro"

        return {
            # Validation metrics
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),

            # Training metrics
            "train_accuracy": accuracy_score(y_train, y_train_pred),

            # Timing
            "train_time": train_time,
            "pred_time": pred_time,
            "cv_time": cv_time,
            "total_time": train_time + pred_time + cv_time,

            # Confusion matrix (validation)
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    #Training
    def _cv_and_train(self, model, param_grid, param_name, categorical,
                      model_key, model_label, k_folds=4, use_pca=False):
        print("-"* 10)
        print(f"{model_label} ({self.problem_type}) {'with PCA' if use_pca else '(no PCA)'}")
        print("-" * 10)
        if use_pca:
            X_train = self.X_train_pca
            X_val = self.X_val_pca
            X_subset = self.X_subset_pca
        else:
            X_train = self.X_train_scaled
            X_val = self.X_val_scaled
            X_subset = self.X_subset_scaled

        key = model_key + ("_pca" if use_pca else "")
        print(f"Hyperparameter grid: {param_grid}\n")

    #Cross-validation for hyperparameter 
        start_cv = time.time()
        grid = GridSearchCV(
            model,
            param_grid,
            cv=k_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_subset, self.y_subset)
        cv_time = time.time() - start_cv

        print(f"\nBest params: {grid.best_params_}")
        print(f"Best mean CV accuracy: {grid.best_score_:.4f}")
        print(f"CV time: {cv_time:.2f}s\n")

        cv_df = pd.DataFrame(grid.cv_results_)
        self.cv_results[key] = dict(
            df=cv_df,
            param_name=param_name,
            categorical=categorical,
            label=model_label + (" + PCA" if use_pca else "")
        )

    #Train best model on training set
        best_model = grid.best_estimator_
        t0 = time.time()
        best_model.fit(X_train, self.y_train)
        train_time = time.time() - t0

    #Training predictions for train accuracy
        y_train_pred = best_model.predict(X_train)

    #Validation prediction 
        t1 = time.time()
        y_pred = best_model.predict(X_val)
        pred_time = time.time() - t1

    #Metrics
        metrics = self._calc_metrics(
            y_true=self.y_val,
            y_pred=y_pred,
            y_train=self.y_train,
            y_train_pred=y_train_pred,
            train_time=train_time,
            pred_time=pred_time,
            cv_time=cv_time
        )

        self.models[key] = best_model
        self.results[key] = metrics

        print(f"{model_label} Train Accuracy:      {metrics['train_accuracy']:.4f}")
        print(f"{model_label} Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_label} Train Time:          {metrics['train_time']:.4f}s")
        print(f"{model_label} Predict Time:        {metrics['pred_time']:.4f}s")
        print(f"{model_label} CV Time:             {metrics['cv_time']:.4f}s")
        print(f"{model_label} TOTAL Time:          {metrics['total_time']:.4f}s\n")

    #NB & SVMs

    def train_naive_bayes(self, k_folds=4, use_pca=False):
        self._cv_and_train(
            GaussianNB(),
            {"var_smoothing": [1e-6, 1e-3, 1e-1, 1, 10]},
            "var_smoothing",
            False,
            "nb",
            "Naive Bayes",
            k_folds,
            use_pca
        )

    def train_svm(self, k_folds=4, use_pca=False):
        self._cv_and_train(
            SVC(C=1.0, random_state=42),
            {"kernel": ["linear, rbf, sigmoid"]},
            "kernel",
            True,
            "svm",
            "SVM",
            k_folds,
            use_pca
        )

    #Plotters lol

    def _ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def save_cv_plots(self, out_dir):
        self._ensure_dir(out_dir)

        for key, info in self.cv_results.items():
            cv_df = info["df"]
            param_name = info["param_name"]
            categorical = info["categorical"]
            label = info["label"]

            param_col = f"param_{param_name}"
            params = cv_df[param_col]
            scores = cv_df["mean_test_score"]

            plt.figure(figsize=(8, 4))

            if categorical:
                x = np.arange(len(params))
                labels_x = [str(p) for p in params]
                plt.bar(x, scores)
                plt.xticks(x, labels_x)

                for i, v in enumerate(scores):
                    plt.text(i, v + 0.001, f"({v:.4f})", ha='center', fontsize=10)

                plt.ylim(min(scores) - 0.01, max(scores) + 0.02)
            else:
                vals = np.array(params, dtype=float)
                plt.plot(vals, scores, marker="o")

                for x_val, y_val in zip(vals, scores):
                    plt.text(x_val, y_val + 0.001, f"({y_val:.4f})",
                             ha='center', fontsize=10)

                plt.xscale("log")
                plt.xticks(vals, [str(v) for v in vals])
                plt.ylim(min(scores) - 0.01, max(scores) + 0.02)

            plt.xlabel(param_name)
            plt.ylabel("Mean CV Accuracy")
            plt.title(f"{label} CV ({self.problem_type})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(out_dir, f"{key}_cv.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved CV plot: {save_path}")

    def save_metrics_comparison(self, model_keys, out_path, title):
        metrics = ["accuracy", "precision", "recall", "f1", "train_accuracy"]
        data = {k: [self.results[k][m] for m in metrics]
                for k in model_keys if k in self.results}

        if not data:
            print("No metrics to plot.")
            return

        df = pd.DataFrame(data, index=metrics)
        ax = df.plot(kind="bar")
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        plt.ylim(0, 1)
        plt.tight_layout()
        self._ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=300)
        plt.close()

    def save_runtime_comparison(self, model_keys, out_path, title):
        train_times, pred_times, cv_times, total_times, labels = [], [], [], [], []

        for key in model_keys:
            if key in self.results:
                r = self.results[key]
                labels.append(key)
                train_times.append(r["train_time"])
                pred_times.append(r["pred_time"])
                cv_times.append(r["cv_time"])
                total_times.append(r["total_time"])

        if not labels:
            print("No runtime data to plot.")
            return

        x = np.arange(len(labels))
        width = 0.2

        plt.figure(figsize=(10, 5))
        plt.bar(x - 1.5 * width, train_times, width, label="Train")
        plt.bar(x - 0.5 * width, pred_times, width, label="Predict")
        plt.bar(x + 0.5 * width, cv_times, width, label="CV")
        plt.bar(x + 1.5 * width, total_times, width, label="Total")

        plt.xticks(x, labels)
        plt.ylabel("Time (s)")
        plt.title(title)
        plt.legend()
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

        self._ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=300)
        plt.close()

    def save_confusion_matrix(self, model_key, out_path, title):
        if model_key not in self.results:
            print(f"No results for {model_key}")
            return

        cm = self.results[model_key]["confusion_matrix"]
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        self._ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=300)
        plt.close()

    def save_all_plots(self, base_dir):
        self._ensure_dir(base_dir)

        self.save_cv_plots(base_dir)

        self.save_metrics_comparison(
            ["nb", "svm", "nb_pca", "svm_pca"],
            os.path.join(base_dir, "metrics.png"),
            f"{self.problem_type.capitalize()} Classification: Metrics"
        )

        self.save_runtime_comparison(
            ["nb", "svm", "nb_pca", "svm_pca"],
            os.path.join(base_dir, "runtime.png"),
            f"{self.problem_type.capitalize()} Classification: Runtime"
        )

        self.save_confusion_matrix(
            "nb",
            os.path.join(base_dir, "nb_confusion.png"),
            f"{self.problem_type.capitalize()} Naive Bayes Confusion Matrix"
        )
        self.save_confusion_matrix(
            "svm",
            os.path.join(base_dir, "svm_confusion.png"),
            f"{self.problem_type.capitalize()} SVM Confusion Matrix"
        )

#Main function
def main():
    print("\n" + "#" * 60)
    print("CSE514 Project 2: Diabetes Classification (NB + SVM)")
    print("#" * 60 + "\n")

    #BINARY CLASSIFICATION
    print("\n BINARY CLASSIFICATION n")
    binary = DiabetesClassifier("binary")
    binary.load_data("b_train.csv", "b_validation.csv", "b_subset.csv")
    binary.scale_data()
    binary.train_naive_bayes()
    binary.train_svm()
    binary.apply_pca()
    binary.save_pca_data(os.path.join(BASE_PLOT_DIR, "binary"))
    binary.train_naive_bayes(use_pca=True)
    binary.train_svm(use_pca=True)
    binary.save_all_plots(os.path.join(BASE_PLOT_DIR, "binary"))

    #MULTICLASS CLASSIFICATION 
    print("\nMULTICLASS CLASSIFICATION\n")
    multi = DiabetesClassifier("multiclass")
    multi.load_data("m_train.csv", "m_validation.csv", "m_subset.csv")
    multi.scale_data()
    multi.train_naive_bayes()
    multi.train_svm()
    multi.apply_pca()
    multi.save_pca_data(os.path.join(BASE_PLOT_DIR, "multiclass"))
    multi.train_naive_bayes(use_pca=True)
    multi.train_svm(use_pca=True)
    multi.save_all_plots(os.path.join(BASE_PLOT_DIR, "multiclass"))

if __name__ == "__main__":
    main()
