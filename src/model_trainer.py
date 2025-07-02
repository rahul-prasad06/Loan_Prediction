import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.utils.logger import getLogger
from src.utils.exception import CustomException

mlflow.set_experiment("Loan Approval Tree Models")

logger = getLogger()

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info("✅ Loaded clean data")
        return df
    except Exception as e:
        raise CustomException("Error loading data", e)

def apply_smote(X, y):
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"✅ Applied SMOTE | Resampled classes: {dict(pd.Series(y_resampled).value_counts())}")
        return X_resampled, y_resampled
    except Exception as e:
        raise CustomException("Error in SMOTE resampling", e)

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")

    cm_path = f"artifacts/conf_matrix_{name}.png"
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(cm_path)
    plt.close()

    return acc, f1, roc_auc, cm_path

def train_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, scoring="accuracy", cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid

def feature_importance_selector(model, X):
    feature_imp = pd.DataFrame(data=model.feature_importances_, index=X.columns, columns=["Importance"])
    return feature_imp[feature_imp["Importance"] > 0].index.tolist()

def train_and_log_models(X_train, X_test, y_train, y_test):
    models_info = {
        "DecisionTree": {
            "model": DecisionTreeClassifier(),
            "params": {"criterion": ["gini", "entropy"], "max_depth": list(range(1, 20))}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": list(range(1, 20))}
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {"n_estimators": list(range(1, 20))}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss'),
            "params": {"n_estimators": list(range(1, 30)), "max_depth": [3, 5, 7, 9], "gamma": [0.1, 0.5, 0.8, 1.0]}
        }
    }

    best_model = None
    best_name = ""
    best_score = 0

    for name, config in models_info.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"🔍 Training {name} model...")
            best_estimator, grid = train_model(config["model"], config["params"], X_train, y_train)

            important_features = feature_importance_selector(best_estimator, X_train)
            X_train_imp = X_train[important_features]
            X_test_imp = X_test[important_features]

            best_estimator.fit(X_train_imp, y_train)
            y_pred = best_estimator.predict(X_train_imp)
            train_acc = accuracy_score(y_train, y_pred)
            cv_score = cross_val_score(best_estimator, X_train_imp, y_train, cv=5, scoring="accuracy").mean()
            logger.info(f"{name} Train Acc: {train_acc:.4f}, CV Score: {cv_score:.4f}")

            acc, f1, roc_auc, cm_path = evaluate_model(best_estimator, X_test_imp, y_test, name)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc})
            mlflow.log_artifact(cm_path)
            mlflow.sklearn.log_model(best_estimator, f"{name}_model")

            if acc > best_score:
                best_score = acc
                best_model = best_estimator
                best_name = name

    return best_model, best_name, best_score

def save_model(model, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"✅ Best model saved at {path}")
    except Exception as e:
        raise CustomException("Error saving model", e)

def main():
    try:
        df = load_data("data/processed/clean_data.csv")
        X = df.drop("loan_status", axis=1)
        y = df["loan_status"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train_bal, y_train_bal = apply_smote(X_train, y_train)

        best_model, best_name, best_score = train_and_log_models(X_train_bal, X_test, y_train_bal, y_test)

        save_model(best_model, "models/best_tree_model.pkl")
        logger.info(f"🏆 Best Model: {best_name} | Accuracy: {best_score:.4f}")

    except Exception as e:
        raise CustomException("Error in training pipeline", e)

if __name__ == "__main__":
    main()
