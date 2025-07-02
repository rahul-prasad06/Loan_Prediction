import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
from src.utils.logger import getLogger
from src.utils.exception import CustomException

logger=getLogger()

MODEL_PATH = "models/best_tree_model.pkl"


def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Calculate total assets
        data["total_assest"] = data["residential_assets_value"] + data["commercial_assets_value"] + data["luxury_assets_value"] + data["bank_asset_value"]

        # Monthly calculations
        data["monthly"] = data["income_annum"] / 12
        data["month_loan"] = data["loan_amount"] / (data["loan_term"] * 12)

        # Ratios
        data["DTI"] = data["month_loan"] / data["monthly"]
        data["LTV"] = data["loan_amount"] / data["total_assest"]
        data["LTI"] = data["loan_amount"] / data["income_annum"]

        # Categorize risk levels
        data['LTI_Category'] = pd.cut(data['LTI'], bins=[0, 3, 5, 50], labels=['Low', 'Moderate', 'High'])
        data['LTV_Category'] = pd.cut(data['LTV'], bins=[0, 0.6, 0.8, 0.95, 2], labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical'])
        data['DTI_Category'] = pd.cut(data['DTI'], bins=[0, 0.2, 0.35, 0.5, 5], labels=['Low', 'Moderate', 'High', 'Critical'])

        # Ordinal Encoding (same as training)
        data["LTI_Category"] = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']]).fit_transform(data[["LTI_Category"]])
        data["LTV_Category"] = OrdinalEncoder(categories=[['Low Risk', 'Moderate Risk', 'High Risk', 'Critical']]).fit_transform(data[["LTV_Category"]])
        data["DTI_Category"] = OrdinalEncoder(categories=[['Low', 'Moderate', 'High', 'Critical']]).fit_transform(data[["DTI_Category"]])

        # Drop unused columns
        data.drop(columns=[
            "monthly", "month_loan", "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value", "income_annum", "loan_amount",
            "loan_term", "cibil_score", "no_of_dependents", "education", "self_employed",
            "total_assest", "LTI", "LTV", "DTI"
        ], inplace=True, errors='ignore')

        return data

    except Exception as e:
        raise CustomException("Error in input preprocessing", e)


def predict_loan_status(raw_input: pd.DataFrame) -> pd.DataFrame:
    try:
        # Load trained model
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully for prediction")

        # Preprocess the input
        processed_input = preprocess_input(raw_input.copy())

        # Predict
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)[:, 1]  # Probability of Approved

        result = pd.DataFrame({
            "prediction": prediction,
            "confidence": probability
        })
        result["loan_status"] = result["prediction"].map({1: "Approved", 0: "Rejected"})
        return result[["loan_status", "confidence"]]

    except Exception as e:
        raise CustomException("Error during prediction", e)
