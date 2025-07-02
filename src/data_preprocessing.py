import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import ttest_ind, chi2_contingency

from src.utils.logger import getLogger
from src.utils.exception import CustomException

logger = getLogger()

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✅ Data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error("❌ Failed to load data")
        raise CustomException(f"Error loading data: {e}", e)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.columns = df.columns.str.replace(' ', '')
        for col in ['education', 'self_employed', 'loan_status']:
            df[col] = df[col].str.lstrip()
        df['residential_assets_value'] = df['residential_assets_value'].replace(-10000, 0)
        df["total_assest"] = (
            df["residential_assets_value"] +
            df["commercial_assets_value"] +
            df["luxury_assets_value"] +
            df["bank_asset_value"]
        )
        df.drop(columns=[
            'residential_assets_value',
            'commercial_assets_value',
            'luxury_assets_value',
            'bank_asset_value'
        ], inplace=True)
        logger.info("✅ Data cleaned successfully")
        return df
    except Exception as e:
        logger.error("❌ Error in clean_data")
        raise CustomException("Error in clean_data", e)

def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["monthly"] = df["income_annum"] / 12
        df["month_loan"] = df["loan_amount"] / (df["loan_term"] * 12)
        df["DTI"] = df["month_loan"] / df["monthly"]
        df["LTV"] = df["loan_amount"] / df["total_assest"]
        df["LTI"] = df["loan_amount"] / df["income_annum"]
        df.drop(columns=["monthly", "month_loan"], inplace=True)
        logger.info("✅ Ratios created")
        return df
    except Exception as e:
        logger.error("❌ Error in create_ratios")
        raise CustomException("Error in create_ratios", e)

def categorize_risks(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['LTI_Category'] = pd.cut(df['LTI'], bins=[0, 3, 5, 10], labels=['Low', 'Moderate', 'High'])
        df['LTV_Category'] = pd.cut(df['LTV'], bins=[0, 0.6, 0.8, 0.95, 1.5], labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical'])
        df['DTI_Category'] = pd.cut(df['DTI'], bins=[0, 0.2, 0.35, 0.5, 5], labels=['Low', 'Moderate', 'High', 'Critical'])
        df.drop(columns=["LTI", "LTV", "DTI"], inplace=True)
        logger.info("✅ Risk categories assigned")
        return df
    except Exception as e:
        logger.error("❌ Error in categorize_risks")
        raise CustomException("Error in categorize_risks", e)

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["loan_status"] = df["loan_status"].map({'Approved': 1, 'Rejected': 0})
        df["LTI_Category"] = OrdinalEncoder(categories=[["Low", "Moderate", "High"]]).fit_transform(df[["LTI_Category"]])
        df["LTV_Category"] = OrdinalEncoder(categories=[["Low Risk", "Moderate Risk", "High Risk", "Critical"]]).fit_transform(df[["LTV_Category"]])
        df["DTI_Category"] = OrdinalEncoder(categories=[["Low", "Moderate", "High", "Critical"]]).fit_transform(df[["DTI_Category"]])
        logger.info("✅ Encoding completed")
        return df
    except Exception as e:
        logger.error("❌ Error in encode_features")
        raise CustomException("Error in encode_features", e)

def drop_unimportant_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['loan_id', 'no_of_dependents', 'education', 'self_employed', 'total_assest'], errors='ignore', inplace=True)
        logger.info("✅ Dropped unimportant features")
        return df
    except Exception as e:
        logger.error("❌ Error in drop_unimportant_features")
        raise CustomException("Error in drop_unimportant_features", e)

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    try:
        X = add_constant(df)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    except Exception as e:
        logger.error("❌ Error calculating VIF")
        raise CustomException("Error calculating VIF", e)

def perform_feature_selection(df: pd.DataFrame) -> None:
    try:
        continuous = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score']
        categorical = ['LTI_Category', 'LTV_Category', 'DTI_Category']

        logger.info("📊 T-Test Results:")
        for col in continuous:
            approved = df[df['loan_status'] == 1][col]
            rejected = df[df['loan_status'] == 0][col]
            stat, p_value = ttest_ind(approved, rejected)
            importance = "✅ Important" if p_value <= 0.05 else "❌ Not Important"
            logger.info(f"{col}: t={stat:.4f}, p={p_value:.4f} → {importance}")

        logger.info("📊 Chi-Square Results:")
        for col in categorical:
            table = pd.crosstab(df[col], df['loan_status'])
            stat, p_value, _, _ = chi2_contingency(table)
            importance = "✅ Important" if p_value <= 0.05 else "❌ Not Important"
            logger.info(f"{col}: chi2={stat:.4f}, p={p_value:.4f} → {importance}")
    except Exception as e:
        logger.error("❌ Error in perform_feature_selection")
        raise CustomException("Error in feature selection", e)

def preprocess(input_path: str, output_path: str):
    try:
        df = load_data(input_path)
        df = clean_data(df)
        df = create_ratios(df)
        df = categorize_risks(df)
        df = encode_features(df)
        df = drop_unimportant_features(df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Cleaned data saved to {output_path}")

        perform_feature_selection(df)
        vif_df = calculate_vif(df)
        logger.info("✅ VIF Results:\n" + vif_df.to_string(index=False))

    except Exception as e:
        logger.error("❌ Error in preprocess pipeline")
        raise CustomException("Error in preprocess pipeline", e)

if __name__ == "__main__":
    preprocess("data/raw/loan_approval_dataset.csv", "data/processed/clean_data.csv")
