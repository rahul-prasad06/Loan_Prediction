import pandas as pd
from src.predict_pipeline import predict_loan_status

# Simulate input
data = pd.DataFrame({
    'no_of_dependents': [2],
    'education': ["Graduate"],
    'self_employed': ["Yes"],
    'income_annum': [850000],
    'loan_amount': [500000],
    'loan_term': [5],
    'cibil_score': [814],
    'residential_assets_value': [0],
    'commercial_assets_value': [0],
    'luxury_assets_value': [0],
    'bank_asset_value': [600000]
})

# Predict
prediction = predict_loan_status(data)
print(prediction)
