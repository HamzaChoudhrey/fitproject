import joblib
import pandas as pd


# Correct model file path
model_path = "C:/Users/LENOVO/Downloads/linear_regression_model.pkl"
loaded_model = joblib.load(model_path)

new_data = pd.DataFrame({
    'Revenue per 1000 Views (USD)': [1.2],
    'Monetized Playbacks (Estimate)': [5000],
    'Playback-Based CPM (USD)': [333.4],
    'CPM (USD)': [444.5],
    'Ad Impressions': [2000],
    'Estimated AdSense Revenue (USD)': [800],
    'YouTube Ads Revenue (USD)': [12000]
})

new_predictions = loaded_model.predict(new_data)
print(f"Predicted Estimated Revenue: {new_predictions[0]}")
