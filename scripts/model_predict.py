import pandas as pd
import joblib

def predict_stock(data_path, model_path):
    # Load data
    df = pd.read_csv(data_path)
    
    # Debug: Print column names to ensure 'Date' is present
    print("Columns in the CSV file:", df.columns)
    
    # Ensure 'Date' column exists and is formatted correctly
    if 'Date' not in df.columns:
        raise ValueError("Expected 'Date' column is missing from the dataset")
    
    # Convert 'Date' to datetime and handle errors
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)  # Drop rows where 'Date' couldn't be parsed

    # Load the trained model
    model = joblib.load(model_path)
    
    # Prepare the feature 'Close' for prediction
    X = df[['Close']]  # Using only 'Close' for prediction
    
    # Predict using the trained model
    df['Predictions'] = model.predict(X)
    
    # Optionally save predictions
    df.to_csv("../data/processed/AAPL_predictions_with_predictions.csv", index=False)
    print(f"Predictions saved at ../data/processed/AAPL_predictions_with_predictions.csv")
