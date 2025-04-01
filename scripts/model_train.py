import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(data_path, model_path):
    # Load data
    df = pd.read_csv(data_path)
    
    # Debug: Print column names to ensure 'Close' is present
    print("Columns in the CSV file:", df.columns)

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("Expected 'Close' column is missing from the dataset")
    
    # Convert 'Close' column to numeric, forcing errors to NaN
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Drop rows where 'Close' is NaN (could be caused by invalid data like 'AAPL')
    df.dropna(subset=['Close'], inplace=True)

    # Feature engineering: Adding a simple binary target (up/down movement based on close price)
    df['Target'] = df['Close'].shift(-1) > df['Close']  # Predicting up/down based on next day's price
    df.dropna(subset=['Target'], inplace=True)  # Drop the last row which will have no target value

    # Prepare features and target variable
    X = df[['Close']]  # Features
    y = df['Target']  # Target variable

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Model Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model trained and saved at {model_path}")

    # Save predictions (for visualization or further analysis)
    df['Predictions'] = model.predict(X)
    df.to_csv("../data/processed/AAPL_predictions.csv", index=False)
    print(f"Predictions saved at ../data/processed/AAPL_predictions.csv")

# Example usage
# train_model('path/to/your/AAPL_stock.csv', 'path/to/your/stock_model.pkl')
