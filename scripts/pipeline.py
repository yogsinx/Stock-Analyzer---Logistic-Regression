import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.database import create_stock_table
from scripts.model_train import train_model
from scripts.model_predict import predict_stock

# Define directories for raw data and model saving
raw_data_dir = "../data/raw"
processed_data_dir = "data/processed"
model_dir = "models"

# Step 1: Create stock table
print("Starting AI pipeline...")
create_stock_table()

# Step 2: Train the model and save it
train_model(f"{raw_data_dir}/AAPL_stock.csv", f"{model_dir}/stock_model.pkl")

# Step 3: Make predictions using the trained model
predict_stock(f"{raw_data_dir}/AAPL_stock.csv", f"{model_dir}/stock_model.pkl")

