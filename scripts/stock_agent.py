from typing import Union, List, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import logging
from pathlib import Path
import joblib
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockAgent:
    """
    Agent for making stock trading decisions using trained ML models.
    
    Attributes:
        model: Trained sklearn model
        feature_names: List of expected feature names
        confidence_threshold: Minimum probability threshold for making decisions
    """
    
    def __init__(
        self, 
        model: BaseEstimator,
        feature_names: List[str] = None,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize StockAgent.
        
        Args:
            model: Trained sklearn model
            feature_names: List of feature names expected by the model
            confidence_threshold: Minimum probability threshold for making decisions
        """
        self.model = model
        self.feature_names = feature_names or [
            'open_price', 'high_price', 'low_price',
            'close_price', 'volume', 'sentiment'
        ]
        self.confidence_threshold = confidence_threshold
        
    def validate_input(self, data: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and prepare input data."""
        try:
            if isinstance(data, (list, np.ndarray)):
                data = np.array(data)
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] != len(self.feature_names):
                    raise ValueError(
                        f"Expected {len(self.feature_names)} features, got {data.shape[1]}"
                    )
            elif isinstance(data, pd.DataFrame):
                missing_features = set(self.feature_names) - set(data.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                data = data[self.feature_names].values
            else:
                raise TypeError("Input must be list, numpy array, or pandas DataFrame")
                
            return data
            
        except Exception as e:
            logging.error(f"Input validation error: {str(e)}")
            raise
            
    def predict(self, stock_data: Union[List, np.ndarray, pd.DataFrame]) -> Dict[str, Union[str, float]]:
        """
        Make trading decision based on stock data.
        
        Args:
            stock_data: Stock market data matching the trained model's features
            
        Returns:
            Dictionary containing decision and confidence score
        """
        try:
            # Validate and prepare input
            data = self.validate_input(stock_data)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(data)
            confidence = max(probabilities[0])
            prediction = self.model.predict(data)[0]
            
            # Make decision based on confidence threshold
            if confidence < self.confidence_threshold:
                decision = "HOLD"
                reason = "Low confidence in prediction"
            else:
                decision = "BUY" if prediction == 1 else "SELL"
                reason = f"Confidence: {confidence:.2%}"
                
            return {
                "decision": decision,
                "confidence": confidence,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise
            
    def save_prediction(self, prediction: Dict, filepath: Union[str, Path]) -> None:
        """Save prediction to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'a') as f:
                f.write(f"{prediction}\n")
                
        except Exception as e:
            logging.error(f"Error saving prediction: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        from train_model import train_model, prepare_training_data, load_data_from_db
        
        # Load and prepare data
        df = load_data_from_db()
        X_train, X_test, y_train, y_test = prepare_training_data(df)
        model, _ = train_model(X_train, y_train)
        
        # Initialize agent
        agent = StockAgent(
            model=model,
            feature_names=X_train.columns.tolist(),
            confidence_threshold=0.6
        )
        
        # Example prediction
        latest_data = {
            'open_price': 175,
            'high_price': 180,
            'low_price': 170,
            'close_price': 178,
            'volume': 5000000,
            'sentiment': 0.3
        }
        
        prediction = agent.predict(pd.DataFrame([latest_data]))
        print("\nStock Trading Recommendation:")
        print(f"Decision: {prediction['decision']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Reason: {prediction['reason']}")
        
        # Save prediction
        agent.save_prediction(
            prediction,
            Path("predictions") / f"predictions_{datetime.now().strftime('%Y%m%d')}.txt"
        )
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")