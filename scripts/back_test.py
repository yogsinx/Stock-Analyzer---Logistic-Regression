import numpy as np

def backtest(model, X, y):
    predictions = model.predict(X)
    
    # Simulate trading
    capital = 100000  # Starting capital
    shares = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            shares = capital // X.iloc[i]['close']  # Assuming 'close' is the stock price
            capital -= shares * X.iloc[i]['close']
        elif predictions[i] == 0 and shares > 0:  # Sell signal
            capital += shares * X.iloc[i]['close']
            shares = 0
    final_capital = capital + (shares * X.iloc[-1]['close'])
    print(f"Final capital after backtesting: {final_capital}")
