import yfinance as yf
import pandas as pd
from ctw import CTW
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the ticker symbol and date range
ticker = '^GSPC'
start_date = '2019-09-10'
end_date = '2024-10-09'

# Fetch the data
df = yf.download(ticker, start=start_date, end=end_date)

binary=False
if binary:
    df['Next_Close_Higher'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    max_depth=20
    nb_symbols=2

else:
    max_depth=13
    nb_symbols=3
    # Calculate daily percentage change
    df['Next_Close_temp'] = df['Close'].pct_change()


    # Define the quantile thresholds
    lower_bound = -0.005
    upper_bound = 0.005

    lower_bound = df['Next_Close_temp'].quantile(0.35)# 25th percentile (Q1)
    upper_bound = df['Next_Close_temp'].quantile(0.65)  # 75th percentile (Q3)

    # Classify the movements
    df['Next_Close_Higher'] = 0  # Default to 0
    df.loc[df['Next_Close_temp'] > upper_bound, 'Next_Close_Higher'] = 2  # Upper movement
    df.loc[df['Next_Close_temp'] < lower_bound, 'Next_Close_Higher'] = 1  # Lower movement


    # Display the first few rows
    print(df.head())

    # Prepare sequence for CTW model
sequence = df['Next_Close_Higher'].values.tolist()

L=[]
depths=[i for i in range(1,max_depth)]
for depth in depths:
    # Initialize CTW model
    ctw_model = CTW(depth=depth, symbols=nb_symbols)  # Binary symbols (0 and 1)

    # Train the CTW model (this is done in the predict_sequence method)
    # We use the first part of the sequence to build the tree and predict
    distributions = ctw_model.predict_sequence(sequence, sideseq=None)
    # The distributions give the probability of the next symbol (higher or not)
    # Convert the distributions into binary predictions (0 or 1)
    predictions = np.argmax(distributions, axis=0)  # Taking the most probable symbol

    # Create a dataframe to compare actual and predicted values
    df_predictions = pd.DataFrame({
        'Actual': df['Next_Close_Higher'][depth:],  # Skip the first few rows due to context length
        'Predicted': predictions
    })



    # Assuming df_predictions is already correctly defined in your previous steps
    accuracy = accuracy_score(df_predictions['Actual'], df_predictions['Predicted'])
    L.append(accuracy)
    print(f'accuracy is for {depth}',accuracy)

plt.plot(L)
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.title("CTW with Different Depths")
plt.show(block=True)
