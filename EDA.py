import yfinance as yf
import pandas as pd
from ctw import CTW
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import skew

# Define the ticker symbol and date range
ticker = '^GSPC'
start_date = '2019-09-10'
end_date = '2024-10-09'

# Fetch the data
df = yf.download(ticker, start=start_date, end=end_date)


# Extract the 'Close' column
close_prices = df['Close'].dropna()

# Calculate statistics
mean_price = float(close_prices.mean())  # Convert to float
std_price = float(close_prices.std())    # Convert to float
skewness = float(skew(close_prices))     # Convert to float

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(close_prices, bins=50, alpha=0.6, color='b', edgecolor='black', density=True)

# Plot mean and standard deviation intervals
plt.axvline(mean_price, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_price:.2f}')
plt.axvline(mean_price + 3*std_price, color='g', linestyle='dashed', linewidth=2, label=f'+3 Std: {(mean_price + 3*std_price):.2f}')
plt.axvline(mean_price - 3*std_price, color='g', linestyle='dashed', linewidth=2, label=f'-3 Std: {(mean_price - 3*std_price):.2f}')

# Add title and labels
plt.title(f'Distribution of Closing Prices ({start_date} to {end_date})\nSkewness: {skewness:.2f}', fontsize=14)
plt.xlabel('Closing Price', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True)

# Show the plot
plt.show(block=True)