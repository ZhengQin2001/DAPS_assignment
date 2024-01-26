import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from model_training import ModelTraining

def fetch_true_prices(start_date, end_date, ticker='AAPL'):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return mse, mae, r2


def plot_results(true_values, predicted_values, save_path=None):
    # Plot for Actual vs Predicted Prices
    plt.figure(figsize=(13, 6))
    plt.plot(true_values, label='Actual Prices')
    plt.plot(predicted_values, label='Predicted Prices')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    if save_path:
        plt.savefig(os.path.join(save_path, 'price_comparison_w_aux.png'))

    # Plot for Residuals
    residuals = true_values - predicted_values
    sns.jointplot(x=true_values, y=residuals, kind='reg')
    plt.xlabel('Actual Prices')
    plt.ylabel('Residuals')

    if save_path:
        plt.savefig(os.path.join(save_path, 'residuals_plot_w_aux.png'))

    # Statistics of residuals
    mean_residual = np.mean(residuals)
    median_residual = np.median(residuals)
    skewness_residual = pd.Series(residuals).skew()
    print(f"Mean Residual: {mean_residual}\nMedian Residual: {median_residual}\nSkewness: {skewness_residual}")

def align_predictions_with_actual(prices, predictions):
    # Convert actual prices to a DataFrame
    prices_df = pd.DataFrame(prices)
    prices_df.reset_index(inplace=True)
    prices_df.columns = ['Date', 'ActualPrice']

    # Create a DataFrame for predictions
    prediction_dates = pd.date_range(start='2023-04-01', end='2023-04-30')
    predictions_df = pd.DataFrame(predictions, index=prediction_dates, columns=['PredictedClose'])

    # Align predictions with actual prices
    aligned_predictions = pd.merge(prices_df, predictions_df, left_on='Date', right_index=True, how='left')
    return aligned_predictions.dropna()  # Drop rows where predictions are not available

def main(aux=True):
    # Load and preprocess data
    filename = 'src/data_preprocessing/filled_combined_data.json'
    model_trainer = ModelTraining(filename, time_step=10)
    model_trainer.train(epochs=30, batch_size=16, auxiliary=aux)

    # Prediction range
    start_date = '2023-04-01'
    end_date = '2023-04-30'

    # Get predictions
    predictions = model_trainer.recursive_forecast(start_date, end_date, auxiliary=aux)

    print(predictions)
    # Fetch true closing prices
    true_prices = fetch_true_prices(start_date, end_date)

    aligned_predictions = align_predictions_with_actual(true_prices, predictions)
    aligned_predictions.to_csv('predictions_w_aux.csv', index=False)
    # aligned_predictions = pd.read_csv('predictions_w_aux.csv')

    actual_prices = aligned_predictions['ActualPrice'].values
    predicted_prices = aligned_predictions['PredictedClose'].values

    # Evaluate predictions
    mse, mae, r2 = evaluate_predictions(actual_prices, predicted_prices)
    print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')

    # Plot results
    plot_results(actual_prices, predicted_prices, "src/model_training")

if __name__ == "__main__":
    main(aux=True)
