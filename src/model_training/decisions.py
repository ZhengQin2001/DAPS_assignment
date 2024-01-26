from model_training import ModelTraining
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from evaluation import fetch_true_prices

def advise_action(predictions, start_date, end_date, budget, holdings, auxiliary=False):
    current_date = pd.to_datetime(start_date)
    start_date = pd.to_datetime(start_date) + timedelta(days=1)

    actions = []
    dates = pd.date_range(start_date, end_date)
    current_price = None
    
    for i, predicted_price in enumerate(predictions):
        if i >= len(dates):  # Ensure we do not go beyond the end date
            break
        # Assume we have the current price from the last available data
        try:
            current_price_data = fetch_true_prices(current_date, dates[i])
            if current_price_data.empty:
                # If there's no data for the current date, assume non-trading day and append 0 action
                actions.append(0)
                continue  # Skip to the next iteration
            else:
                current_price = current_price_data.iloc[-1]  # Get the last price if there are multiple
        except Exception as e:
            print(f"An error occurred while fetching true prices: {e}")
            actions.append(0)
            continue  # Skip to the next iteration
        print(current_price)
        print("continue")
        
        if current_price:
            # Calculate the number of stocks that can be bought or sold
            if predicted_price > current_price:
                # If predicted price is higher than current, buy
                stocks_to_buy = budget // current_price
                if stocks_to_buy > 0:
                    budget -= stocks_to_buy * current_price
                    holdings += stocks_to_buy
                    actions.append(stocks_to_buy)
                else:
                    actions.append(0)
            elif predicted_price < current_price:
                # If predicted price is lower than current, sell
                if holdings > 0:
                    budget += holdings * current_price
                    actions.append(-holdings)
                    holdings = 0
                else:
                    actions.append(0)
            else:
                # If predicted price is same as current, hold
                actions.append(0)

        resulting_assets = budget + holdings * current_price  # assuming we can sell all holdings at the last known price
    return resulting_assets, actions

def evaluate_actions(start_date, end_date, actions, trading_days, actual_prices):
    budget = 10000
    holdings = 0
    trading_day_index = 0
    ideal_final_asset = budget

    for i, action in enumerate(actions):
        # Ensure trading_day_index is within the range of actual_prices
        if trading_day_index >= len(actual_prices) - 1:
            break

        # Find the next trading day
        while trading_days[trading_day_index] < start_date:
            trading_day_index += 1
            if trading_day_index >= len(trading_days):
                break  # Avoid going out of range
        
        if trading_day_index >= len(trading_days):
            break  # Stop if no more trading days

        # Get the price for the current trading day
        current_price = actual_prices[trading_day_index]

        if action > 0 and budget >= current_price:
            # Buy action
            stocks_to_buy = budget // current_price
            budget -= stocks_to_buy * current_price
            holdings += stocks_to_buy
        elif action < 0 and holdings > 0:
            # Sell action
            stocks_to_sell = holdings
            budget += stocks_to_sell * current_price
            holdings -= stocks_to_sell
        # Calculate the ideal final asset assuming perfect foresight
        if trading_day_index < len(actual_prices) - 1 and actual_prices[trading_day_index + 1] > current_price:
            ideal_stocks_to_buy = ideal_final_asset // current_price
            ideal_final_asset += ideal_stocks_to_buy * (actual_prices[trading_day_index + 1] - current_price)
        
        trading_day_index += 1
        if trading_day_index >= len(trading_days) or trading_days[trading_day_index] > end_date:
            break

    resulting_asset = budget + holdings * actual_prices[trading_day_index - 1]
    return resulting_asset, ideal_final_asset

def trim_actions_on_non_trading_days(all_dates, actions, trading_days):
    """
    Trims the actions list to remove actions on non-trading days.

    :param all_dates: A list of dates including both trading and non-trading days.
    :param actions: A list of actions corresponding to all_dates.
    :param trading_days: A list of actual trading days.
    :return: A list of actions trimmed to include only trading days.
    """
    trimmed_actions = []
    for date, action in zip(all_dates, actions):
        if date in trading_days:
            trimmed_actions.append(action)
    return trimmed_actions


def main():
    model_trainer = ModelTraining('src/data_preprocessing/filled_combined_data.json')
    auxiliary = True
    model_trainer.train(epochs=15, batch_size=16, auxiliary=auxiliary)

    start_date = '2023-01-01'
    end_date = '2023-01-31'
    all_dates = pd.date_range(start_date, end_date)


    # Initial budget and holdings
    initial_budget = 10000
    initial_holdings = 0
    prediction = model_trainer.recursive_forecast(start_date, end_date, auxiliary)

    # Test Example
    # prediction = [139.17366, 134.81131, 128.39774, 119.58645, 109.352005, 98.32252, 186.36118, 74.44431, 63.325397, 63.143955, 62.971233, 162.84242, 62.766193, 62.74123, 62.76565, 62.838123, 62.95324, 63.097416, 63.252914, 63.393303, 63.513447, 63.59833, 63.646427, 63.662975, 63.655685, 63.6331, 63.604057, 63.57682, 63.55762, 63.549824]

    # Use the advise_action method to get advised actions and the resulting assets
    resulting_assets, advised_actions = advise_action(
        predictions=prediction,
        start_date=start_date,
        end_date=end_date,
        budget=initial_budget,
        holdings=initial_holdings,
        auxiliary=auxiliary  
    )

    print(f"Advised actions from {start_date} to {end_date}: {advised_actions}")
    print(f"Resulting assets after following advised actions: ${resulting_assets:.2f}")

    data = yf.download('AAPL', start=start_date, end=end_date)

    # Get the actual closing prices and trading days
    actual_prices = data['Close'].tolist()
    trading_days = data.index.tolist()
    
    # Trim actions on non-trading days
    trimmed_actions = trim_actions_on_non_trading_days(all_dates, advised_actions, trading_days)

    # Evaluate the actions
    resulting_asset, ideal_final_asset = evaluate_actions(
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        actions=trimmed_actions,
        trading_days=trading_days,
        actual_prices=actual_prices
    )
    print(f"Resulted: {resulting_asset}, Ideal:{ideal_final_asset}")

if __name__ == "__main__":
    main()
