import requests
import subprocess
import time
import json
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import pandas_ta as ta

class preprocessing:
    def __init__(self):
        pass
    
    def start_api_server(self):
        print("Starting API server...")
        # Start the API server.
        subprocess.Popen(["python", "src/acquire_and_store/myAPI.py"])

        # Wait for a short period to ensure the API server has time to start up
        time.sleep(8)

    def _extract_keys_from_file(self, file_path):
        keys = []

        with open(file_path, 'r') as file:
            for line in file:
                # Split the line at the colon
                parts = line.split(':')
                if parts and len(parts) > 1:
                    # The first part before the colon is the key
                    key = parts[0].strip()
                    keys.append(key)
    
        return keys

    def fetch_full_data_from_api(self, collection_str):
        self.start_api_server()
        response = requests.get(f'http://localhost:5000/{collection_str}')
        data = response.json()
        return data

    def _combine_data(self, fs_data, ratio_data, fs_keys, ratio_keys):

        combined_data = {entry['date']: {} for entry in fs_data + ratio_data}

        # Populate the dictionary with data from fs_data
        for entry in fs_data:
            date = entry['date']
            for key in fs_keys:
                combined_data[date][key] = entry.get(key, None)

        # Populate the dictionary with data from ratio_data
        for entry in ratio_data:
            date = entry['date']
            for key in ratio_keys:
                # Update the dictionary only if the key does not already exist
                if key not in combined_data[date]:
                    combined_data[date][key] = entry.get(key, None)

        return combined_data

    def processed_ratio(self):
        fs_file_path = 'src/data_preprocessing/financial_stats_keys.txt'
        ratio_file_path = 'src/data_preprocessing/ratio_keys.txt'

        fs_keys = self._extract_keys_from_file(fs_file_path)
        ratio_keys = self._extract_keys_from_file(ratio_file_path)
        
        # Fetch data from two collections
        fs_data = self.fetch_full_data_from_api("Full_Financial_Stats")
        ratio_data = self.fetch_full_data_from_api("Ratio")

        # Combine data from both collections
        combined_data = self._combine_data(fs_data, ratio_data, fs_keys, ratio_keys)

        # Convert the combined data to a list of dictionaries
        processed_data = [{"date": date, **data} for date, data in combined_data.items()]

        # with open("src\data_preprocessing\processed_ratio.json", 'w', encoding='utf-8') as file:
        #     json.dump(processed_data, file, ensure_ascii=False, indent=4)

        return processed_data


class sentiments_scoring(preprocessing):
    def __init__(self):
        super().__init__()
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()

    def trim_dataset(self):
        dataset = self.fetch_full_data_from_api("Stock_News")
        trimmed_data = []

        for entry in dataset:
            # Extracting the publishedDate and converting it to the desired format
            published_date = datetime.strptime(entry['publishedDate'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

            # Creating a new dictionary with the desired fields
            new_entry = {
                'date': published_date,
                'title': entry['title'],
                'text': entry['text']
            }

            trimmed_data.append(new_entry)

        # print(json.dumps(trimmed_data, indent=4))
        return trimmed_data

    def _calculate_sentiment_score(self, title, text):
        title_sentiment = self.sia.polarity_scores(title)["compound"]
        text_sentiment = self.sia.polarity_scores(text)["compound"]
        return (title_sentiment + text_sentiment) / 2

    def analyse_news(self):
        trimmed_data = self.trim_dataset()
        date_sentiments = {}

        for entry in trimmed_data:
            sentiment_score = self._calculate_sentiment_score(entry['title'], entry['text'])
            date = entry['date']
            if date in date_sentiments:
                date_sentiments[date].append(sentiment_score)
            else:
                date_sentiments[date] = [sentiment_score]

        # Combine sentiments for each date
        sentiment_dataset = []
        for date, sentiments in date_sentiments.items():
            # Example: Using simple average, replace with your chosen method
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            sentiment_dataset.append({'date': date, 'StockNewsSentimentScore': avg_sentiment})

        # print(json.dumps(sentiment_dataset, indent=4))
        return sentiment_dataset
        

class DataCombiner(sentiments_scoring):
    def __init__(self):
        super().__init__()

    def fetch_stock_price(self):
        # Fetch stock price data
        stock_price_data = self.fetch_full_data_from_api("AAPL")

        # Process and structure the stock price data
        processed_stock_prices = {}
        for entry in stock_price_data["AAPL"]:
            processed_stock_prices[entry['Date']] = {
                'Open': entry['Open'],
                'High': entry['High'],
                'Low': entry['Low'],
                'Close': entry['Close'],
                'Volume': entry['Volume']
            }
        # print(json.dumps(processed_stock_prices, indent=4))
        return processed_stock_prices

    def fetch_additional_data(self, collection_name):
        additional_data = self.fetch_full_data_from_api(collection_name)
        processed_data = {}
        for entry in additional_data:
            if collection_name == "Temperature":
                # Normalizing the date format
                date = entry['Date'].split(' ')[0] if ' ' in entry['Date'] else entry['Date']
                processed_data[date] = {k: v for k, v in entry.items() if k != '_id' and k != 'Date'}
            else:
                date = entry['date'].split(' ')[0] if ' ' in entry['date'] else entry['date']
                processed_data[date] = {k: v for k, v in entry.items() if k != '_id' and k != 'date'}
                
        # with open(f"src\data_preprocessing\{collection_name}.json", 'w', encoding='utf-8') as file:
        #     json.dump(processed_data, file, ensure_ascii=False, indent=4)
        return processed_data

    def get_closest_previous_stock_price(self, target_date, stock_prices):
        # Convert string dates to datetime objects for comparison
        stock_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in stock_prices.keys()])
        target_date = datetime.strptime(target_date, '%Y-%m-%d')

        # Find the closest previous date with stock price data
        closest_date = max([date for date in stock_dates if date <= target_date], default=None)
        if closest_date:
            return stock_prices[closest_date.strftime('%Y-%m-%d')]
        return None

    def combine_all_data(self):
        # Fetch and process each dataset
        financial_stats = self.processed_ratio()
        sentiment_scores = self.analyse_news()
        stock_prices = self.fetch_stock_price()
        econ_indicators = self.fetch_additional_data("Econ_Indicators")
        sentiments = self.fetch_additional_data("Sentiments")
        temperature = self.fetch_additional_data("Temperature")

        # Create a dictionary for each date's data
        combined_data = {}

        for date, stock_data in stock_prices.items():
            # Initialize with stock data
            combined_data[date] = {**stock_data, 'Data': {}}

            sentiment_data = next((item for item in sentiment_scores if item['date'] == date), None)
            if sentiment_data:
                combined_data[date]['Data']['StockNewsSentimentScore'] = sentiment_data['StockNewsSentimentScore']

            combined_data[date]['Data'].update(econ_indicators.get(date, {}))
            combined_data[date]['Data'].update({k: v for k, v in sentiments.get(date, {}).items() if k != 'symbol'})
            combined_data[date]['Data'].update(temperature.get(date, {}))

        for fs_entry in financial_stats:
            fs_date = fs_entry['date']
            stock_data = stock_prices.get(fs_date) or self.get_closest_previous_stock_price(fs_date, stock_prices)
            if stock_data:
                combined_data[fs_date] = {**stock_data, 'Data': {k: v for k, v in fs_entry.items() if k != 'date'}}
                combined_data[fs_date]['Data'].update(econ_indicators.get(date, {}))
                combined_data[fs_date]['Data'].update({k: v for k, v in sentiments.get(date, {}).items() if k != 'symbol'})
                combined_data[fs_date]['Data'].update(temperature.get(date, {}))

        # Sort by date and convert to a list
        final_dataset = {date: combined_data[date] for date in sorted(combined_data)}

        self.insert_seven_indicators(final_dataset)

        # Save to JSON file
        with open("src/data_preprocessing/combined_data.json", 'w', encoding='utf-8') as file:
            json.dump(combined_data, file, ensure_ascii=False, indent=4, sort_keys=True)

    def fill_missing_features(self, data):
        # This method takes a while
        # Convert dates to datetime objects for easy manipulation
        sorted_dates = sorted([datetime.strptime(date, '%Y-%m-%d') for date in data.keys()])

        # Store all possible keys from the 'Data' section
        all_keys = set()
        for date in data:
            all_keys.update(data[date]['Data'].keys())

        # Function to find the closest date with available data for a given key
        def find_closest_data(target_date, key):
            min_distance = float('inf')
            closest_value = None
            for date in sorted_dates:
                if key in data[date.strftime('%Y-%m-%d')]['Data'] and data[date.strftime('%Y-%m-%d')]['Data'][key] is not None:
                    distance = abs((target_date - date).days)
                    if distance < min_distance:
                        min_distance = distance
                        closest_value = data[date.strftime('%Y-%m-%d')]['Data'][key]
            return closest_value

        # Fill missing data
        for date_str in data:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            for key in all_keys:
                if key not in data[date_str]['Data'] or data[date_str]['Data'][key] is None:
                    data[date_str]['Data'][key] = find_closest_data(date_obj, key)

        return data

    def insert_seven_indicators(self, data):
        # Preparing a DataFrame from the nested 'Data' dictionaries
        df = pd.DataFrame.from_dict({k: v for k, v in data.items() if k != 'Data'}, orient='index')
        df.index = pd.to_datetime(df.index)  # Ensure the index is in datetime format

        # Ensure columns needed for TA are in correct format
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        # Calculate technical indicators
        df['OBV']= ta.obv(df['Close'], df['Volume'])
        df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])

        adx = ta.adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx['ADX_14']
        aroon = ta.aroon(df['High'], df['Low'])
        df['Aroon_Oscillator'] = aroon['AROONOSC_14']
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Histogram'] = macd['MACDh_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['RSI'] = ta.rsi(df['Close'])
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stochastic_k'] = stoch['STOCHk_14_3_3']
        df['Stochastic_d'] = stoch['STOCHd_14_3_3']

         # Fill NaN values
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill

        # Loop through each date and update the 'Data' key
        for date in data.keys():
            indicators = {
                'OBV': df.loc[date, 'OBV'],
                'AD': df.loc[date, 'AD'],
                'ADX': df.loc[date, 'ADX'],
                'Aroon_Oscillator': df.loc[date, 'Aroon_Oscillator'],
                'MACD': df.loc[date, 'MACD'],
                'MACD_Histogram': df.loc[date, 'MACD_Histogram'],
                'MACD_Signal': df.loc[date, 'MACD_Signal'],
                'RSI': df.loc[date, 'RSI'],
                'Stochastic_k': df.loc[date, 'Stochastic_k'],
                'Stochastic_d': df.loc[date, 'Stochastic_d']
            }

            # Update the 'Data' key for each date
            data[date]['Data'].update(indicators)
        
        return data




# Usage
combiner = DataCombiner()
combiner.combine_all_data()
with open('src/data_preprocessing/combined_data.json', 'r', encoding='utf-8') as file:
    combined_data = json.load(file)

filled_data = combiner.fill_missing_features(combined_data)

# Saving the filled data back to JSON
with open('src/data_preprocessing/filled_combined_data.json', 'w', encoding='utf-8') as file:
    json.dump(filled_data, file, ensure_ascii=False, indent=4, sort_keys=True)
