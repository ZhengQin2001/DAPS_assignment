import pandas as pd
import matplotlib.pyplot as plt
import json

# Load JSON data into DataFrame
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Flatten the nested 'Data' dictionary
    for date, values in data.items():
        for key, value in values['Data'].items():
            values[key] = value

    # Create DataFrame
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)
    return df

# Visualizer class
class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_and_save_features(self, features):
        for feature in features:
            plt.figure(figsize=(10, 6))
            plt.plot(pd.to_datetime(self.data['date']), self.data[feature], marker='o', linestyle='-')
            plt.title(feature)
            plt.xlabel('Date')
            plt.ylabel(feature)
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'plots/{feature}.png')  # Save each plot as a PNG file
            plt.close()

# Usage
file_path = 'src/data_preprocessing/filled_combined_data.json'  # Replace with your file path
combined_data = load_data_from_json(file_path)


visualizer = DataVisualizer(combined_data)
features_to_plot = ['StockNewsSentimentScore', 'Temperature', 'cashFlowToDebtRatio', "inflationRate", 'GDP']  
visualizer.plot_and_save_features(features_to_plot)
