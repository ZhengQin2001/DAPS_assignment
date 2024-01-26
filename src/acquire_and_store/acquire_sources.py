import pandas as pd

import requests
import json
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

class BaseClass(BaseModel):
    def to_json(self):
        return jsonable_encoder(self, exclude_none=True)

    def to_bson(self):
        data = self.dict(by_alias=True, exclude_none=True)
        if data.get("_id") is None:
            data.pop("_id", None)
        return data

class StockNews(BaseClass):
    _id: Optional[str] = None
    symbol: str
    publishedDate: str
    title: str
    image: str
    site: str
    text: str
    url: str
    
class FullFinStats(BaseClass):
    _id: Optional[str] = None
    


class acquire_sources():
    def _acquire_from_fmp(self, query):  
        """
        Send the get request to Financial Modelling Prep
        
        Args:
        query (str): The formatted url to request for data

        Returns:
            (ArrayLike): Data in json format
        """      
        response = requests.get(query)

        response.raise_for_status()

        print("Response status code:", response.status_code)
        # format the response as JSON and store it in a dictionary variable
        stat = response.json()

        return stat


    def acquire_stock_news(self, ticker: str):
        """
        Accquiring historical stock news for a given ticker or company name.
        
        Args:
        ticker (str): ticker of the stock, e.g. AAPL

        Returns:
            (ArrayLike): data in json format
        """
        apikey = "VFG3H573gG4RO3NbWA8bKKzKkdStc0YN"
        stock_news_query = f"""https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=10000&apikey={apikey}"""
        news = self._acquire_from_fmp(stock_news_query)
        for idx in range(0, len(news)):
            publishedDate = news[idx].get("publishedDate")
            if publishedDate[0:7] == "2023-03":
                break
        trimmed_news = news[idx-1:]
        return trimmed_news


    def acquire_full_financial_stat(self, ticker: str):
        """
        Accquiring full financial statements, including cash flows, income statements, balance sheets and etc.
        These data indicate the financial health of a company

        Args:
        ticker (str): ticker of the stock, e.g. AAPL

        Returns:
            (ArrayLike): data in json format
        """
        apikey = "VFG3H573gG4RO3NbWA8bKKzKkdStc0YN"
        financial_stat_query = f"""https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{ticker}?period=quarter&limit=30&apikey={apikey}"""
        stat = self._acquire_from_fmp(financial_stat_query)
        start_idx = 0
        end_idx = 0
        for i in range(0, len(stat)):
            date = stat[i].get("date")
            if date[0:7] == "2019-03":
                start_idx = i
            if date[0:7] == "2023-04":
                end_idx = i
        trimmed_stat = stat[end_idx:start_idx+1]
        return trimmed_stat

    def acquire_ratio(self, ticker: str):
        """
        Accquiring ratios, such as P/B ratio and the ROE. Assess a company's financial health and compare it to its competitors.
        
        Args:
        ticker (str): ticker of the stock, e.g. AAPL

        Returns:
            (ArrayLike): data in json format
        """
        apikey = "VFG3H573gG4RO3NbWA8bKKzKkdStc0YN"
        ratio_query = f"""https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=quarter&apikey={apikey}"""
        ratio = self._acquire_from_fmp(ratio_query)
        start_idx = 0
        end_idx = 0
        for i in range(0, len(ratio)):
            date = ratio[i].get("date")
            if date[0:7] == "2019-03":
                start_idx = i
            if date[0:7] == "2023-04":
                end_idx = i
        trimmed_ratio = ratio[end_idx:start_idx+1]
        return trimmed_ratio

    def acquire_sentiments(self, ticker: str):
        """
        Accquiring historical social sentiment data for a given ticker or company name.
        
        Args:
        ticker (str): ticker of the stock, e.g. AAPL

        Returns:
            (ArrayLike): data in json format
        """
        apikey = "VFG3H573gG4RO3NbWA8bKKzKkdStc0YN"
        sentiments = []
        start_date = "2023-03-31"
        page = 0
        start_append = False

        # Acquire data from API
        for page in range(0, 181):
            sent_query = f"""https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol={ticker}&page={page}&apikey={apikey}"""
            sentiment_in_page = self._acquire_from_fmp(sent_query)
            for dict in sentiment_in_page:
                date = dict.get("date")
                if date[0:10] == start_date:
                    start_append = True
                if start_append:
                    sentiments.append(dict)
        return sentiments

    def acquire_econ_indicators(self, from_date: str, to_date: str):
        """
        Accquiring historical economic data for a variety of economic indicators, including GDP, unemployment, consumer sentiment and inflation.
        
        Args:
        from_date (str): start date of the 

        Returns:
            (ArrayLike): data in json format
        """
        indicators = ["GDP", "inflationRate", "consumerSentiment", "unemploymentRate"]
        apikey = "VFG3H573gG4RO3NbWA8bKKzKkdStc0YN"
        combined_data = {}
        # Acquire data from API
        for indicator in indicators:
            ind_query = f"""https://financialmodelingprep.com/api/v4/economic?name={indicator}&from={from_date}&to={to_date}&apikey={apikey}"""
            ind_data = self._acquire_from_fmp(ind_query)
            self._update_combined_data(combined_data, ind_data, indicator)
        
        # Add None for missing values
        for date in combined_data:
            for indicator in indicators:
                if indicator not in combined_data[date]:
                    combined_data[date][indicator] = None
        
        # Sort the combined data by date in descending order
        sorted_combined_data = sorted(combined_data.items(), key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"), reverse=True)
        combined_list = [{'date': date, **data} for date, data in sorted_combined_data]

        return combined_list

    def _update_combined_data(self, combined_data: dict, data_list: list, indicator: str):
        """
        Sub function of acquire_econ_indicators()
        """
        for entry in data_list:
            date = entry['date']
            if date in combined_data:
                combined_data[date][indicator] = entry['value']
            else:
                combined_data[date] = {indicator: entry['value']}

    def acquire_temperature(self, latitude: float, longitude: float, start_date: str, end_date: str):
        """
        Acquires data from the Open-meteo API for a given location and time period.
        API endpoint: https://archive-api.open-meteo.com/v1/archive.

        Args:
            latitude (float): latitude of the location
            longitude (float): longitude of the location
            start_date (str): start date of the data in the format YYYY-MM-DD
            end_date (str): end date of the data in the format YYYY-MM-DD
            
        Returns:
            (dict): dictionary of data
        """
        url = "https://archive-api.open-meteo.com/v1/archive/"

        # pack the parameters into a dictionary
        params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "America/Los_Angeles"
        }

        # send the get request
        response = requests.get(url, params=params)
        response.raise_for_status()

        # parse the time series data
        weather = response.json()

        weather_data = weather['daily']
        date_time = weather_data['time']
        temp = weather_data['temperature_2m_mean']

        data_list = []
        for date, temperature in zip(date_time, temp):
            data_dict = {
                "Date": date,
                "Temperature": temperature
            }
            data_list.append(data_dict)

        return data_list

if __name__ == "__main__":
    acc = acquire_sources()
    print(acc.acquire_temperature(37.3875, 122.0575, "2019-04-01", "2023-03-31"))
    # print(acc.acquire_econ_indicators("2019-04-01", "2023-03-31"))
    # print(acc.acquire_stock_news("AAPL"))

        
