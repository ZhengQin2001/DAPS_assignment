import yfinance as yf

from datetime import datetime
import datetime as dt
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

start_d = dt.date(2019, 4, 1)
end_d = dt.date(2023, 3, 31)

# The stock price base model object
class StockPrice(BaseModel):
    _id: Optional[str] = None
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int

    def to_json(self):
        return jsonable_encoder(self, exclude_none=True)

    def to_bson(self):
        data = self.dict(by_alias=True, exclude_none=True)
        if data.get("_id") is None:
            data.pop("_id", None)
        return data


class acquire_prices():
    """
    Task 1.1: Accquire historical stock prices spanning from a given period
    """
    def __init__(self, ticker: str, start_d: dt.date, end_d: dt.date) -> None:
        self.ticker = ticker
        self.start_d = start_d
        self.end_d = end_d
        self.df = self.acquire_daily_index() 
    
    def acquire_daily_index(self) -> pd.DataFrame:
        df = yf.Ticker(self.ticker).history(start=start_d, end=self.end_d, interval="1d")
        trimed_df = df.loc[:, :'Volume']
        self.df = trimed_df
        return trimed_df

    def plot_close_prices(self) -> (plt.Figure, plt.Axes): 
        close_p = self.df.loc[:, 'Close']
        title_p = "Close Price of " + self.ticker
        fig, ax = self._plot_prices(close_p, title_p, "Time", "Price (USD)")
        file_name = "plots/" + title_p + ".png"
        plt.savefig(file_name)
        # plt.show()
    
    def plot_volume(self) -> (plt.Figure, plt.Axes): 
        vol = self.df.loc[:, 'Volume']
        title_p = "Stock Volume of " + self.ticker
        fig, ax = self._plot_prices(vol, title_p, "Time", "Volume")
        file_name = "plots/" + title_p + ".png"
        plt.savefig(file_name)
        # plt.show()

    def plot_all_stats(self) -> (plt.Figure, plt.Axes): 
        low = self.df.loc[:, 'Low']
        high = self.df.loc[:, 'High']
        close_p = self.df.loc[:, 'Close']
        title_p = "Stock Prices of " + self.ticker

        fig, ax = self._plot_prices(close_p, title_p, "Time", "Volume")
        ax.fill_between(self.df.index, low, high, alpha=0.4)
        file_name = "plots/" + title_p + ".png"
        ax.legend(["Close price", "Stock price"])
        plt.savefig(file_name)
        plt.show()
        
    def _plot_prices(self, price: ArrayLike, title: str, x_label: str, y_label: str) -> (plt.Figure, plt.Axes):
        
        fig, ax = plt.subplots(1, figsize=(30, 5.5))

        # add the data to the plot
        ax.plot(price)

        # remove whitespace before and after
        ax.margins(x=0)

        # format the axes
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_locator(plt.MaxNLocator(30))  # Set maximum number of ticks to 12 (one per month)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.subplots_adjust(bottom=0.2)
        return fig, ax       

    
# create_plot = acquire_prices("AAPL", "2019-04-01", "2023-03-31")
# create_plot.plot_all_stats()