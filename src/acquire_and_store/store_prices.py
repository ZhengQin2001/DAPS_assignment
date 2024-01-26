import acquire_prices as ap
import json

import pymongo
import pandas as pd

from database_connect import connect_to_mangodb

class store_prices():
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.client = connect_to_mangodb()
    
    def store_df(self):
        db = self.client.AAPLstock
        aapl = db.AAPL
        stock_price = self.df.to_json(orient='index', date_format='iso')
        # Parse the JSON string back into a Python object
        stock_price_data = json.loads(stock_price)

        formatted_data = []
        for date, values in stock_price_data.items():
        # Create a new dictionary with the date and merge it with the stock values
            stock_entry = {'Date': date[0:10]}
            stock_entry.update(values)
            formatted_data.append(stock_entry)

        # Delete the original data in the collection
        aapl.delete_many({})

        # Insert the data into MongoDB
        try:
            aapl.insert_many(formatted_data)
            print("Successfully Uploaded")
        except pymongo.errors as e:
            print("Error: ", e)
        


if __name__ == "__main__" :
    start_d = ap.start_d
    end_d = ap.end_d
    apple = ap.acquire_prices("AAPL", start_d, end_d)
    sp = store_prices(apple.df)
    sp.store_df()