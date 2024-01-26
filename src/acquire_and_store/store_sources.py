import json

import pymongo
import pandas as pd

from database_connect import connect_to_mangodb
from acquire_sources import acquire_sources

class store_sources():
    def __init__(self) -> None:
        self.start_d = "2019-04-01"
        self.end_d = "2023-03-31"
        self.latitude = 37.3875
        self.longitude = 122.0575
        self.stock_news = acquire_sources().acquire_stock_news("AAPL")
        self.full_fin_stat = acquire_sources().acquire_full_financial_stat("AAPL")
        self.ratio = acquire_sources().acquire_ratio("AAPL")
        self.sentiments = acquire_sources().acquire_sentiments("AAPL")
        self.econ_indicators = acquire_sources().acquire_econ_indicators(self.start_d, self.end_d)
        self.temperature = acquire_sources().acquire_temperature(self.latitude, self.longitude, self.start_d, self.end_d)
        self.client = connect_to_mangodb()
    
    def store_all_df(self):
        db = self.client.AAPLstock    
        
        collection_names = ["Stock_News", "Full_Financial_Stats", "Ratio", "Sentiments", "Econ_Indicators", "Temperature"]
        dfs = [self.stock_news, self.full_fin_stat, self.ratio, self.sentiments, self.econ_indicators, self.temperature]

        for collection_name, df in zip(collection_names, dfs):
            if collection_name in db.list_collection_names():
                print(f"Collection '{collection_name}' already exists. Handle the situation as needed.")
            else:
                #   create new collections for each type
                collection = db[collection_name]
                self._insert_data_to_MDB(collection, df)
    
        print("Data upload completed.")


    def _insert_data_to_MDB(self, collection, df):
        try:
            # Delete existing documents
            collection.delete_many({})

            # Insert the fresh data
            collection.insert_many(df)
        except pymongo.errors.PyMongoError as e:
            print("Error: ", e)

if __name__ == "__main__":
    sc = store_sources()
    sc.store_all_df()




