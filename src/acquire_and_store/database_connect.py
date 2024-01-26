# import pyodbc
# from pyodbc import Error
# import mysql.connector
# from mysql.connector import Error
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

server_name = "tcp:zhengqindaps.database.windows.net,1433"
database_name = 'zhengqin'
uid = 'dapszhengqin'
pwd = 'Harry018854**'

# dataapi = "zhengqindaps"
# dataapikey = "QXIvMq55GYQIuWixnoQlDS08Mc2rVqfzejXqFNZ89dn0DthbaliKrz5tbQiEi62I"

def connect_to_mangodb() -> pymongo.MongoClient:
    uri = "mongodb+srv://zceezqi:Harry018854@zhengqindaps.kvjpszr.mongodb.net/?retryWrites=true&w=majority"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
        
    return client
