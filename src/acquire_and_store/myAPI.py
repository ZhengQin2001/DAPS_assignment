from pymongo.collection import Collection, ReturnDocument
import os

import flask
from flask import Flask, jsonify, request, url_for
import json
from flask_pymongo import PyMongo
from bson.json_util import dumps
import requests
from datetime import datetime, timedelta
from acquire_prices import StockPrice

from database_connect import connect_to_mangodb


app = Flask(__name__)

mongo_uri = "mongodb+srv://zceezqi:Harry018854@zhengqindaps.kvjpszr.mongodb.net/?retryWrites=true&w=majority"
app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app, mongo_uri)

client = connect_to_mangodb()
db = client["AAPLstock"]
aapl: Collection = db.AAPL


# Create (POST) an item in the collection
@app.route('/AAPL/post', methods=['POST'])
def create_stock():
    data = request.json
    date_string = data.get("Date")
    
    try:
        date_time = datetime.strptime(date_string, '%Y-%m-%d')

    except ValueError:
        return jsonify({'message': 'Invalid date format. Please provide the date in the format YYYY-MM-DD'}), 400
    
    # Check for duplicate entry by Date
    existing_stock = aapl.find_one({'Date': data['Date']})
    if existing_stock:
        return jsonify({'message': 'Stock data for this date already exists'}), 400

    # Convert the datetime object back to a string
    data['Date'] = date_time.strftime('%Y-%m-%d')

    result = aapl.insert_one(data)
    return jsonify({'message': 'Stock data created successfully', 'id': str(result.inserted_id)}), 201


# Read (GET) all items in the collection
@app.route('/AAPL', methods=['GET'])
def get_all_stocks():
    cursor = aapl.find().sort("_id")
    return {
        "AAPL": [StockPrice(**doc).to_json() for doc in cursor]
    }

@app.route("/AAPL/<string:date>", methods=["GET"])
def get_price_by_date(date):
    stock_data = aapl.find_one({"Date": str(date)})
    
    if stock_data:
        stock_dict = {
            "_id": stock_data["_id"],
            "Date": stock_data["Date"],
            "Open": stock_data["Open"],
            "High": stock_data["High"],
            "Low": stock_data["Low"],
            "Close": stock_data["Close"],
            "Volume": stock_data["Volume"]
        }
        
        return StockPrice(**stock_dict).to_json()
    else:
        return jsonify({"message": "Stock data not found"}), 404


# Update (PUT) a specific item by Date
@app.route('/AAPL/update/<string:date>', methods=['PUT'])
def update_stock(date):
    data = request.json
    date_string = data.get("Date")
    
    try:
        date_time = datetime.strptime(date_string, '%Y-%m-%d')

    except ValueError:
        return jsonify({'message': 'Invalid date format. Please provide the date in the format YYYY-MM-DD'}), 400
    
    # Convert the datetime object back to a string
    data['Date'] = date_time.strftime('%Y-%m-%d')

    result = aapl.find_one_and_update(
        {'Date': data['Date']},
        {'$set': data},
        return_document=ReturnDocument.AFTER
    )

    if result:
        return jsonify({'message': 'Stock data updated successfully', 'stock': dumps(result)}), 200
    else:
        return jsonify({'message': 'Stock data not found'}), 404


# Delete (DELETE) a specific item by Date
@app.route('/AAPL/delete/<string:date>', methods=['DELETE'])
def delete_stock(date):
    try:
        date_time = datetime.strptime(date, '%Y-%m-%d')

    except ValueError:
        return jsonify({'message': 'Invalid date format. Please provide the date in the format YYYY-MM-DD'}), 400
    
    # Convert the datetime object back to a string
    date_string = date_time.strftime('%Y-%m-%d')

    deleted_stock = aapl.find_one_and_delete({'Date': date_string})
    if deleted_stock:
        return jsonify({'message': 'Stock data deleted successfully', 'stock': dumps(deleted_stock)}), 200
    else:
        return jsonify({'message': 'Stock data not found'}), 404
    

def get_collection(collection_name):
    return db[collection_name]

@app.route('/<collection_name>/post', methods=['POST'])
def create_document(collection_name):
    collection = get_collection(collection_name)
    document = request.json
    result = collection.insert_one(document)
    return jsonify({'message': 'Document added successfully', 'id': str(result.inserted_id)}), 201

@app.route('/<collection_name>', methods=['GET'])
def get_all_documents(collection_name):
    collection = get_collection(collection_name)
    documents = list(collection.find({}, {'_id': 0}))
    return jsonify(documents)

def get_query_for_dates(date_str: str, collection_name: str):
    try:
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        # If the date format is incorrect
        return jsonify({'message': 'Invalid date format. Use YYYY-MM-DD format.'}), False

    # Calculate the end of the day
    end_date = start_date + timedelta(days=1)

    # Query to find documents within the date range
    if collection_name == "Stock_News":
        query = {'publishedDate': {'$gte': start_date, '$lt': end_date}}
    elif collection_name == "Temperature":
        query = {'Date': date_str}
    else:
        query =  {'date': date_str}

    return query, True

@app.route('/<collection_name>/<date_str>', methods=['GET'])
def get_document_by_date(collection_name, date_str):
    collection = get_collection(collection_name)

    query, valid = get_query_for_dates(date_str, collection_name)
    if not valid:
        return jsonify({'message': query['error']}), 400  # Return the error message

    document_count = collection.count_documents(query)
    if document_count > 0:
        documents = collection.find(query, {'_id': 0})
        return jsonify([doc for doc in documents])
    else:
        return jsonify({'message': 'No documents found for this date'}), 404


@app.route('/<collection_name>/update/<date_str>', methods=['PUT'])
def update_document_by_date(collection_name, date_str):
    collection = get_collection(collection_name)

    query, valid = get_query_for_dates(date_str, collection_name)
    print(query)
    if not valid:
        # If the query is not valid, return an error
        return jsonify({'message': 'Invalid date format. Use YYYY-MM-DD format.'}), 400

    updated_data = request.json

    # Use only the query part in update_many
    result = collection.update_many(query, {'$set': updated_data})

    if result.modified_count > 0:
        return jsonify({'message': 'Documents updated successfully', 'modified_count': result.modified_count}), 200
    else:
        return jsonify({'message': 'No documents found for this date to update'}), 404
        

@app.route('/<collection_name>/delete/<date_str>', methods=['DELETE'])
def delete_document_by_date(collection_name, date_str):
    collection = get_collection(collection_name)

    query, valid = get_query_for_dates(date_str, collection_name)
    if not valid:
        # If the query is not valid, return an error
        return jsonify({'message': 'Invalid date format. Use YYYY-MM-DD format.'}), 400

    # Ensure that 'query' is a dictionary
    if not isinstance(query, dict):
        return jsonify({'message': 'Invalid query generated'}), 400

    # Delete all documents that match the query
    result = collection.delete_many(query)

    if result.deleted_count > 0:
        return jsonify({'message': 'Documents deleted successfully', 'deleted_count': result.deleted_count}), 200
    else:
        return jsonify({'message': 'No documents found for this date to delete'}), 404


if __name__ == '__main__':
    app.run(debug=True)
    