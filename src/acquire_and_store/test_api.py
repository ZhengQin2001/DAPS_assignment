import requests
from datetime import datetime, timedelta

def test_create_method():
    # Define the item data in JSON format
    new_stock_data = {
        "Date": "2023-10-01",
        "Open": 150.0,
        "High": 155.0,
        "Low": 148.0,
        "Close": 152.5,
        "Volume": 1000000
    }

    # Send a POST request to create the item
    response = requests.post("http://localhost:5000/AAPL/post", json=new_stock_data)

    # Check the response
    if response.status_code == 201:
        print("Stock item created successfully")
        print("Item ID:", response.json()["id"])
    else:
        print("Failed to create stock item")
        print("Response:", response.text)

def test_get_method():
    # Send a GET request to retrieve all stock items
    response = requests.get("http://localhost:5000/AAPL")

    # Check the response
    if response.status_code == 200:
        stock_items = response.json()["AAPL"]
        for item in stock_items:
            print("Stock Item:", item)
    else:
        print("Failed to retrieve stock items")
        print("Response:", response.text)

def test_get_date_method():
    # Specify the date of the item you want to retrieve
    date_to_retrieve = "2023-10-01"

    # Send a GET request to retrieve the item by date
    response = requests.get(f"http://localhost:5000/AAPL/{date_to_retrieve}")
    # Check the response
    if response.status_code == 200:
        stock_item = response.json()
        print("Stock Item:", stock_item)
    else:
        print("Failed to retrieve stock item")
        print("Response:", response.text)

def test_update_method():
    # Specify the date of the item you want to update
    date_to_update = "2023-10-01"

    # Define the updated stock data
    updated_stock_data = {
        "Date": date_to_update,
        "Open": 135.0,
        "High": 160.0,
        "Low": 152.0,
        "Close": 158.5,
        "Volume": 1200000
    }

    # Send a PUT request to update the item by date
    response = requests.put(f"http://localhost:5000/AAPL/update/{date_to_update}", json=updated_stock_data)

    # Check the response
    if response.status_code == 200:
        print("Stock Item updated successfully")
        print("Updated Item:", updated_stock_data)
    elif response.status_code == 404:
        print("Stock Item not found for update")
    else:
        print("Failed to update stock item")
        print("Response:", response.text)


def test_delete_method():
    # Specify the date of the item you want to delete
    date_to_delete = "2023-10-01"

    # Send a DELETE request to delete the item by date
    response = requests.delete(f"http://localhost:5000/AAPL/delete/{date_to_delete}")

    # Check the response
    if response.status_code == 200:
        deleted_stock_item = response.json()
        print("Stock Item deleted successfully")
        print("Deleted Item:", deleted_stock_item)
    else:
        print("Failed to delete stock item")
        print("Response:", response.text)


# Base URL of your Flask API
BASE_URL = "http://127.0.0.1:5000"

def create_document(collection_name, document):
    response = requests.post(f"{BASE_URL}/{collection_name}/post", json=document)
    return response.json()

def get_all_documents(collection_name):
    response = requests.get(f"{BASE_URL}/{collection_name}")
    return response.json()

def get_documents_by_date(collection_name, date_str):
    response = requests.get(f"{BASE_URL}/{collection_name}/{date_str}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return f"Failed to decode JSON. Status Code: {response.status_code}, Response Text: {response.text}"

def update_documents_by_date(collection_name, date_str, updated_data):
    response = requests.put(f"{BASE_URL}/{collection_name}/update/{date_str}", json=updated_data)
    return response.json()

def delete_documents_by_date(collection_name, date_str):
    response = requests.delete(f"{BASE_URL}/{collection_name}/delete/{date_str}")
    return response.json()

def test_nosql():
    # Test data
    collection_name = "Ratio"
    today_date_str = datetime.now().strftime("%Y-%m-%d")
    test_document = {"date": today_date_str, "operatingincomeloss": 36016000000}

    # # Testing the API
    print("Creating a document...")
    create_response = create_document(collection_name, test_document)
    print(create_response)

    # print("\nFetching all documents...")
    # all_docs = get_all_documents(collection_name)
    # print(all_docs)

    # print("\nFetching documents by date...")
    # docs_by_date = get_documents_by_date(collection_name, "2019-12-28")
    # print(docs_by_date)

    # print("\nUpdating documents by date...")
    # update_response = update_documents_by_date(collection_name, today_date_str, {"operatingincomeloss": "36"})
    # print(update_response)

    # print("\nDeleting documents by date...")
    # delete_response = delete_documents_by_date(collection_name, today_date_str)
    # print(delete_response)

    

if __name__ == "__main__":
    # test_get_method()
    # test_get_date_method()
    # test_create_method()
    # test_update_method()
    # test_delete_method()
    test_nosql()