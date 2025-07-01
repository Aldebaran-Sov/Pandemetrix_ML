from pymongo import MongoClient

_mongo_client = None

def get_mongo_client():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            host='mongodb',
            port= 27017,
            username= 'mongo_user',
            password= 'mongo_password'
        )
    return _mongo_client

def get_db():
    return get_mongo_client()['api_db']