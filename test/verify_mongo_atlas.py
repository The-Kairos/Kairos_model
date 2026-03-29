import os
from pymongo import MongoClient
from bson import ObjectId
from src.storage_utils import StorageManager

def verify_real_mongo():
    mongo_uri = "mongodb+srv://tehreemmasroor_db_user:UONIXKjScAsYl40z@kairos.szzedkh.mongodb.net/"
    client = MongoClient(mongo_uri)
    db = client.get_database("kairos")
    
    print("--- Verifying MongoDB Connection ---")
    try:
        client.admin.command('ping')
        print("Connected successfully to Atlas.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 1. Try to find an existing chat
    chat = db.chats.find_one()
    if chat:
        chat_id = str(chat['_id'])
        print(f"Found existing chat: {chat_id}")
    else:
        # 2. Create a test chat if none exists
        result = db.chats.insert_one({
            "title": "Test Chat for VM Migration",
            "pipeline": {"state": "initializing"},
            "createdAt": "2024-03-28T00:00:00.000Z"
        })
        chat_id = str(result.inserted_id)
        print(f"Created new test chat: {chat_id}")

    # 3. Test StorageManager with this ID
    print(f"\n--- Testing StorageManager with ID: {chat_id} ---")
    sm = StorageManager(chat_id=chat_id, mongo_uri=mongo_uri)
    
    print("Updating state to 'testing'...")
    sm.update_pipeline_state(stage="verification", percent=50, status="testing")
    
    # Verify the update
    updated_chat = db.chats.find_one({"_id": ObjectId(chat_id)})
    print(f"Verification result: {updated_chat.get('pipeline', {})}")
    
    if updated_chat.get('pipeline', {}).get('state') == 'testing':
        print("\nSUCCESS: StorageManager is correctly communicating with your Atlas MongoDB!")
    else:
        print("\nFAILURE: Update didn't seem to apply.")

if __name__ == "__main__":
    verify_real_mongo()
