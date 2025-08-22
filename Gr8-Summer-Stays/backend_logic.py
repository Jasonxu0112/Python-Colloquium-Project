import json
import os
from datetime import datetime

USERS_FILE = os.path.join('..', 'datasets', 'users.json')
PROPERTIES_FILE = os.path.join('..', 'datasets', 'property_listings.json')

def load_users():
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def load_properties():
    with open(PROPERTIES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)["properties"]

def authenticate(user_id, password):
    users = load_users()
    for user in users:
        if user["user_id"] == user_id and user["password"] == password:
            return user
    return None

def add_user(user):
    users = load_users()
    users.append(user)
    save_users(users)

def save_property_for_user(user_id, property_id):
    users = load_users()
    for user in users:
        if user["user_id"] == user_id:
            if "saved_property" not in user:
                user["saved_property"] = []
            if property_id not in user["saved_property"]:
                user["saved_property"].append(property_id)
    save_users(users)

def get_saved_properties(user_id):
    users = load_users()
    properties = load_properties()
    for user in users:
        if user["user_id"] == user_id:
            saved_ids = user.get("saved_property", [])
            return [p for p in properties if p["property_id"] in saved_ids]
    return []
