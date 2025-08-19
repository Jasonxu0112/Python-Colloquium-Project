
# main.py
# Entry point for the property listing application.
# Provides a simple CLI for user login and sign up.

from recommenders.sbert_recommender import SbertRecommender
from models.users import User

import json
import hashlib
import os

BASE_DIR = os.path.dirname(__file__)

# Path to the users JSON file
USERS_FILE = os.path.abspath(
    os.path.join(BASE_DIR, "datasets", "users.json")
)  # Change this path if your file is elsewhere

# Path to the property listings JSON file (robust to script location)
PROPERTIES_FILE = os.path.abspath(
    os.path.join(BASE_DIR, "datasets", "property_listings.json")
)

_RECOMMENDER = None  # Placeholder for the recommender instance

def load_users(users_file):
    """
    Load users from a JSON file.
    Returns a list of user dictionaries.
    """
    if not os.path.exists(users_file):
        return []
    with open(users_file, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
        
def save_users(users, users_file):
    """
    Save users to a JSON file.
    user: user dictionaries.
    """
    with open(users_file, 'w') as f:
        json.dump([u for u in users], f, indent=4)
        
def ensure_user(user_id, users):
    """
    Ensure a user exists in the users list by user_id.
    """
    for u in users:
        if isinstance(u, User):
            if u.user_id == user_id:
                return u
        elif isinstance(u, dict):
            if u.get("user_id") == user_id:
                return User.from_dict(u)
    return None

def login(user, raw_password, users, max_attempts=3):
    """
    Handle user login.
    """
    while max_attempts > 0:
        # hashed_input = hashlib.sha256(password.encode()).hexdigest()
        
        if user.verify_password(raw_password):
            print(f"✅ Login successful! Welcome {user.name}.")
            login_menu(user, users)
            return
        else:
            max_attempts -= 1
            if max_attempts > 0:
                print(f"❌ Incorrect password. You have {max_attempts} attempts left.")
                raw_password = input("Please enter your password again: ")
            else:
                print("❌ Incorrect password.")
            return
    print("❌ User ID not found.")
        
def sign_up():
    """
    Handle user sign up.
    Prompts for user details and saves them to the users JSON file.
    """
    users = load_users(USERS_FILE)
    print("\n" + "="*30)
    print("      SIGN UP")
    print("="*30)
    
    user_id = input("Enter User ID: ").lower().strip()
    if ensure_user(user_id, users):
        print("❌ User ID already exists. Please try a different one.")
        return
    
    name = input("Enter Name: ")
    group_size = input("Enter Group Size: ")
    pref_input = input("Enter Preferred Environment(s) (comma-separated): ")
    preferred_environment = [pref.strip() for pref in pref_input.split(',') if pref.strip()]
    budget = input("Enter Budget: ")
    
    password = input("Create Password: ").strip()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    new_user = {
        "user_id": user_id,
        "name": name,
        "group_size": group_size,
        "preferred_environment": preferred_environment,
        "budget": budget,
        "password_hash": hashed_password
    }
    
    users.append(new_user)
    
    # Write the updated users list back to the JSON file
    save_users(users, USERS_FILE)
    
    print(f"✅ Sign up successful! Welcome {name}. You can now log in.")

def login_menu(user, users):
    print("\n" + "="*30)
    print("      USER DASHBOARD")
    print("="*30)
    print("1. View User Profile")
    print("2. View Property Listings")
    print("3. Logout")
    print("="*30)
    choice = input("Enter your choice: ")
    if choice == '1':
        manage_user_profile(user, users)
    elif choice == '2':
        print(view_property_listings())
    elif choice == '3':
        main_menu()
    else:
        print("Invalid choice. Please try again.")
        login_menu(user_id, users)

def manage_user_profile(user_id, users):
    print("\n" + "-"*30)
    print("   MANAGE USER PROFILE")
    print("-"*30)
    print("1. View Profile")
    print("2. Edit Profile")
    print("3. Delete Profile")
    print("4. Back to Main Menu")
    print("-"*30)
    choice = input("Enter your choice: ")
    if choice == '1':
        view_user_profile(user, users)
        manage_user_profile(user, users)
    elif choice == '2':
        edit_user_profile(user, users)
        manage_user_profile(user, users)
    elif choice == '3':
        delete_user_profile(user, users)
        main_menu()
    elif choice == '4':
        login_menu(user, users)
    else:
        print("Invalid choice.")
        manage_user_profile(user, users)

def view_user_profile(user, users):
    """
    Display the user's profile information.
    """
    print("\n" + "*"*30)
    print("      USER PROFILE")
    print("*"*30)
    print(f"User ID:              {user.user_id}")
    print(f"Name:                 {user.name}")
    print(f"Group Size:           {user.group_size}")
    print(f"Preferred Environment:{', '.join(user.preferred_environment)}")
    print(f"Budget:               {user.budget}")
    print("*"*30 + "\n")

def edit_user_profile(user, users):
    """
    Edit the user's profile information.
    """
    print("\n" + "-"*30)
    print("      EDIT PROFILE")
    print("-"*30)
    name = input(f"Name ({user.name}): ") or user.name
    
    group_input = input(f"Group Size ({user.group_size}): ")
    group_size = group_input or user.group_size
    
    pref_input = input(f"Preferred Environment(s) (comma-separated) ({user.preferred_environment}): ")
    if pref_input:
        # Split by comma, strip whitespace, and filter out empty strings
        preferred_environment = [pref.strip() for pref in pref_input.split(',') if pref.strip()]
    else:
        preferred_environment = user.preferred_environment
        
    budget_input = input(f"Budget ({user.budget}): ")
    budget = budget_input or user.budget
    
    # Update the user object
    user.name = name
    user.group_size = group_size
    user.preferred_environment = preferred_environment
    user.budget = budget
    
    for i, u in enumerate(users):
        if u.get("user_id") == user.user_id:
            users[i] = user.to_dict()
            break

    # Write the updated users list back to the JSON file
    save_users(users, USERS_FILE)
    print("\nProfile updated successfully.\n")

def delete_user_profile(user, users):
    """
    Delete the user's profile.
    """
    confirm = input(f"⚠️ Are you sure you want to delete profile for {user.name}? (y/n): ").strip().lower()
    if confirm != "y":
        print("❎ Deletion cancelled.")
        return

    for u in users:
        if u.get("user_id") == user.user_id:
            users.remove(u)
            save_users(users, USERS_FILE)   
            print(f"\n✅ Profile for {user.name} deleted successfully.\n")
            return
    

def main_menu():
    """
    Display the main menu and handle user input for login or sign up.
    """
    while True:
        print("\n" + "="*30)
        print("   PROPERTY LISTING APP")
        print("="*30)
        print("1. Login")
        print("2. Sign Up")
        print("3. Exit")
        print("="*30)
        choice = input("Enter your choice: ")
        
        if choice == '1':
            users = load_users(USERS_FILE)
            entered_user_id = input("Enter User ID: ").lower().strip()
            user = ensure_user(entered_user_id, users)
            if not user:
                print("❌ User ID not found. Please sign up first!")
                continue
            entered_password = input("Enter Password: ").strip()
            login(user, entered_password, users)

        elif choice == '2':
            sign_up()
        elif choice == '3':
            print("Thank you for choosing our app! See you again!")
            break
        else:
            print("Invalid choice. Please try again.")



if __name__ == "__main__":
    main_menu()








