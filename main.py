
# main.py
# Entry point for the property listing application.
# Provides a simple CLI for user login and sign up.

import json
import hashlib
import os

# Path to the users JSON file
USERS_FILE = './datasets/users.json'  # Change this path if your file is elsewhere

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

def login(user_id, password, users):
    """
    Authenticate a user by user_id and password.
    Returns a success or error message.
    """
    hashed_input = hashlib.sha256(password.encode()).hexdigest()
    for user in users:
        if user["user_id"] == user_id:
            if user["password"] == hashed_input:
                print(f"✅ Login successful! Welcome {user['name']}.")
                login_menu(user_id, users)
            else:
                print("❌ Incorrect password.")
            return
    print("❌ User ID not found.")
        

def login_menu(user_id, users):
    print("\n" + "="*30)
    print("      USER DASHBOARD")
    print("="*30)
    print("1. View User Profile")
    print("2. View Property Listings")
    print("3. Logout")
    print("="*30)
    choice = input("Enter your choice: ")
    if choice == '1':
        manage_user_profile(user_id, users)
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
        view_user_profile(user_id, users)
        manage_user_profile(user_id, users)
    elif choice == '2':
        edit_user_profile(user_id, users)
        manage_user_profile(user_id, users)
    elif choice == '3':
        delete_user_profile(user_id, users)
        main_menu()
    elif choice == '4':
        login_menu(user_id, users)
    else:
        print("Invalid choice.")
        manage_user_profile(user_id, users)

def view_user_profile(user_id, users):
    """
    Display the user's profile information.
    """
    for user in users:
        if user["user_id"] == user_id:
            print("\n" + "*"*30)
            print("      USER PROFILE")
            print("*"*30)
            print(f"User ID:              {user['user_id']}")
            print(f"Name:                 {user['name']}")
            print(f"Group Size:           {user['group_size']}")
            print(f"Preferred Environment:{user['preferred_environment']}")
            print(f"Budget:               {user['budget']}")
            print("*"*30 + "\n")
            return

def edit_user_profile(user_id, users):
    """
    Edit the user's profile information.
    """
    for user in users:
        if user["user_id"] == user_id:
            print("\n" + "-"*30)
            print("      EDIT PROFILE")
            print("-"*30)
            name = input(f"Name ({user['name']}): ") or user['name']
            group_size = input(f"Group Size ({user['group_size']}): ") or user['group_size']
            pref_input = input(f"Preferred Environment(s) (comma-separated) ({user['preferred_environment']}): ")
            if pref_input:
                # Split by comma, strip whitespace, and filter out empty strings
                preferred_environment = [pref.strip() for pref in pref_input.split(',') if pref.strip()]
            else:
                preferred_environment = user['preferred_environment']
            budget = input(f"Budget ({user['budget']}): ") or user['budget']
            user.update({
                "name": name,
                "group_size": group_size,
                "preferred_environment": preferred_environment,
                "budget": budget
            })
            # Write the updated users list back to the JSON file
            with open(USERS_FILE, 'w') as f:
                json.dump(users, f, indent=4)
            print("\nProfile updated successfully.\n")
            return

def delete_user_profile(user_id, users):
    """
    Delete the user's profile.
    """
    for user in users:
        if user["user_id"] == user_id:
            users.remove(user)
            # Write the updated users list back to the JSON file
            with open(USERS_FILE, 'w') as f:
                json.dump(users, f, indent=4)
            print("\nProfile deleted successfully.\n")
            return

def main_menu():
    """
    Display the main menu and handle user input for login or sign up.
    """
    print("\n" + "="*30)
    print("   PROPERTY LISTING APP")
    print("="*30)
    print("1. Login")
    print("2. Sign Up")
    print("="*30)
    choice = input("Enter your choice: ")
    if choice == '1':
        users = load_users(USERS_FILE)
        entered_user_id = input("Enter User ID: ")
        entered_password = input("Enter Password: ")
        login(entered_user_id, entered_password, users)
    elif choice == '2':
        print("You selected Sign Up.")
        # Call sign up function here
    else:
        print("Invalid choice. Please try again.")
        main_menu()


if __name__ == "__main__":
    main_menu()
    







