# --- Embedding DB Check and Creation ---
def ensure_embeddings_db():
    import os, sqlite3
    db_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vector embeddings', 'property_vector_db.sqlite'))
    print(f"[LOG] Checking for embeddings DB at: {db_file}")
    if not os.path.exists(db_file):
        print("[LOG] Embeddings DB not found. Creating embeddings...")
        run_create_embeddings()
        return
    # Check if table exists
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    print("[LOG] Checking for property_embeddings table in DB...")
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='property_embeddings'")
    exists = c.fetchone()
    conn.close()
    if not exists:
        print("[LOG] Embeddings table not found. Creating embeddings...")
        run_create_embeddings()
    else:
        print("[LOG] Embeddings DB and table found. Ready to use.")

# --- Run create_embeddings.py as a subprocess ---
def run_create_embeddings():
    import subprocess, sys
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vector embeddings', 'create_embeddings.py'))
    print(f"[LOG] Running embedding creation script: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print("[LOG] Embedding script stdout:")
    print(result.stdout)
    if result.returncode != 0:
        print("[LOG] Error creating embeddings:", result.stderr)
        print("[LOG] Embedding creation script failed, but continuing to main logic.")
    else:
        print("[LOG] Embedding creation completed successfully or skipped (table exists). Continuing to main logic.")


# main.py
# Unified launcher for CLI and UI (Streamlit) modes, using shared core.py logic.

import os
import sys
import hashlib
import core

def cli_login():
    users = core.load_users()
    user_id = input("Enter User ID: ")
    password = input("Enter Password: ")
    user = core.authenticate(user_id, password)
    if user:
        print(f"✅ Login successful! Welcome {user['name']}.")
        login_menu(user)
    else:
        print("❌ Invalid credentials.")
        main_menu()

def cli_sign_up():
    print("\n" + "="*30)
    print("      SIGN UP")
    print("="*30)
    user_id = input("Enter User ID: ")
    users = core.load_users()
    if any(user["user_id"] == user_id for user in users):
        print("❌ User ID already exists. Please try a different one.")
        return
    name = input("Enter Name: ")
    group_size = input("Enter Group Size: ")
    pref_input = input("Enter Preferred Environment(s) (comma-separated): ")
    preferred_environment = [pref.strip() for pref in pref_input.split(',') if pref.strip()]
    budget = input("Enter Budget: ")
    password = input("Create Password: ")
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    new_user = {
        "user_id": user_id,
        "name": name,
        "group_size": group_size,
        "preferred_environment": preferred_environment,
        "budget": budget,
        "password": hashed_password
    }
    core.add_user(new_user)
    print(f"✅ Sign up successful! Welcome {name}. You can now log in.")

def login_menu(user):
    while True:
        print("\n" + "="*30)
        print("      USER DASHBOARD")
        print("="*30)
        print("1. View User Profile")
        print("2. View Property Listings")
        print("3. View Saved Properties")
        print("4. Logout")
        print("="*30)
        choice = input("Enter your choice: ")
        if choice == '1':
            view_user_profile(user)
        elif choice == '2':
            property_listings_menu(user)
        elif choice == '3':
            show_saved_properties(user)
        elif choice == '4':
            print("Logging out...")
            break
        else:
            print("Invalid choice. Please try again.")

# --- Property Listings Menu ---
def property_listings_menu(user):
    ensure_embeddings_db()
    print("\n" + "-"*30)
    print("   PROPERTY LISTINGS")
    print("-"*30)
    print("1. Get recommended options according to your preferences")
    print("2. Chat with the AI travel agent to plan your vacation")
    print("-"*30)
    choice = input("Enter your choice: ")
    if choice == '1':
        recommended_properties = recommend_properties_by_preferences(user)
        show_properties_with_descriptions(recommended_properties, user)
        print("\nNow starting chat with the AI travel agent...\n")
        travel_agent_chat(user, recommended_properties)
    elif choice == '2':
        travel_agent_chat(user)
    else:
        print("Invalid choice.")
        property_listings_menu(user)

# --- Vector Search Recommendation Logic ---
def recommend_properties_by_preferences(user, top_k=3):
    import sys, importlib.util, os
    embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vector embeddings', 'create_embeddings.py'))
    spec = importlib.util.spec_from_file_location("create_embeddings", embeddings_path)
    create_embeddings = importlib.util.module_from_spec(spec)
    sys.modules["create_embeddings"] = create_embeddings
    spec.loader.exec_module(create_embeddings)
    # Compose a query from user preferences
    if isinstance(user["preferred_environment"], list):
        query = ' '.join(user["preferred_environment"])
    else:
        query = str(user["preferred_environment"])
    query += f" {user['group_size']} {user['budget']}"
    model = create_embeddings.get_embedding_model()
    db_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vector embeddings', 'property_vector_db.sqlite'))
    results = create_embeddings.search_property(query, db_file, model, top_k=top_k)
    return results

# --- Show Properties with Appealing Descriptions ---
def show_properties_with_descriptions(properties, user):
    print("\nRecommended Properties:")
    for prop in properties:
        description = generate_property_description(prop, user)
        print(f"\nProperty ID: {prop['property_id']} | Similarity: {prop['similarity']:.4f}")
        print(f"  Location: {prop['location']}")
        print(f"  Type: {prop['type']}")
        print(f"  Features: {prop['features']}")
        print(f"  Tags: {prop['tags']}")
        print(f"  Description: {description}")

# --- OpenRouter DeepSeek LLM API ---
def query_openrouter_deepseek_llm(prompt):
    import requests, os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # If dotenv is not installed, skip silently
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not API_KEY:
        return "[ERROR] OpenRouter API key not set. Please set the OPENROUTER_API_KEY environment variable or add it to a .env file."
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai/",
        "X-Title": "Python-Colloquium-Project"
    }
    payload = {
        "model": "mistralai/mistral-large",
        "messages": [
            {"role": "system", "content": "You are a helpful AI travel agent assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                return data["choices"][0]["message"]["content"]
            else:
                return "[ERROR] No response from DeepSeek LLM."
        else:
            return f"[ERROR] LLM API error: {response.status_code} {response.text}"
    except Exception as e:
        return f"[ERROR] LLM API exception: {e}"

# --- Generate Appealing Description using DeepSeek LLM ---
def generate_property_description(property_data, user):
    # Short, concise description (30-40 words max)
    features = ', '.join(property_data['features'][:4])
    tags = ', '.join(property_data['tags'][:3])
    desc = (
        f"A {property_data['type']} in {property_data['location']} with {features}. "
        f"Great for {tags}. "
        f"Enjoy comfort and adventure at ${property_data.get('price', 'your budget')} per night."
    )
    # Ensure description is about 30-40 words
    words = desc.split()
    if len(words) > 40:
        desc = ' '.join(words[:40]) + '...'
    return desc
    
    def generate_property_description(property_data, user):
        # Short, concise description (30-40 words max)
        features = ', '.join(property_data['features'][:4])
        tags = ', '.join(property_data['tags'][:3])
        desc = (
            f"A {property_data['type']} in {property_data['location']} with {features}. "
            f"Great for {tags}. "
            f"Enjoy comfort and adventure at ${property_data.get('price', 'your budget')} per night."
        )
        # Ensure description is about 30-40 words
        words = desc.split()
        if len(words) > 40:
            desc = ' '.join(words[:40]) + '...'
        return desc

# --- AI Travel Agent Chat ---
def travel_agent_chat(user, recommended_properties=None):
    print("\n--- Welcome to the AI Travel Agent! ---")
    print("Type 'exit' to end the chat.")
    chat_history = []
    # Prepare context string for the LLM
    if recommended_properties:
        context_str = "Here are 3 recommended properties for the user: "
        for idx, prop in enumerate(recommended_properties, 1):
            context_str += f"\nProperty #{idx}: ID: {prop['property_id']}, Location: {prop['location']}, Type: {prop['type']}, Features: {prop['features']}, Tags: {prop['tags']}"
    else:
        context_str = "No recommended properties yet."
    property_id_map = {prop['property_id']: prop for prop in recommended_properties} if recommended_properties else {}
    while True:
        user_input = input("\033[96mYou:\033[0m ")  # Cyan for user
        if user_input.lower() == 'exit':
            print("\033[92mAI: Have a great trip! Goodbye!\033[0m")  # Green for AI
            break
        # Check if user refers to a recommended property by ID
        found = False
        matched_prop = None
        if property_id_map:
            for pid, prop in property_id_map.items():
                if pid.lower() in user_input.lower():
                    found = True
                    matched_prop = prop
                    print(f"\033[92mAI: Here are the details for property ID {pid}:\033[0m")
                    print(f"  Location   : {prop['location']}")
                    print(f"  Type       : {prop['type']}")
                    print(f"  Features   : {prop['features']}")
                    print(f"  Tags       : {prop['tags']}")
                    print(f"  Similarity : {prop['similarity']:.4f}")
                    print(f"  Description: {generate_property_description(prop, user)}")
                    break
        # Rule-based weather answer if user asks about weather for a property
        if (("weather" in user_input.lower() or "climate" in user_input.lower()) and (found or recommended_properties)):
            if not matched_prop and recommended_properties:
                matched_prop = recommended_properties[0]
            if matched_prop:
                location = matched_prop['location'].lower()
                tags = matched_prop['tags'].lower() if isinstance(matched_prop['tags'], str) else str(matched_prop['tags']).lower()
                # Simple rule-based weather summary
                if 'mountain' in location or 'mountain' in tags:
                    weather = "Expect cool to cold temperatures, especially at night. Weather can change quickly in the mountains."
                elif 'desert' in location or 'desert' in tags:
                    weather = "Expect hot days and cool nights. Summers can be extremely hot."
                elif 'beach' in location or 'beach' in tags or 'ocean' in tags:
                    weather = "Generally mild and breezy, with pleasant temperatures."
                elif 'city' in tags:
                    weather = "Typical urban climate, varies by season."
                elif 'temperate' in location or 'temperate' in tags:
                    weather = "Mild temperatures, not too hot or cold."
                elif 'cold' in location or 'cold' in tags:
                    weather = "Cold climate, especially in winter. Snow is possible."
                else:
                    weather = "Weather is generally pleasant, but check the forecast for details."
                print(f"\033[92mAI: The weather at {matched_prop['location']} is: {weather}\033[0m")
                chat_history.append((user_input, weather))
                continue
        if found:
            continue
        # Otherwise, use LLM for general questions
        short_context = "Here are some recommended properties: "
        if recommended_properties:
            for idx, prop in enumerate(recommended_properties, 1):
                short_context += f"\nProperty #{idx}: {prop['type']} in {prop['location']} (ID: {prop['property_id']})"
        else:
            short_context = "No recommended properties yet."
        # Limit chat history to last 2 turns
        limited_history = chat_history[-2:] if len(chat_history) > 2 else chat_history
        history_str = '\n'.join([f"User: {u}\nAI: {a}" for u, a in limited_history])
        prompt = (
            f"You are an AI travel agent. The user profile is: {user}. "
            f"{short_context}\n"
            f"Chat history:\n{history_str}\n"
            f"User: {user_input}\nAI:"
        )
        response = query_openrouter_deepseek_llm(prompt)
        if not response or response.strip() == "":
            print("\033[92mAI: [No response from LLM. Please try again or check API status.]\033[0m")
        elif response.startswith("[ERROR]"):
            print(f"\033[91mAI: {response}\033[0m")  # Red for errors
        else:
            print(f"\033[92mAI: {response.strip()}\033[0m")
        chat_history.append((user_input, response.strip() if response else ""))

# --- Extract Keywords using LLM (DeepSeek) ---
def extract_keywords_with_llm(prompt):
    extraction_prompt = f"Extract the main keywords and preferences from this travel request: '{prompt}'. Return a comma-separated list."
    response = query_openrouter_deepseek_llm(extraction_prompt)
    return response.strip()

# --- Recommend Properties by Prompt ---
def recommend_properties_by_prompt(prompt, top_k=3):
    import sys, importlib.util, os
    embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vector embeddings', 'create_embeddings.py'))
    spec = importlib.util.spec_from_file_location("create_embeddings", embeddings_path)
    create_embeddings = importlib.util.module_from_spec(spec)
    sys.modules["create_embeddings"] = create_embeddings
    spec.loader.exec_module(create_embeddings)
    model = create_embeddings.get_embedding_model()
    db_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Vector embeddings', 'property_vector_db.sqlite'))
    results = create_embeddings.search_property(prompt, db_file, model, top_k=top_k)
    return results

# --- Weather Suitability Check (Rule-based) ---
def check_weather_suitability(properties):
    import datetime
    if not properties:
        return ""
    prop = properties[0]
    location = prop['location'].lower()
    tags = prop['tags'].lower() if isinstance(prop['tags'], str) else str(prop['tags']).lower()
    month = datetime.datetime.now().month
    if (('desert' in location or 'hot' in tags) and month in [6,7,8]):
        return "Note: This destination may be very hot in summer. Consider if this suits your comfort."
    if (('mountain' in location or 'cold' in tags) and month in [12,1,2]):
        return "Note: This destination may be very cold in winter. Pack accordingly!"
    return "Weather looks suitable for your trip!"

# --- Itinerary Preferences ---
def ask_itenary_preferences():
    print("AI: Would you like me to generate an itinerary for your trip?")
    resp = input("(yes/no): ")
    if resp.lower() == 'yes':
        days = input("How many days is your trip?: ")
        style = input("What kind of itinerary do you want? (energetic/relaxed/mixed): ")
        extras = input("Do you want recommendations for restaurants, activities, or anything else? (yes/no): ")
        print(f"AI: Great! I'll prepare a {style} itinerary for {days} days. I'll also include extra recommendations: {extras}.")
        print("(Itinerary generation coming soon!)")
    else:
        print("AI: No problem! Let me know if you need anything else.")

def view_user_profile(user):
    print("\n" + "*"*30)
    print("      USER PROFILE")
    print("*"*30)
    print(f"User ID:              {user['user_id']}")
    print(f"Name:                 {user['name']}")
    print(f"Group Size:           {user['group_size']}")
    print(f"Preferred Environment:{user['preferred_environment']}")
    print(f"Budget:               {user['budget']}")
    print("*"*30 + "\n")

def show_saved_properties(user):
    saved = core.get_saved_properties(user['user_id'])
    if not saved:
        print("No properties saved yet.")
    for prop in saved:
        print(f"{prop['type']} in {prop['location']} (ID: {prop['property_id']})")
        print(f"  Price per night: ${prop['price_per_night']}")
        print(f"  Features: {', '.join(prop['features'])}")
        print(f"  Tags: {', '.join(prop['tags'])}")
        print(f"  Booked Dates: {', '.join(prop.get('booked_dates', []))}")
        print(f"  Coordinates: {prop['coordinates']}")
        print()


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
    print("\n" + "="*30)
    print("   PROPERTY LISTING APP")
    print("="*30)
    print("1. Login")
    print("2. Sign Up")
    print("3. Exit")
    print("="*30)
    choice = input("Enter your choice: ")
    if choice == '1':
        cli_login()
    elif choice == '2':
        cli_sign_up()
        main_menu()
    elif choice == '3':
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice. Please try again.")
        main_menu()



def launcher():
    print("\n" + "="*40)
    print("Welcome to Gr8 Summer Stays!")
    print("="*40)
    print("1. Launch CLI")
    print("2. Launch UI (Streamlit)")
    print("3. Exit")
    print("="*40)
    choice = input("Enter your choice: ")
    if choice == '1':
        main_menu()
    elif choice == '2':
        print("Launching Streamlit UI...")
        venv_streamlit = os.path.join(os.path.dirname(sys.executable), 'streamlit')
        cmd = f'"{venv_streamlit}" run Gr8-Summer-Stays/app.py'
        os.system(cmd)
    elif choice == '3':
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice. Please try again.")
        launcher()

if __name__ == "__main__":
    launcher()








