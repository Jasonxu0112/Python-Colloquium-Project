import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import core as logic
import hashlib
import requests
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Gr8 Summer Stays", layout="wide")
st.title("Gr8 Summer Stays")

if "user" not in st.session_state:
    st.session_state.user = None

# --- Login/Signup ---
def login_form():
    st.subheader("Login")
    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = logic.authenticate(user_id, password)
        if user:
            st.session_state.user = user
            st.success(f"Welcome, {user['name']}!")
            st.rerun()
        else:
            st.error("Invalid credentials.")

def signup_form():
    st.subheader("Sign Up")
    user_id = st.text_input("User ID", key="signup_id")
    name = st.text_input("Name", key="signup_name")
    group_size = st.number_input("Group Size", min_value=1, max_value=20, key="signup_group")
    preferred_env = st.text_input("Preferred Environment(s) (comma-separated)", key="signup_env")
    budget = st.number_input("Budget", min_value=1, key="signup_budget")
    password = st.text_input("Password", type="password", key="signup_pass")
    if st.button("Create Account"):
        users = logic.load_users()
        if any(u["user_id"] == user_id for u in users):
            st.error("User ID already exists.")
        else:
            hashed = hashlib.sha256(password.encode()).hexdigest()
            user = {
                "user_id": user_id,
                "name": name,
                "group_size": group_size,
                "preferred_environment": [e.strip() for e in preferred_env.split(",") if e.strip()],
                "budget": budget,
                "password": hashed
            }
            logic.add_user(user)
            st.success("Account created! Please log in.")
            st.rerun()

def login_signup_page():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login_form()
    with tab2:
        signup_form()

# --- Dashboard ---
def dashboard():
    st.sidebar.title(f"Welcome, {st.session_state.user['name']}")
    page = st.sidebar.radio("Go to", [
        "Recommended Properties",
        "Saved Properties",
        "Profile",
        "AI Travel Agent Chat",
        "Logout"
    ])
    if page == "Recommended Properties":
        recommended_properties_page()
    elif page == "Saved Properties":
        saved_properties_page()
    elif page == "Profile":
        profile_page()
    elif page == "AI Travel Agent Chat":
        ai_travel_agent_chat_page()
    elif page == "Logout":
        st.session_state.user = None
        st.rerun()

# --- Recommended Properties ---
def recommended_properties_page():
    st.header("Recommended Properties")
    n = st.slider("How many top properties do you want to see?", 1, 20, 5)
    # For demo, just show top n properties (replace with your recommend logic)
    properties = logic.load_properties()[:n]
    df = pd.DataFrame(properties)
    for i, prop in enumerate(properties):
        with st.expander(f"{prop['type']} in {prop['location']} (ID: {prop['property_id']})"):
            st.write(f"**Price per night:** ${prop['price_per_night']}")
            st.write(f"**Features:** {', '.join(prop['features'])}")
            st.write(f"**Tags:** {', '.join(prop['tags'])}")
            st.write(f"**Booked Dates:** {', '.join(prop.get('booked_dates', []))}")
            coords = pd.DataFrame([prop['coordinates']]).rename(columns={"lat": "latitude", "lng": "longitude"})
            st.map(coords)
            if st.button(f"Save Property {prop['property_id']}", key=f"save_{i}"):
                logic.save_property_for_user(st.session_state.user['user_id'], prop['property_id'])
                st.success("Property saved!")

# --- Saved Properties ---
def saved_properties_page():
    st.header("Your Saved Properties")
    saved = logic.get_saved_properties(st.session_state.user['user_id'])
    if not saved:
        st.info("No properties saved yet.")
    for i, prop in enumerate(saved):
        with st.expander(f"{prop['type']} in {prop['location']} (ID: {prop['property_id']})"):
            st.write(f"**Price per night:** ${prop['price_per_night']}")
            st.write(f"**Features:** {', '.join(prop['features'])}")
            st.write(f"**Tags:** {', '.join(prop['tags'])}")
            st.write(f"**Booked Dates:** {', '.join(prop.get('booked_dates', []))}")
            coords = pd.DataFrame([prop['coordinates']]).rename(columns={"lat": "latitude", "lng": "longitude"})
            st.map(coords)

# --- Profile Page ---
# --- Profile Page ---
def profile_page():
    st.header("Profile")
    user = st.session_state.user
    with st.form("edit_profile_form"):
        name = st.text_input("Name", value=user["name"])
        group_size = st.number_input("Group Size", min_value=1, max_value=20, value=int(user["group_size"]))
        preferred_env = st.text_input("Preferred Environment(s) (comma-separated)", value=", ".join(user["preferred_environment"]))
        budget = st.number_input("Budget", min_value=1, value=int(user["budget"]))
        submitted = st.form_submit_button("Save Changes")
        if submitted:
            user["name"] = name
            user["group_size"] = group_size
            user["preferred_environment"] = [e.strip() for e in preferred_env.split(",") if e.strip()]
            user["budget"] = budget
            # Update in users.json
            users = logic.load_users()
            for u in users:
                if u["user_id"] == user["user_id"]:
                    u.update(user)
            logic.save_users(users)
            st.session_state.user = user
            st.success("Profile updated!")


# --- AI Travel Agent Chat Page ---
def ai_travel_agent_chat_page():
    st.header("AI Travel Agent Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user = st.session_state.user
    st.info("Ask the AI travel agent anything about your trip, preferences, or properties!")
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Prepare last 3 exchanges for context
        history = st.session_state.chat_history[-6:]  # 3 user+assistant pairs
        context_str = "\n".join([
            ("User: " + m["content"]) if m["role"] == "user" else ("AI: " + m["content"]) for m in history
        ])
        with st.spinner("AI is thinking..."):
            ai_response = query_openrouter_deepseek_llm(
                f"User profile: {user}.\nChat history:\n{context_str}\nUser: {user_input}"
            )
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        st.chat_message("assistant").write(ai_response)

# --- LLM Query Logic (copied from CLI) ---
def query_openrouter_deepseek_llm(prompt):
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
        "max_tokens": 1024
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

# --- Main App Logic ---
if st.session_state.user is None:
    login_signup_page()
else:
    dashboard()
