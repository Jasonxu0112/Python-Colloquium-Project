# gui.py — Streamlit GUI with: Login/Signup, Filters (incl. availability range + sort),
# instant semantic recommendations from chat (uses existing SQLite embeddings),
# map view, and short LLM guidance.
# Usage: streamlit run gui.py

import time
import json
import sqlite3
import numpy as np
import requests
from pathlib import Path
import os
import traceback
import hashlib
import pandas as pd
import datetime as dt
import re
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(page_title="Gr8stayz", page_icon="🏡", layout="wide")

# ---------------- Safe mode ----------------
# Set SAFE_MODE=1 to skip embeddings/model on startup (UI still loads, but semantic search is disabled)
SAFE_MODE = os.environ.get("SAFE_MODE", "0") == "1"

# ---------------- OpenRouter (demo key — replace for your own) ----------------
# Security note: do not commit real keys to public repos. This is for local demo only.
OPENROUTER_API_KEY = "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"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
VEC_DIR = PROJECT_ROOT / "Vector embeddings"
DB_FILE = VEC_DIR / "property_vector_db.sqlite"
PROPERTIES_JSON = PROJECT_ROOT / "datasets" / "property_listings.json"
USERS_JSON = PROJECT_ROOT / "datasets" / "users.json"
CREATE_EMB_SCRIPT = VEC_DIR / "create_embeddings.py"

# ---------------- Utilities ----------------
def sanitize(text: str) -> str:
    """Remove asterisks so the UI shows no '*'."""
    if isinstance(text, str):
        return text.replace("*", "")
    return text

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------------- Users: load/save/verify/edit ----------------
def load_users_dict():
    """Load users from datasets/users.json in either list or object format."""
    if not USERS_JSON.exists():
        st.session_state["users_format"] = "list"
        return {}
    with open(USERS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    users_list, fmt = [], "list"
    if isinstance(data, list):
        users_list, fmt = data, "list"
    elif isinstance(data, dict) and isinstance(data.get("users"), list):
        users_list, fmt = data["users"], "object"

    st.session_state["users_format"] = fmt
    users_dict = {}
    for u in users_list:
        uid = u.get("user_id")
        if uid:
            users_dict[uid] = u
    return users_dict

def persist_users_dict(users_dict):
    """Persist users in the original format we loaded."""
    fmt = st.session_state.get("users_format", "list")
    users_list = list(users_dict.values())
    USERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        if fmt == "object":
            json.dump({"users": users_list}, f, ensure_ascii=False, indent=2)
        else:
            json.dump(users_list, f, ensure_ascii=False, indent=2)

def verify_login(users_dict, user_id, password) -> bool:
    """Validate password by sha256 comparison."""
    u = users_dict.get(user_id)
    if not u:
        return False
    return u.get("password") == sha256(password)

def signup_user(users_dict, new_user: dict):
    uid = new_user["user_id"]
    if uid in users_dict:
        return False, "User ID already exists."
    users_dict[uid] = new_user
    persist_users_dict(users_dict)
    return True, "Account created."

def update_user(users_dict, user_id, patch: dict):
    if user_id not in users_dict:
        return False, "User not found."
    users_dict[user_id].update(patch)
    persist_users_dict(users_dict)
    return True, "Updated."

# ---------------- Embeddings (existing vector DB) ----------------
def ensure_db():
    """Ensure the SQLite vector DB exists; build it with the project's script if missing."""
    def table_exists() -> bool:
        if not DB_FILE.exists():
            return False
        try:
            conn = sqlite3.connect(str(DB_FILE))
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='property_embeddings'")
            ok = c.fetchone() is not None
            conn.close()
            return ok
        except Exception:
            return False

    if not table_exists():
        import runpy
        with st.spinner("Building vector database from datasets/property_listings.json ..."):
            runpy.run_path(str(CREATE_EMB_SCRIPT))

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    with st.spinner("Loading embedding model (first time may take a minute) ..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# ---------------- Dataset loading ----------------
def load_properties():
    """Load properties and force property_id to string for consistent matching."""
    with open(PROPERTIES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    props = {}
    for p in data["properties"]:
        pid = str(p.get("property_id"))
        p["property_id"] = pid
        props[pid] = p
    return props

# ---------------- Availability helpers ----------------
def parse_date(s):
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None

def is_available(prop, date_range):
    """
    Availability supports:
      - available_from (ISO), available_to (ISO)
      - available_dates: [ISO...]
    The property is considered available only if it covers the entire selected date range.
    date_range must be a (start_date, end_date) tuple (inclusive).
    """
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        return False  # we enforce a date range selection in the UI

    start, end = date_range
    if not isinstance(start, dt.date) or not isinstance(end, dt.date):
        return False
    if end < start:
        return False

    af = parse_date(prop.get("available_from"))
    at = parse_date(prop.get("available_to"))
    dates = prop.get("available_dates")

    # Range window coverage
    if af and at:
        return (start >= af and start <= at) and (end >= af and end <= at)

    # Explicit dates list coverage
    if isinstance(dates, list) and dates:
        ds = {parse_date(x) for x in dates if parse_date(x)}
        cur = start
        while cur <= end:
            if cur not in ds:
                return False
            cur += dt.timedelta(days=1)
        return True

    # No availability info -> treat as not available for strict range filtering
    return False

# ---------------- Semantic candidates for chat (instant cards) ----------------
def semantic_candidates_for_query(query_text: str, properties: dict, top_k: int = 12):
    """
    Use the existing SQLite vector DB to retrieve semantically closest properties
    to the user's chat text. Returns full property dicts ready for card rendering.
    """
    if not query_text or SAFE_MODE:
        return []

    # Encode query
    model = get_embedding_model()
    q = model.encode([query_text])[0]

    # Fetch embeddings
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute("SELECT property_id, embedding FROM property_embeddings")
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []

    prop_ids, embs = [], []
    for pid, emb_blob in rows:
        prop_ids.append(str(pid))
        embs.append(np.frombuffer(emb_blob, dtype=np.float32))
    embs = np.vstack(embs)

    # Cosine similarity
    denom = (np.linalg.norm(embs, axis=1) * (np.linalg.norm(q) + 1e-10)) + 1e-10
    sims = np.dot(embs, q) / denom
    order = np.argsort(sims)[::-1]

    # Map back to property dicts
    results = []
    for idx in order[:top_k]:
        pid = prop_ids[idx]
        p = properties.get(pid)
        if p:
            p_copy = dict(p)
            p_copy["_semantic_sim"] = float(sims[idx])
            results.append(p_copy)
    return results

# ---------------- Filters UI (date range enforced via two date inputs) ----------------
def render_filters(properties, profile):
    locations = sorted({(p.get("location", "") or "") for p in properties.values()})

    tag_pool = set()
    for p in properties.values():
        for t in p.get("tags", []):
            if isinstance(t, str) and t.strip():
                tag_pool.add(t.strip())
    env_choices = sorted(tag_pool)
    lower_to_choice = {c.lower(): c for c in env_choices}

    default_group_size = int(profile.get("group_size", 2))
    prices = [p.get("price_per_night", 0) for p in properties.values()]
    max_price = int(max(prices) if prices else 1000)
    default_budget = int(profile.get("budget", 300))
    default_budget = min(max(default_budget, 0), max_price)

    pref_from_profile = [e for e in profile.get("preferred_environment", []) if isinstance(e, str)]
    pref_default = []
    for e in pref_from_profile:
        key = e.strip().lower()
        if key in lower_to_choice:
            pref_default.append(lower_to_choice[key])

    with st.container(border=True):
        st.subheader("Filters")
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 2, 1.4, 2])
        with col1:
            location = st.selectbox("Location", ["Any"] + locations, index=0)
        with col2:
            group_size = st.number_input("Group size", min_value=1, step=1, value=default_group_size)
        with col3:
            budget = st.slider("Budget per night", 0, max_price, value=default_budget)
        with col4:
            pref_env = st.multiselect("Preferred environment", env_choices, default=pref_default)
        with col5:
            sort_choice = st.selectbox("Sort by price", ["Low to High", "High to Low"], index=0)
        with col6:
            # Two separate date inputs to avoid "Choose a date range" placeholder
            today = dt.date.today()
            start_date = st.date_input("Start date", value=today, format="YYYY-MM-DD")
            end_date = st.date_input("End date", value=today + dt.timedelta(days=2),
                                     min_value=start_date, format="YYYY-MM-DD")
            if end_date < start_date:
                st.error("End date must be on or after the start date.")
                avail_range = None
            else:
                avail_range = (start_date, end_date)

    return location, group_size, budget, pref_env, sort_choice, avail_range

# ---------------- Filtered recommendations (uses availability range strictly) ----------------
def filter_and_rank(properties, location, group_size, budget, pref_env, sort_choice, avail_range):
    sel_loc = (location or "").strip().lower()

    # Hard filter by location, price, and availability range (must be fully covered)
    filtered = []
    for p in properties.values():
        prop_loc = (p.get("location", "") or "").strip().lower()
        price = float(p.get("price_per_night") or 1e12)
        if sel_loc != "any" and prop_loc != sel_loc:
            continue
        if price > float(budget):
            continue
        if not is_available(p, avail_range):
            continue
        filtered.append(p)

    # Optional semantic boost for tiebreaking (if embeddings enabled)
    sim_map = {}
    if not SAFE_MODE:
        try:
            env_text = " ".join(pref_env) if pref_env else ""
            # Use a compact query describing constraints to guide semantic scoring
            query = f"{env_text} group {group_size} budget {budget}"
            # Reuse embedding table for tiebreaker
            model = get_embedding_model()
            q = model.encode([query])[0]

            conn = sqlite3.connect(str(DB_FILE))
            c = conn.cursor()
            c.execute("SELECT property_id, embedding FROM property_embeddings")
            rows = c.fetchall()
            conn.close()

            if rows:
                prop_ids, embs = [], []
                for pid, emb_blob in rows:
                    prop_ids.append(str(pid))
                    embs.append(np.frombuffer(emb_blob, dtype=np.float32))
                embs = np.vstack(embs)
                denom = (np.linalg.norm(embs, axis=1) * (np.linalg.norm(q) + 1e-10)) + 1e-10
                sims = np.dot(embs, q) / denom
                sim_map = {prop_ids[i]: float(sims[i]) for i in range(len(prop_ids))}
        except Exception:
            sim_map = {}

    def apply_sort(items, sim_map):
        if sort_choice == "Low to High":
            return sorted(items, key=lambda x: (float(x.get("price_per_night") or 1e-12),
                                                -sim_map.get(x["property_id"], 0.0)))
        else:
            return sorted(items, key=lambda x: (-float(x.get("price_per_night") or 0.0),
                                                -sim_map.get(x["property_id"], 0.0)))

    ranked = apply_sort(filtered, sim_map)
    return ranked[:12], sim_map

# ---------------- LLM (short guidance only; cards come from dataset instantly) ----------------
def llm_short_guidance(user_text, shown_properties):
    """
    Call OpenRouter to produce a short guidance paragraph.
    The UI already shows property cards; the assistant adds a concise summary or tips.
    """
    key = (OPENROUTER_API_KEY or "").strip()
    if not key:
        return "No OpenRouter API key configured in code."

    lines = []
    for p in shown_properties[:12]:
        lines.append(f"- {p.get('property_id')} | {p.get('type','')} | {p.get('location','')} | ${p.get('price_per_night','?')}")

    system = {
        "role": "system",
        "content": (
            "You are a concise travel assistant. The UI already shows property cards; "
            "your job is to add a short friendly guidance based on the user's ask and these displayed properties. "
            "Do not invent properties. Prefer to mention property_id when referring to any listing."
        ),
    }
    user = {"role": "user", "content": f"User query: {user_text}\nDisplayed properties:\n" + "\n".join(lines)}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Gr8stayz",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [system, user],
        "temperature": 0.6,
    }

    try:
        resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=60)
        raw_text = resp.text or ""
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            snippet = raw_text[:400].replace("\n", " ")
            return f"(Note) Model response is not JSON (status {status}). {snippet}"
        if status >= 400:
            if status == 401:
                msg = data.get("error", {}).get("message", "")
                return f"(Note) Authentication failed (401). {msg}"
            if status == 429:
                return "(Note) Model is at capacity (429). Please try again."
            return f"(Note) Model HTTP {status}: {data}"
        if "choices" in data:
            return sanitize(data["choices"][0]["message"]["content"])
        elif "output" in data:
            return sanitize(data["output"])
        return sanitize(str(data))
    except Exception as e:
        return f"(Note) Model request failed: {e}"

# ---------------- UI helpers ----------------
def property_card(prop, sim=None):
    title = sanitize(f"{prop.get('type','Home')} in {prop.get('location','')}")
    price = prop.get("price_per_night", None)
    tags = sanitize(", ".join(prop.get("tags", [])[:3]))

    # Availability preview text
    avail_txt = "unknown"
    if prop.get("available_from") or prop.get("available_to"):
        af, at = prop.get("available_from"), prop.get("available_to")
        avail_txt = f"{af or '?'} → {at or '?'}"
    elif isinstance(prop.get("available_dates"), list) and prop.get("available_dates"):
        avail_txt = f"{len(prop.get('available_dates'))} dates"

    st.markdown(
        f"""
        <div style="border:1px solid #eee;border-radius:16px;padding:12px;">
            <div style="font-weight:600;margin-bottom:4px;">{title}</div>
            <div style="opacity:.7;margin-bottom:6px;">ID: {sanitize(str(prop.get('property_id')))}</div>
            <div style="margin-bottom:6px;">Price per night: {price}</div>
            <div style="margin-bottom:6px;">Availability: {avail_txt}</div>
            <div style="opacity:.8;">{tags}</div>
            {f"<div style='opacity:.6;margin-top:6px;'>similarity={sim:.3f}</div>" if sim is not None else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

def show_map(items):
    rows = []
    for p in items:
        lat = p.get("latitude")
        lon = p.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            rows.append({"lat": float(lat), "lon": float(lon), "property_id": p.get("property_id")})
    if rows:
        df = pd.DataFrame(rows)
        st.map(df, latitude="lat", longitude="lon", size=80)
    else:
        st.caption("Map is hidden because listings do not include latitude or longitude.")

# ---------------- App ----------------
try:
    with st.spinner("Initializing …"):
        time.sleep(0.1)
        properties = load_properties()
        if not SAFE_MODE:
            ensure_db()

    # Build location set (useful in captions)
    all_locations = {p.get("location", "") for p in properties.values()}

    # Auth gate
    def login_view():
        st.markdown("<h1 style='margin:0;text-align:center'>Gr8stayz</h1>", unsafe_allow_html=True)
        st.markdown("### Sign in or create an account")
        tab_login, tab_signup = st.tabs(["Sign in", "Create account"])
        users = load_users_dict()
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                lid = st.text_input("User ID")
                lpw = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in")
            if submitted:
                if verify_login(users, lid.strip(), lpw):
                    st.session_state["auth_user"] = lid.strip()
                    st.success("Signed in.")
                    st.rerun()
                else:
                    st.error("Invalid user or password.")
        with tab_signup:
            with st.form("signup_form"):
                uid = st.text_input("User ID")
                name = st.text_input("Name")
                group_size = st.number_input("Group size", min_value=1, step=1, value=2)
                budget = st.number_input("Budget per night", min_value=0, step=10, value=300)
                pref_env_raw = st.text_input("Preferred environment (comma separated)", value="Beach, Mountain")
                pw = st.text_input("Password", type="password")
                pw2 = st.text_input("Confirm password", type="password")
                created = st.form_submit_button("Create account")
            if created:
                if not uid or not pw:
                    st.error("User ID and password are required.")
                elif pw != pw2:
                    st.error("Passwords do not match.")
                else:
                    new_user = {
                        "user_id": uid.strip(),
                        "name": name.strip(),
                        "group_size": int(group_size),
                        "budget": float(budget),
                        "preferred_environment": [e.strip() for e in pref_env_raw.split(",") if e.strip()],
                        "password": sha256(pw),
                    }
                    ok, msg = signup_user(users, new_user)
                    if ok:
                        st.success("Account created. You can sign in now.")
                    else:
                        st.error(msg)

    if "auth_user" not in st.session_state:
        login_view()
        st.stop()

    users_dict = load_users_dict()
    auth_user = st.session_state["auth_user"]
    profile = users_dict.get(auth_user, {})

    # Top bar
    def user_popover(users_dict, user_id):
        with st.popover("👤", use_container_width=False):
            st.write("User")
            st.caption(user_id)
            st.divider()
            u = users_dict.get(user_id, {})
            with st.form("profile_edit"):
                name = st.text_input("Name", value=u.get("name", ""))
                group_size = st.number_input("Group size", min_value=1, step=1, value=int(u.get("group_size", 2)))
                budget = st.number_input("Budget per night", min_value=0, step=10, value=int(u.get("budget", 300)))
                pref_raw = st.text_input("Preferred environment (comma separated)",
                                         value=", ".join(u.get("preferred_environment", [])))
                save_btn = st.form_submit_button("Save")
            if save_btn:
                ok, msg = update_user(users_dict, user_id, {
                    "name": name.strip(),
                    "group_size": int(group_size),
                    "budget": float(budget),
                    "preferred_environment": [e.strip() for e in pref_raw.split(",") if e.strip()],
                })
                if ok:
                    st.success("Profile saved.")
                else:
                    st.error(msg)
            st.divider()
            if st.button("Sign out", use_container_width=True):
                st.session_state.pop("auth_user", None)
                st.rerun()

    left, right = st.columns([10, 1])
    with left:
        st.markdown("<h1 style='margin:0'>Gr8stayz</h1>", unsafe_allow_html=True)
    with right:
        user_popover(users_dict, auth_user)

    # Chat: instant semantic dataset cards + short LLM guidance
    with st.container(border=True):
        st.subheader("Trip Assistant")
        if "chat" not in st.session_state:
            st.session_state["chat"] = []

        user_msg = st.chat_input("Type your plan, e.g., stay in Banff in January under $300")
        if user_msg:
            st.session_state["chat"].append({"role": "user", "content": user_msg})

            # Instant semantic suggestions from embeddings
            instant = semantic_candidates_for_query(user_msg, properties, top_k=12)

            st.markdown("#### Instant suggestions")
            if instant:
                show_map(instant)
                rows = (len(instant) + 2) // 3
                i = 0
                for _ in range(rows):
                    cols = st.columns(3)
                    for c in cols:
                        if i < len(instant):
                            with c:
                                property_card(instant[i], sim=instant[i].get("_semantic_sim"))
                            i += 1
            else:
                st.info("No instant semantic matches for this message. Try adding a location keyword or disable SAFE_MODE.")

            guide = llm_short_guidance(user_msg, shown_properties=instant)
            st.session_state["chat"].append({"role": "assistant", "content": guide})

        for msg in st.session_state["chat"][-6:]:
            if msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(sanitize(msg["content"]))

    # Filters section (date range enforced)
    location, group_size, budget, pref_env, sort_choice, avail_range = render_filters(properties, profile)

    # Filtered recommendations
    st.markdown("### Recommended listings")
    if avail_range and isinstance(avail_range, tuple):
        cap = f"Location: {location} | Budget ≤ {budget} | Preferred: {', '.join(pref_env) if pref_env else 'Any'}"
        cap += f" | Dates: {avail_range[0]} → {avail_range[1]} | Sort: {sort_choice}"
        st.caption(cap)
        ranked, sim_map = filter_and_rank(properties, location, group_size, budget, pref_env, sort_choice, avail_range)

        show_map(ranked)
        rows = (len(ranked) + 2) // 3
        i = 0
        for _ in range(rows):
            cols = st.columns(3)
            for c in cols:
                if i < len(ranked):
                    with c:
                        property_card(ranked[i], sim=sim_map.get(ranked[i]["property_id"], 0.0))
                    i += 1

        if not ranked:
            st.info("No listings match the current filters and date range. Try widening location or budget, or changing dates.")
    else:
        st.warning("Please select a valid start date and end date in the filter to see recommendations.")

except Exception as e:
    st.error("Startup error:")
    st.exception(e)
    st.code(traceback.format_exc())
