# create_embeddings.py
# This script loads property listings, generates embeddings, and stores them in a vector database for later querying.

import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

import sqlite3

# Path to the property listings JSON file (robust to script location)
import os
PROPERTIES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'property_listings.json'))
# Path to save the SQLite database (always in the Vector embeddings folder)
SQLITE_DB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'property_vector_db.sqlite'))


# Check if the embeddings table already exists; if so, exit early
def embeddings_table_exists(db_file):
    import sqlite3
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='property_embeddings'")
    exists = c.fetchone()
    conn.close()
    return bool(exists)



_embedding_model_instance = None

def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is None:
        print("[LOG] Loading embedding model (singleton)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model_instance = SentenceTransformer('all-MiniLM-L6-v2')
        print("[LOG] Model loaded.")
    return _embedding_model_instance

def create_embeddings_db():
    if embeddings_table_exists(SQLITE_DB_FILE):
        print(f"[LOG] Embeddings table already exists in {SQLITE_DB_FILE}. Skipping embedding creation.")
        return False
    try:
        print("[LOG] Loading property listings...")
        # Load property listings
        with open(PROPERTIES_FILE, 'r', encoding='utf-8') as f:
            properties_data = json.load(f)
            properties = properties_data["properties"]
        print(f"[LOG] Loaded {len(properties)} properties.")

        model = get_embedding_model()

        # Prepare data for embedding
        property_texts = []
        property_ids = []
        for prop in properties:
            # Concatenate relevant fields for embedding, joining features and tags lists
            features_str = ' '.join(prop['features']) if isinstance(prop['features'], list) else str(prop['features'])
            tags_str = ' '.join(prop['tags']) if isinstance(prop['tags'], list) else str(prop['tags'])
            text = f"{prop['location']} {prop['type']} {features_str} {tags_str}"
            property_texts.append(text)
            property_ids.append(prop['property_id'])

        print("[LOG] Generating embeddings...")
        # Generate embeddings
        embeddings = model.encode(property_texts, show_progress_bar=True)
        print("[LOG] Embeddings generated.")

        print("[LOG] Writing embeddings and metadata to SQLite database...")
        # --- Save embeddings and metadata to SQLite ---
        conn = sqlite3.connect(SQLITE_DB_FILE)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS property_embeddings (
                property_id TEXT PRIMARY KEY,
                embedding BLOB,
                location TEXT,
                type TEXT,
                features TEXT,
                tags TEXT
            )
        ''')

        for i, prop in enumerate(properties):
            emb_blob = embeddings[i].tobytes()
            c.execute('''
                INSERT OR REPLACE INTO property_embeddings (property_id, embedding, location, type, features, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prop['property_id'],
                emb_blob,
                prop['location'],
                prop['type'],
                ','.join(prop['features']) if isinstance(prop['features'], list) else str(prop['features']),
                ','.join(prop['tags']) if isinstance(prop['tags'], list) else str(prop['tags'])
            ))
        conn.commit()
        conn.close()
        print(f"[LOG] Embeddings and metadata written to {SQLITE_DB_FILE}.")
        return True
    except Exception as e:
        import traceback
        print("[ERROR] Exception occurred during embedding creation:")
        traceback.print_exc()
        raise

# Only run embedding creation if called as a script
if __name__ == "__main__":
    create_embeddings_db()
    print(f"Vector database created and saved to {SQLITE_DB_FILE}.")


def search_property(query, db_file, model, top_k=3):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    query_emb = model.encode([query])[0]
    # Fetch all embeddings and property_ids
    c.execute('SELECT property_id, embedding, location, type, features, tags FROM property_embeddings')
    rows = c.fetchall()
    prop_ids, embs, meta = [], [], []
    for row in rows:
        prop_ids.append(row[0])
        embs.append(np.frombuffer(row[1], dtype=np.float32))
        meta.append(row[2:])
    embs = np.vstack(embs)
    sims = np.dot(embs, query_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(query_emb) + 1e-10)
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            'property_id': prop_ids[i],
            'similarity': float(sims[i]),
            'location': meta[i][0],
            'type': meta[i][1],
            'features': meta[i][2],
            'tags': meta[i][3]
        })
    conn.close()
    return results

if __name__ == "__main__":
    print(f"Vector database created and saved to {SQLITE_DB_FILE}.")
    while True:
        user_query = input("\nEnter a property search query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Exiting search.")
            break
        results = search_property(user_query, SQLITE_DB_FILE, get_embedding_model(), top_k=3)
        print("\n\033[1mTop Matching Properties\033[0m")
        print("="*60)
        for idx, res in enumerate(results, 1):
            print(f"\033[94m┌{'─'*57}┐\033[0m")
            print(f"\033[94m│\033[0m \033[1mResult #{idx}\033[0m{' '*(47-len(str(idx)))}\033[94m│\033[0m")
            print(f"\033[94m├{'─'*57}┤\033[0m")
            print(f"\033[94m│\033[0m Property ID   : \033[92m{res['property_id']}\033[0m{' '*(30-len(res['property_id']))}\033[94m│\033[0m")
            print(f"\033[94m│\033[0m Similarity    : \033[93m{res['similarity']:.4f}\033[0m{' '*(30-len(f'{res['similarity']:.4f}'))}\033[94m│\033[0m")
            print(f"\033[94m│\033[0m Location      : {res['location']}{' '*(30-len(res['location']))}\033[94m│\033[0m")
            print(f"\033[94m│\033[0m Type          : {res['type']}{' '*(30-len(res['type']))}\033[94m│\033[0m")
            print(f"\033[94m│\033[0m Features      : {res['features']}{' '*(30-len(res['features']))}\033[94m│\033[0m")
            print(f"\033[94m│\033[0m Tags          : {res['tags']}{' '*(30-len(res['tags']))}\033[94m│\033[0m")
            print(f"\033[94m└{'─'*57}┘\033[0m\n")
        print("="*60)
