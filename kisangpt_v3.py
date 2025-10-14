"""
kisangpt_pipeline_v3.py
⚡ Upgraded for Speed and Efficiency:
✅ Intelligent fertilizer intent detection
✅ Mandatory fertilizer calculation (when applicable)
✅ Smart 24-hour caching for KCC API data
✅ Parallel data retrieval (KCC + RAG + Embeddings)
✅ Structured fertilizer recommendation output
✅ Uses fertilizer_compositions.csv dynamically
"""

import os
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import numpy as np

# ----------------------------
# CONFIGURATION
# ----------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
KCC_API_KEY = os.getenv("KCC_API_KEY")

if not GEMINI_KEY:
    raise SystemExit("❌ Set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_KEY)
GEMINI_MODEL = "models/gemini-2.5-flash"  # ⚡ Faster than 2.5-flash for short answers

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
KCC_CACHE_DIR = DATA_DIR / "kcc_cache"
KCC_CACHE_DIR.mkdir(exist_ok=True)
CHROMA_DIR = "chroma_db"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # ⚡ Smaller & faster
KCC_API_BASE = "https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f"
BATCH_SIZE = 1000
MAX_RECORDS = 20000
REQUEST_TIMEOUT = 15
RETRY_LIMIT = 4
CACHE_EXPIRY_HOURS = 24  # ⚡ Cache refresh period

# Fertilizer data
FERT_PATH = DATA_DIR / "fertilizer_compositions.csv"
if FERT_PATH.exists():
    fert_df = pd.read_csv(FERT_PATH)
else:
    fert_df = pd.DataFrame(columns=["Fertilizer Name", "N%", "P%", "K%", "Type"])
    print("⚠️ fertilizer_compositions.csv not found!")

# ----------------------------
# IDEAL NPK reference table
# ----------------------------
IDEAL_NPK = {
    "Rice": (100, 60, 40),
    "Wheat": (120, 60, 40),
    "Maize": (150, 75, 50),
    "Groundnut": (25, 25, 25),
    "Cotton": (80, 40, 40),
    "Sugarcane": (200, 100, 100),
    "Potato": (120, 80, 100),
    "Paddy": (100, 60, 40),
    "Soybean": (30, 60, 30),
    "Barley": (60, 40, 20),
    "Sorghum": (90, 45, 45),
    "Pearl Millet": (60, 30, 20),
    "Finger Millet": (60, 30, 30),
    "Oat": (80, 40, 20),
    "Chickpea": (20, 40, 20),
    "Pigeon Pea": (20, 50, 20),
    "Black Gram": (20, 40, 20),
    "Green Gram": (20, 40, 20),
    "Lentil": (20, 40, 20),
    "Pea": (20, 40, 20),
    "Mustard": (80, 40, 40),
    "Rapeseed": (80, 40, 40),
    "Sunflower": (60, 60, 40),
    "Sesame": (40, 20, 20),
    "Linseed": (40, 20, 20),
    "Castor": (60, 40, 40),
    "Safflower": (40, 20, 20),
    "Tobacco": (80, 40, 40),
    "Jute": (80, 40, 40),
    "Mesta": (60, 30, 30),
    "Cotton": (80, 40, 40),
    "Sugarbeet": (120, 60, 100),
    "Carrot": (80, 60, 100),
    "Onion": (100, 50, 50),
    "Garlic": (100, 50, 50),
    "Tomato": (120, 60, 60),
    "Brinjal": (100, 50, 50),
    "Chilli": (80, 40, 40),
    "Capsicum": (80, 40, 40),
    "Okra": (80, 40, 40),
    "Cabbage": (120, 60, 60),
    "Cauliflower": (120, 60, 60),
    "Radish": (60, 40, 40),
    "Turnip": (60, 40, 40),
    "Spinach": (60, 40, 40),
    "Fenugreek": (40, 20, 20),
    "Coriander": (40, 20, 20),
    "Cumin": (40, 20, 20),
    "Fennel": (40, 20, 20),
    "Dill": (40, 20, 20),
    "Mint": (60, 40, 40),
    "Basil": (40, 20, 20),
    "Parsley": (40, 20, 20),
    "Pumpkin": (60, 40, 40),
    "Bottle Gourd": (60, 40, 40),
    "Bitter Gourd": (60, 40, 40),
    "Ridge Gourd": (60, 40, 40),
    "Sponge Gourd": (60, 40, 40),
    "Cucumber": (60, 40, 40),
    "Watermelon": (60, 40, 40),
    "Muskmelon": (60, 40, 40),
    "Papaya": (100, 60, 60),
    "Banana": (200, 60, 300),
    "Mango": (100, 50, 100),
    "Guava": (100, 50, 100),
    "Sapota": (80, 40, 80),
    "Pomegranate": (80, 40, 80),
    "Citrus": (100, 50, 100),
    "Grapes": (120, 60, 120),
    "Apple": (80, 40, 80),
    "Pear": (80, 40, 80),
    "Peach": (80, 40, 80),
    "Plum": (80, 40, 80),
    "Apricot": (80, 40, 80),
    "Cherry": (80, 40, 80),
    "Strawberry": (80, 40, 80),
    "Pineapple": (100, 50, 100),
    "Jackfruit": (80, 40, 80),
    "Cashew": (80, 40, 80),
    "Coconut": (100, 50, 100),
    "Arecanut": (100, 50, 100),
    "Coffee": (80, 40, 80),
    "Tea": (80, 40, 80),
    "Rubber": (80, 40, 80),
    "Oil Palm": (120, 60, 120),
    "Betel Vine": (80, 40, 80),
    "Turmeric": (80, 40, 80),
    "Ginger": (80, 40, 80),
    "Cardamom": (80, 40, 80),
    "Black Pepper": (80, 40, 80),
    "Clove": (80, 40, 80),
    "Nutmeg": (80, 40, 80),
    "Vanilla": (80, 40, 80),
    "Areca": (80, 40, 80),
    "Lemon Grass": (80, 40, 80),
    "Sugarcane": (200, 100, 100),
    "Sweet Potato": (80, 40, 80),
    "Yam": (80, 40, 80),
    "Colocasia": (80, 40, 80),
    "Amaranthus": (60, 40, 40),
    "Drumstick": (60, 40, 40),
    "Beetroot": (80, 40, 80),
    "Lettuce": (60, 40, 40),
    "Broccoli": (80, 40, 80),
    "Kale": (80, 40, 80),
    "Leek": (60, 40, 40),
    "Celery": (60, 40, 40),
    "Artichoke": (80, 40, 80),
    "Asparagus": (80, 40, 80),
}

# ----------------------------
# Utility: Safe API fetch with retry
# ----------------------------
def safe_get_json(url, params):
    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"⚠️ API error: {e} — retrying ({attempt+1}/{RETRY_LIMIT})")
            time.sleep(2 ** attempt)
    return {}

def normalize_state(s): 
    return re.sub(r"\s+", " ", s.strip().upper()) if s else ""

# ----------------------------
# STEP 1: Fetch KCC dataset (state-level, cached + refresh every 24h)
# ----------------------------
def fetch_kcc_state(state: str):
    state_up = normalize_state(state)
    cache_path = KCC_CACHE_DIR / f"kcc_{state_up}.csv"

    # ⚡ Smart cache reuse if < 24 hours old
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=CACHE_EXPIRY_HOURS):
            print(f"⚡ Using cached KCC data for {state_up}")
            return pd.read_csv(cache_path)

    print(f"🌐 Fetching fresh KCC data for {state_up}")
    records, offset = [], 0
    while offset < MAX_RECORDS:
        params = {
            "api-key": KCC_API_KEY,
            "format": "json",
            "limit": BATCH_SIZE,
            "offset": offset,
            "filters[StateName]": state_up,
        }
        data = safe_get_json(KCC_API_BASE, params)
        batch = data.get("records", [])
        if not batch:
            print("🔚 No more records.")
            break
        records.extend(batch)
        offset += len(batch)
        print(f" → fetched {len(batch)} rows (total {len(records)})")
        if len(batch) < BATCH_SIZE:
            break

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(cache_path, index=False)
        print(f"💾 Cached {len(df)} rows for {state_up}")
    return df

# ----------------------------
# STEP 2: Local filtering (crop/category/sector)
# ----------------------------
def filter_kcc(df, crop=None, category=None, sector=None):
    if df.empty:
        return df
    d = df.copy()
    for c in ["Crop", "Category", "Sector", "QueryText", "QueryType", "KccAns"]:
        if c not in d.columns:
            d[c] = ""
    if crop:
        d = d[d["Crop"].str.contains(re.escape(crop), case=False, na=False)]
    if category:
        d = d[d["Category"].str.contains(re.escape(category), case=False, na=False)]
    if sector:
        d = d[d["Sector"].str.contains(re.escape(sector), case=False, na=False)]
    print(f"🔎 Filtered {len(d)} rows (crop={crop}, cat={category}, sec={sector})")
    return d

# ----------------------------
# STEP 3: Semantic similarity ranking (optimized)
# ----------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def semantic_rank(df, query, top_k=5):
    if df.empty: return []
    texts = (df["QueryType"].fillna("") + " " + df["QueryText"].fillna("")).tolist()
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    doc_embs = embed_model.encode(texts, normalize_embeddings=True)
    scores = (np.array(doc_embs) @ q_emb.T).squeeze()
    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(top_k)
    results = [
        {
            "query_text": row["QueryText"],
            "answer": row["KccAns"],
            "category": row.get("Category", ""),
            "crop": row.get("Crop", ""),
            "score": row["score"],
        }
        for _, row in top.iterrows()
    ]
    print(f"🟢 Semantic ranking returned {len(results)} results")
    return results

# ----------------------------
# STEP 4: RAG document retrieval
# ----------------------------
def retrieve_docs(question, k=5):
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
        col = client.get_or_create_collection("kisangpt", embedding_function=ef)
        res = col.query(query_texts=[question], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        context = "\n\n".join([f"Source: {m.get('source')} p.{m.get('page')} - {d[:300]}" for d, m in zip(docs, metas)])
        print(f"📚 Retrieved {len(docs)} doc chunks.")
        return context
    except Exception as e:
        print("⚠️ RAG retrieval failed:", e)
        return ""

# ----------------------------
# STEP 5: Fertilizer calculator (dynamic)
# ----------------------------
def calculate_fertilizer(crop, N, P, K):
    crop_key = crop.capitalize()
    if crop_key not in IDEAL_NPK:
        return {"error": f"No ideal NPK found for {crop_key}"}

    ideal_N, ideal_P, ideal_K = IDEAL_NPK[crop_key]
    deficit = {
        "N": max(0, ideal_N - N),
        "P": max(0, ideal_P - P),
        "K": max(0, ideal_K - K),
    }

    recs = []
    for nutrient, thresh, col in [("N", 20, "N%"), ("P", 10, "P%"), ("K", 10, "K%")]:
        val = deficit[nutrient]
        if val > 0 and not fert_df.empty:
            f = fert_df.loc[fert_df[col] > thresh].sort_values(col, ascending=False).iloc[0]
            qty = (val / f[col]) * 100
            recs.append({
                "nutrient": nutrient,
                "fertilizer": f["Fertilizer Name"],
                "qty_kg_per_ha": round(qty, 2),
                "type": f.get("Type", "N/A")
            })
    return {"deficit": deficit, "recs": recs}

# ----------------------------
# STEP 6: Intent detection
# ----------------------------
def fertilizer_intent(question):
    keywords = ["fertilizer", "urea", "dap", "mop", "recommend", "dose", "apply"]
    return any(k in question.lower() for k in keywords)

# ----------------------------
# STEP 7: Ask Gemini
# ----------------------------
def ask_gemini(query, kcc_results, docs, fert_result=None):
    kcc_ctx = "\n".join([f"Q: {r['query_text']}\nA: {r['answer']}" for r in kcc_results]) or "No KCC context."
    fert_ctx = ""
    if fert_result:
        if "error" in fert_result:
            fert_ctx = fert_result["error"]
        else:
            fert_ctx = "Fertilizer Calculation:\n" + json.dumps(fert_result, indent=2)

    prompt = f"""
You are KisanGPT, a farmer assistant.
Answer shortly (3–6 lines) in simple Hinglish using only the context below.
If information is not available, say "I don't know, please consult KVK".

--- KCC Context ---
{kcc_ctx}

--- Document Context ---
{docs or 'None'}

--- Fertilizer Info ---
{fert_ctx or 'None'}

Farmer Question: {query}
"""
    try:
        resp = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print("⚠️ Gemini failed:", e)
        return "I don't know, please consult KVK."

# ----------------------------
# STEP 8: Full optimized flow with parallel execution
# ----------------------------
def kisangpt_answer(question, state, crop=None, category=None, sector=None, npk_values=None):
    with ThreadPoolExecutor() as executor:
        # Run fetches concurrently
        f_kcc = executor.submit(fetch_kcc_state, state)
        f_docs = executor.submit(retrieve_docs, question)

        df = f_kcc.result()
        docs = f_docs.result()

    df = filter_kcc(df, crop, category, sector)
    kcc_res = semantic_rank(df, question)

    fert_result = None
    if npk_values and crop and fertilizer_intent(question):
        try:
            N, P, K = map(float, npk_values)
            fert_result = calculate_fertilizer(crop, N, P, K)
        except:
            fert_result = {"error": "Invalid NPK values."}

    answer = ask_gemini(question, kcc_res, docs, fert_result)
    return answer

# ----------------------------
# MAIN LOOP
# ----------------------------
if __name__ == "__main__":
    print("🌾 KisanGPT v3 ready (optimized).")
    while True:
        q = input("\n👨‍🌾 Farmer question (or 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        s = input("🏠 StateName (required): ")
        crop = input("🌾 Crop (optional): ") or None
        cat = input("📘 Category (optional): ") or None
        sec = input("🏢 Sector (optional): ") or None
        npk_str = input("🧪 N,P,K (optional): ") or ""
        npk_vals = [float(x) for x in npk_str.split(",")] if npk_str else None
        ans = kisangpt_answer(q, s, crop, cat, sec, npk_vals)
        print("\n🤖 KisanGPT:", ans)
