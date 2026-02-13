import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS


# -------------------------
# FAISS PERSISTENCE
# -------------------------

INDEX_PATH = "storage/faiss_index"
CACHE_FILE = "storage/image_cache.json"


def save_faiss(vectorstore):
    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)


def load_faiss(embeddings):
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


# -------------------------
# IMAGE HASHING
# -------------------------

def hash_image(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs("storage", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# -------------------------
# ASYNC IMAGE PROCESSING
# -------------------------

def process_images_async(image_data_list, analyze_function):

    cache = load_cache()
    updated = False
    documents = []

    def process_single(item):
        image_bytes, metadata = item
        image_hash = hash_image(image_bytes)

        if image_hash in cache:
            return cache[image_hash], metadata, False

        description = analyze_function(image_bytes)
        return description, metadata, image_hash

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_single, image_data_list))

    for result, metadata, image_hash in results:

        if image_hash and image_hash not in cache:
            cache[image_hash] = result
            updated = True

        documents.append((result, metadata))

    if updated:
        save_cache(cache)

    return documents
