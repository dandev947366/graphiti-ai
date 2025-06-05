import singlestoredb as s2
import struct
import os
from dotenv import load_dotenv
import singlestoredb as s2

# Load environment variables from .env file
load_dotenv()

SINGLESTORE_URL = os.environ.get("SINGLESTORE_URL")
conn = s2.connect(SINGLESTORE_URL)
print("Connection is alive?", conn.is_connected())

# Sample data
text_data = "example sentence"
embedding = [0.12, 0.98, 0.45, 0.33]  # your embedding vector (floats)


# Convert float list to bytes (float32)
def floats_to_blob(floats):
    return struct.pack(f"{len(floats)}f", *floats)


vector_blob = floats_to_blob(embedding)


with conn:
    with conn.cursor() as cur:
        sql = "INSERT INTO myvectortable (text, vector) VALUES (%s, %s)"
        cur.execute(sql, (text_data, vector_blob))
        print("Inserted embedding into myvectortable")
