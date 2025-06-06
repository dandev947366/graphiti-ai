"""
This script connects to a SingleStore database and inserts a sample text along with its
vector embedding into a table named 'myvectortable'. The embedding is represented as
a list of float values which are converted to a binary blob using struct for storage.

Functionality:
- Loads database connection URL from a .env file.
- Connects to SingleStore using `singlestoredb`.
- Converts a list of floats (embedding) to binary format.
- Inserts the text and vector data into the database table.

Requirements:
- .env file containing SINGLESTORE_URL
- singlestoredb
- python-dotenv

Use Case:
Typically used in applications involving vector databases, such as semantic search,
retrieval-augmented generation (RAG), or machine learning feature storage.
"""

import singlestoredb as s2
import struct
import os
from dotenv import load_dotenv
import singlestoredb as s2

load_dotenv()

SINGLESTORE_URL = os.environ.get("SINGLESTORE_URL")
conn = s2.connect(SINGLESTORE_URL)
print("Connection is alive?", conn.is_connected())
text_data = "example sentence"
embedding = [0.12, 0.98, 0.45, 0.33]


# Convert float list to bytes (float32)
def floats_to_blob(floats):
    return struct.pack(f"{len(floats)}f", *floats)


vector_blob = floats_to_blob(embedding)


with conn:
    with conn.cursor() as cur:
        sql = "INSERT INTO myvectortable (text, vector) VALUES (%s, %s)"
        cur.execute(sql, (text_data, vector_blob))
        print("Inserted embedding into myvectortable")
