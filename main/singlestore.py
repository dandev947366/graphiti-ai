import os
from dotenv import load_dotenv
import singlestoredb as s2

# Load environment variables from .env file
load_dotenv()

SINGLESTORE_URL = os.environ.get("SINGLESTORE_URL")
conn = s2.connect(SINGLESTORE_URL)
print("Connection is alive?", conn.is_connected())

with conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        result = cur.fetchone()
        print("Query result:", result)
