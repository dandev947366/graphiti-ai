from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def test_connection(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            print("Connection successful, test query returned:", record["test"])
    except Exception as e:
        print("Failed to connect to Neo4j:", e)
    finally:
        driver.close()


if __name__ == "__main__":
    test_connection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
