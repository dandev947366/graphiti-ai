"""
This script tests connectivity to a Neo4j database using credentials loaded from a .env file.

Functionality:
- Loads Neo4j connection details (URI, username, password) from environment variables.
- Establishes a connection to the Neo4j database using the official Python driver.
- Runs a simple test Cypher query (`RETURN 1 AS test`) to verify the connection.
- Prints the result of the test query or an error message if the connection fails.

Requirements:
- .env file containing NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD variables.
- neo4j Python driver (`neo4j` package).
- python-dotenv to load environment variables from the .env file.

Use Case:
Useful as a simple connectivity check to ensure that your application can
connect to a Neo4j instance before running more complex queries or transactions.
"""

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
