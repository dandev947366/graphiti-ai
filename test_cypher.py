from neo4j import GraphDatabase, basic_auth

# Set your Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this to your password

# Cypher test queries
QUERIES = [
    ("Basic return", "RETURN 1 AS test"),
    ("Create node", "CREATE (:Person {name: 'Alice', age: 30})"),
    ("Match nodes", "MATCH (p:Person) RETURN p"),
    (
        "Create relationship",
        "CREATE (a:Person {name: 'Bob'})-[:FRIENDS_WITH]->(b:Person {name: 'Charlie'})",
    ),
    (
        "Match relationship",
        """
        MATCH (a:Person)-[r:FRIENDS_WITH]->(b:Person)
        RETURN a.name AS from, type(r) AS relationship, b.name AS to
    """,
    ),
]


def run_test_queries(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    try:
        with driver.session() as session:
            for name, query in QUERIES:
                print(f"\n=== {name} ===")
                result = session.run(query)
                for record in result:
                    print(dict(record))
    except Exception as e:
        print(f"Failed to run test queries: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    run_test_queries(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
