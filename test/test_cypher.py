from neo4j import GraphDatabase, basic_auth

# Set your Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Replace with your actual password

# Cypher test queries
QUERIES = [
    ("Clean up database", "MATCH (n) DETACH DELETE n"),
    # Create nodes
    (
        "Create Alice",
        "CREATE (:Person {name: 'Alice', age: 30, gender: 'female', occupation: 'Engineer', city: 'Helsinki'})",
    ),
    (
        "Create Bob",
        "CREATE (:Person {name: 'Bob', age: 34, gender: 'male', occupation: 'Designer', city: 'Helsinki'})",
    ),
    (
        "Create Charlie",
        "CREATE (:Person {name: 'Charlie', age: 29, gender: 'male', occupation: 'Data Scientist', city: 'Espoo'})",
    ),
    # Create relationships
    (
        "Alice FRIENDS_WITH Bob",
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:FRIENDS_WITH {since: 2015}]->(b)",
    ),
    (
        "Bob WORKS_WITH Charlie",
        "MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'}) CREATE (b)-[:WORKS_WITH {project: 'UX Research'}]->(c)",
    ),
    (
        "Alice LIVES_IN_SAME_CITY AS Bob",
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:LIVES_IN_SAME_CITY]->(b)",
    ),
    # Query nodes
    (
        "Match all people",
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age, p.gender AS gender, p.occupation AS occupation, p.city AS city",
    ),
    # Query relationships
    (
        "List friendships",
        "MATCH (a:Person)-[r:FRIENDS_WITH]->(b:Person) RETURN a.name AS from, type(r) AS relationship, b.name AS to, r.since AS since",
    ),
    (
        "List colleagues",
        "MATCH (a:Person)-[r:WORKS_WITH]->(b:Person) RETURN a.name AS from, b.name AS to, r.project AS project",
    ),
    (
        "List people in same city",
        "MATCH (a:Person)-[r:LIVES_IN_SAME_CITY]->(b:Person) RETURN a.name AS from, b.name AS to",
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
