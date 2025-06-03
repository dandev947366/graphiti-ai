from neo4j import GraphDatabase, basic_auth
import requests
import json

# Neo4j config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Ollama config (Mistral running locally)
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral"


def get_people_profiles(driver):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person)
            RETURN p.name AS name, p.age AS age, p.gender AS gender,
                   p.occupation AS occupation, p.city AS city
        """
        )
        return [dict(record) for record in result]


def get_relationships(driver):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Person)-[r]->(b:Person)
            RETURN a.name AS from, type(r) AS relationship, b.name AS to, r
        """
        )
        relationships = []
        for record in result:
            rel = {
                "from": record["from"],
                "relationship": record["relationship"],
                "to": record["to"],
                "properties": dict(record["r"].items()),
            }
            relationships.append(rel)
        return relationships


def ask_mistral(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant who analyzes and summarizes people's information and their relationships.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
    )
    try:
        people = get_people_profiles(driver)
        relationships = get_relationships(driver)

        if not people:
            print("No people data found.")
            return

        prompt = (
            f"Here is a list of people with detailed info:\n{json.dumps(people, indent=2)}\n\n"
            f"And here are the relationships between them:\n{json.dumps(relationships, indent=2)}\n\n"
            "Please provide a summary of the people and their connections."
        )

        summary = ask_mistral(prompt)

        print("\n=== Mistral Response ===")
        print(summary)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
