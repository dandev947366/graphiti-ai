from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this to your Neo4j password

cypher_script = """
// Clean DB first
MATCH (n) DETACH DELETE n;

// Create company
CREATE (c:Company {id: 1, name: 'Green Energy Inc', phone: '123456789', address: '123 Energy St', logo: 'logo.png', details: 'Leading renewable energy company'});

// Create users
CREATE (u1:User {id: 1, firstname: 'Alice', lastname: 'Smith', username: 'alice', email: 'alice@green.com', role: 'admin', company_id: 1});
CREATE (u2:User {id: 2, firstname: 'Bob', lastname: 'Brown', username: 'bob', email: 'bob@green.com', role: 'user', company_id: 1});

// Link users to company
MATCH (c:Company {id: 1}), (u:User)
WHERE u.company_id = 1
CREATE (c)-[:HAS_USER]->(u);

// Create sites
CREATE (s1:Site {id: 1, name: 'Solar Plant A', description: 'Main solar site', company_id: 1, created_by: 1});
CREATE (s2:Site {id: 2, name: 'Wind Farm B', description: 'Wind site', company_id: 1, created_by: 2});

// Link sites to company and users
MATCH (c:Company {id:1}), (s:Site)
WHERE s.company_id = 1
CREATE (c)-[:HAS_SITE]->(s);
MATCH (u:User), (s:Site)
WHERE u.id = s.created_by
CREATE (u)-[:CREATED_SITE]->(s);

// Create systems related to sites
CREATE (sys1:System {id: 1, name: 'PV System 1', site_id: 1});
CREATE (sys2:System {id: 2, name: 'Wind Turbine System', site_id: 2});

// Link systems to sites
MATCH (s:Site), (sys:System)
WHERE s.id = sys.site_id
CREATE (s)-[:HAS_SYSTEM]->(sys);

// Create devices (sensors) linked to systems
CREATE (d1:Device {id: 1, name: 'Sensor 1', manufacture_type: 'MQTT', system_id: 1});
CREATE (d2:Device {id: 2, name: 'Sensor 2', manufacture_type: 'MQTT', system_id: 2});

// Link devices to systems
MATCH (sys:System), (d:Device)
WHERE sys.id = d.system_id
CREATE (sys)-[:HAS_DEVICE]->(d);

// Create vehicles
CREATE (v1:Vehicle {id: 1, name: 'Service Truck 1', company_id: 1, created_by: 1});
CREATE (v2:Vehicle {id: 2, name: 'Service Van 2', company_id: 1, created_by: 2});

// Link vehicles to company and users
MATCH (c:Company {id:1}), (v:Vehicle)
WHERE v.company_id = 1
CREATE (c)-[:HAS_VEHICLE]->(v);
MATCH (u:User), (v:Vehicle)
WHERE u.id = v.created_by
CREATE (u)-[:CREATED_VEHICLE]->(v);

// Add sample energy consumption data as nodes or properties on sites or systems
CREATE (ec1:EnergyConsumption {site_id: 1, date: date('2025-06-02'), consumption_kwh: 1500, cost: 200, co2_emission: 300});
CREATE (ec2:EnergyConsumption {site_id: 2, date: date('2025-06-02'), consumption_kwh: 900, cost: 120, co2_emission: 180});

// Link energy consumption to sites
MATCH (s:Site), (ec:EnergyConsumption)
WHERE s.id = ec.site_id
CREATE (s)-[:HAS_ENERGY_CONSUMPTION]->(ec);
"""


def run_cypher_script(uri, user, password, script):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # Neo4j driver does not support multiple statements separated by semicolons directly,
        # so we split the script and run statements one by one.
        for statement in script.strip().split(";"):
            if statement.strip():
                session.run(statement)
    driver.close()
    print("Data setup completed.")


if __name__ == "__main__":
    run_cypher_script(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, cypher_script)
