"""
Integration Tests for Neo4j-Powered Energy Dashboard Graph Schema

This test suite verifies the integrity of key relationships and data structures within a Neo4j graph database
that supports an Energy Management Dashboard. Each test ensures that the graph model is correctly populated
with meaningful and connected entities.

Tested Entities and Relationships:
- `Company` nodes and their `HAS_USER` relationships to `User` nodes
- `Site` nodes with associated `EnergyConsumption` via `HAS_ENERGY_CONSUMPTION` relationships
- `System` nodes and their linked `Device` nodes via `HAS_DEVICE`
- `User`-created `Vehicle` nodes via `CREATED_VEHICLE` relationships
- `Site` nodes containing `System` nodes via `HAS_SYSTEM` relationships

Features:
- Uses Neo4j’s Python driver to query the graph
- Employs pytest-style class-based setup/teardown
- Verifies presence and integrity of connected nodes
- Confirms data existence and structure for a specific date (e.g., for energy reports)

Requirements:
- A running Neo4j instance at `bolt://localhost:7687`
- A populated graph with nodes: Company, User, Site, EnergyConsumption, System, Device, Vehicle
- Python environment with `neo4j` driver installed

Usage:
Run the tests using a test runner like pytest:
    pytest test_energy_dashboard.py

This test suite helps validate the correctness of graph migrations, seed data, or ETL pipelines
before using the graph data in production dashboards or analytical tools.
"""

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # change this


class TestEnergyDashboard:

    @classmethod
    def setup_class(cls):
        cls.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    @classmethod
    def teardown_class(cls):
        cls.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def test_company_has_users(self):
        query = """
        MATCH (c:Company)-[:HAS_USER]->(u:User)
        RETURN c.name AS company, collect(u.username) AS users
        LIMIT 1
        """
        results = self.run_query(query)
        assert results, "No company with users found"
        assert "company" in results[0]
        assert "users" in results[0]
        assert len(results[0]["users"]) > 0

    def test_site_energy_consumption(self):
        query = """
        MATCH (s:Site)-[:HAS_ENERGY_CONSUMPTION]->(ec:EnergyConsumption)
        WHERE ec.date = date($date)
        RETURN s.name AS site, ec.consumption_kwh AS consumption, ec.cost AS cost, ec.co2_emission AS co2
        """
        date = "2025-06-02"
        results = self.run_query(query, {"date": date})
        assert results, f"No energy consumption data found for date {date}"
        for record in results:
            assert "consumption" in record
            assert record["consumption"] > 0

    def test_systems_and_devices(self):
        query = """
        MATCH (sys:System)-[:HAS_DEVICE]->(d:Device)
        RETURN sys.name AS system, collect(d.name) AS devices
        LIMIT 5
        """
        results = self.run_query(query)
        assert results, "No systems with devices found"
        for record in results:
            assert "system" in record
            assert isinstance(record["devices"], list)

    def test_vehicle_ownership(self):
        query = """
        MATCH (v:Vehicle)<-[:CREATED_VEHICLE]-(u:User)
        RETURN v.name AS vehicle, u.username AS owner
        LIMIT 5
        """
        results = self.run_query(query)
        assert results, "No vehicles with owners found"
        for record in results:
            assert "vehicle" in record
            assert "owner" in record

    def test_site_systems(self):
        query = """
        MATCH (s:Site)-[:HAS_SYSTEM]->(sys:System)
        RETURN s.name AS site, collect(sys.name) AS systems
        LIMIT 5
        """
        results = self.run_query(query)
        assert results, "No sites with systems found"
        for record in results:
            assert "site" in record
            assert isinstance(record["systems"], list)
