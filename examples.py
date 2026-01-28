"""Showcase examples for specplot README."""

from specplot import diagram, node


def hero_example():
    """Hero example: Simple 3-tier web architecture in 12 lines."""
    with diagram(filename="docs/hero"):
        user = node(icon="person", label="User")

        with node(icon="cloud", label="Web App", show_as="group", grid=(1, 2)):
            frontend = node(icon="web", label="Frontend")
            api = node(icon="api", label="API")

        db = node(icon="database", label="PostgreSQL")

        user >> frontend >> api >> db


def example_microservices():
    """Example 1: E-commerce microservices architecture."""
    with diagram(filename="docs/example_microservices"):
        # Entry points
        web = node(icon="web", label="Web Client")
        mobile = node(icon="smartphone", label="Mobile App")

        # API Gateway
        gateway = node(icon="api", label="API Gateway",
                      description="Auth, rate limiting, routing")

        # Services as a group
        with node(icon="cloud", label="Services", show_as="group", grid=(2, 3)):
            users = node(icon="person", label="User Service")
            products = node(icon="inventory_2", label="Product Service")
            orders = node(icon="receipt_long", label="Order Service")
            payments = node(icon="payments", label="Payment Service")
            inventory = node(icon="warehouse", label="Inventory Service")
            notify = node(icon="notifications", label="Notification Service")

        # Data stores
        with node(icon="storage", label="Data Layer", show_as="group", grid=(1, 3)):
            userdb = node(icon="database", label="Users DB")
            productdb = node(icon="database", label="Products DB")
            orderdb = node(icon="database", label="Orders DB")

        queue = node(icon="sync_alt", label="Message Queue",
                    description="Async event processing")

        # Connections
        web >> gateway
        mobile >> gateway
        gateway >> users
        gateway >> products
        gateway >> orders
        orders >> payments | "process"
        orders >> inventory | "reserve"
        payments >> notify | "confirm"
        users >> userdb
        products >> productdb
        orders >> orderdb
        orders >> queue


def example_layered():
    """Example 2: Clean Architecture with outline mode."""
    with diagram(filename="docs/example_layered"):
        # Presentation layer
        with node(icon="web", label="Presentation",
                 description="UI components and controllers",
                 show_as="outline"):
            node(label="React Components")
            node(label="REST Controllers")
            node(label="GraphQL Resolvers")

        # Application layer
        with node(icon="account_tree", label="Application",
                 description="Use cases and orchestration",
                 show_as="outline") as app:
            node(label="Command Handlers")
            node(label="Query Handlers")
            node(label="Event Handlers")

        # Domain layer
        with node(icon="hub", label="Domain",
                 description="Business logic and rules",
                 show_as="outline") as domain:
            node(label="Entities")
            node(label="Value Objects")
            node(label="Domain Services")
            node(label="Repository Interfaces")

        # Infrastructure layer
        with node(icon="dns", label="Infrastructure",
                 description="External concerns",
                 show_as="outline") as infra:
            node(label="Database Repositories")
            node(label="External API Clients")
            node(label="Message Brokers")
            node(label="Caching")

        # Dependency flow (inward)
        app >> domain | "depends on"
        infra >> domain | "implements"


def example_pipeline():
    """Example 3: ML Data Pipeline with mixed modes."""
    with diagram(filename="docs/example_pipeline"):
        # Data sources
        with node(icon="source", label="Data Sources", show_as="group", grid=(1, 3)):
            api_src = node(icon="api", label="REST APIs")
            db_src = node(icon="database", label="Databases")
            files = node(icon="folder", label="File Storage")

        # Ingestion
        ingest = node(icon="input", label="Ingestion Layer",
                     description="Apache Kafka / Kinesis")

        # Processing with outline
        with node(icon="memory", label="Processing",
                 description="Spark / Flink jobs",
                 show_as="outline") as proc:
            node(label="Validation & Cleaning")
            node(label="Feature Engineering")
            node(label="Aggregations")

        # Storage
        lake = node(icon="waves", label="Data Lake",
                   description="S3 / Delta Lake")

        # ML Pipeline as group
        with node(icon="psychology", label="ML Pipeline", show_as="group", grid=(1, 2)):
            train = node(icon="model_training", label="Training")
            serve = node(icon="cloud_upload", label="Model Serving")

        # Outputs
        with node(icon="analytics", label="Outputs", show_as="group", grid=(1, 2)):
            dash = node(icon="dashboard", label="Dashboards")
            alerts = node(icon="notifications", label="Alerts")

        # Flow
        api_src >> ingest
        db_src >> ingest
        files >> ingest
        ingest >> proc | "stream"
        proc >> lake | "batch"
        lake >> train | "features"
        train >> serve | "deploy"
        serve >> dash | "predictions"
        serve >> alerts | "anomalies"


if __name__ == "__main__":
    import os
    os.makedirs("docs", exist_ok=True)

    print("Generating hero example...")
    hero_example()

    print("Generating microservices example...")
    example_microservices()

    print("Generating layered architecture example...")
    example_layered()

    print("Generating data pipeline example...")
    example_pipeline()

    print("\nAll examples generated in docs/")
