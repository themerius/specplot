"""Showcase examples for specplot README."""

from specplot import diagram, node, edge


def hero_example():
    """Hero example: Shows all key features in one compact diagram."""
    with diagram(
            filename="docs/hero"
    ):
        # Node without description
        user = node(icon="person", label="User")

        # Group node with children (grid layout)
        with node(
                icon="cloud",
                label="Our WebApp",
                description="Layered architecture style.",
                show_as="group",
                grid=(3, 1)
        ) as app:
            node(icon="web", label="Presentation Layer",
                 description="User interface components")
            node(icon="account_tree", label="Business Layer",
                 description="Business logic and rules")
            persistance = node(icon="storage", label="Persistence Layer",
                               description="Data access and ORM")

        # Outline node (bullet list style)
        with node(
                icon="database",
                label="Database Layer",
                show_as="outline"
        ) as dbs:
            node(label="PostgreSQL")
            node(label="Redis Cache")

        # Edges with and without labels
        user >> app
        persistance >> dbs | "read/write"


def example_pipeline():
    """Example 3: ML Data Pipeline with mixed modes."""
    with diagram(
            filename="docs/example_pipeline",
            layout=(
                    ("LR",),  # Sources
                    ("LR", "LR"),  # Ingestion + Processing
                    ("LR",),  # Lake
                    ("LR", "LR"),  # ML + Outputs
            )
    ):
        # Data sources
        with node(icon="source", label="Data Sources", show_as="group", grid=(1, 3), pos=1):
            api_src = node(icon="api", label="APIs")
            db_src = node(icon="database", label="Databases")
            files = node(icon="folder", label="Files")

        # Ingestion
        ingest = node(icon="input", label="Ingestion",
                      description="Kafka / Kinesis", pos=2)

        # Processing with outline
        with node(icon="memory", label="Processing",
                  description="Spark / Flink",
                  show_as="outline", pos=3) as proc:
            node(label="Validation")
            node(label="Feature Engineering")
            node(label="Aggregations")

        # Storage
        lake = node(icon="waves", label="Data Lake",
                    description="S3 / Delta Lake", pos=4)

        # ML Pipeline as group
        with node(icon="psychology", label="ML Pipeline", show_as="group", grid=(2, 1), pos=5) as ml:
            train = node(icon="model_training", label="Training")
            serve = node(icon="cloud_upload", label="Serving")

        # Outputs
        with node(icon="analytics", label="Outputs", show_as="group", grid=(2, 1), pos=6):
            dash = node(icon="dashboard", label="Dashboards")
            alerts = node(icon="notifications", label="Alerts")

        # Flow
        api_src >> ingest
        db_src >> ingest
        files >> ingest
        ingest >> proc | "stream"
        proc >> lake | "batch"
        lake >> ml | "features"
        train >> serve | "deploy"
        serve >> dash | "predictions"
        serve >> alerts | "anomalies"


def example_event_driven():
    """Event-Driven Architecture - Broker topology.

    Reference: Fundamentals of Software Architecture, Chapter 14
    """
    with diagram(
            filename="docs/example_event_driven",
            pathfinding=False,
            layout=(
                    ("TB", "TB", "TB"),  # Initiator, channels, processors
            )
    ):
        # Initiating Event
        initiator = node(
            icon="bolt",
            label="Initiating Event",
            description="Triggers the flow",
            pos=1
        )

        # Event Channels (broker)
        with node(
                icon="sync_alt",
                label="Event Channels",
                description="Message broker",
                show_as="group",
                grid=(3, 1),
                pos=2
        ) as channels:
            ch1 = node(icon="valve", label="Event Channel 1")
            ch2 = node(icon="valve", label="Event Channel 2")
            ch3 = node(icon="valve", label="Event Channel 3")

        # Event Processors (left side)
        with node(icon="settings", label="Processor", show_as="outline", pos=1) as proc2:
            node(label="Component")
            node(label="Component")

        with node(icon="settings", label="Processor", show_as="group", grid=(3, 1), pos=1) as proc4:
            node(label="Component")
            node(label="Component")
            proc_event2 = node(icon="bolt", label="Processing Event")

        # Event Processors (right side)
        with node(icon="settings", label="Processor", show_as="group", grid=(3, 1), pos=3) as proc1:
            node(label="Component")
            node(label="Component")
            proc1_event = node(icon="bolt", label="Processing Event")

        with node(icon="settings", label="Processor", show_as="outline", pos=3) as proc3:
            node(label="Component")
            node(label="Component")

        with node(icon="settings", label="Processor", show_as="outline", pos=3) as proc5:
            node(label="Component")
            node(label="Component")

        # Connections
        edge(initiator, ch1, style='..>')
        edge(ch1, proc1, style='..>')
        edge(proc1_event, ch2, style='..>')
        edge(ch2, proc2, style='..>')
        edge(ch2, proc3, style='..>')
        edge(ch2, proc4, style='..>')
        edge(proc_event2, ch3, style='..>')
        edge(ch3, proc5, style='..>')


if __name__ == "__main__":
    import os

    os.makedirs("docs", exist_ok=True)

    print("Generating hero example...")
    hero_example()

    print("Generating data pipeline example...")
    example_pipeline()

    print("Generating event-driven architecture...")
    example_event_driven()

    print("\nAll examples generated in docs/")
