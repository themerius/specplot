"""Example usage of specplot."""

from specplot import diagram, node


def sandwich_example():
    """Create a sandwich layout example (3 rows: 1 zone, 3 zones, 1 zone)."""
    with diagram(
        filename="sandwich_example",
        layout=(("LR",), ("TB", "TB", "TB"), ("LR",))
    ):
        # pos=1: top LR zone
        user = node(icon="person", label="User", pos=1)

        # pos=2,3,4: middle TB columns
        api = node(icon="api", label="API Gateway", pos=2)
        auth = node(icon="lock", label="Auth Service", pos=2)

        svc = node(icon="dns", label="Core Service", pos=3)
        cache = node(icon="memory", label="Cache", pos=3)

        db = node(icon="database", label="Database", pos=4)
        backup = node(icon="backup", label="Backup", pos=4)

        # pos=5: bottom LR zone
        logs = node(icon="description", label="Logs", pos=5)
        metrics = node(icon="monitoring", label="Metrics", pos=5)

        # Edges
        user >> api >> svc >> db
        api >> auth
        svc >> cache
        db >> backup
        svc >> logs
        svc >> metrics

    print("Sandwich diagram saved to sandwich_example.svg")


def main():
    """Create an example architecture diagram."""
    with diagram(filename="example"):
        # Simple node - no 'with' needed when no children
        user = node(icon="person", label="User")

        # Group with nested nodes - use 'with' for children
        with node(
            icon="hive",
            label="Swarm Colony",
            description="Contains essential enviornment for agents to work.",
        ) as env:
            cli = node(icon="terminal", label="Command Line Interface")
            goal = node(icon="assignment", label="Overall Goal")
            db = node(icon="database", label="Data Store", description="hello db")
            dbf = node(icon="database", label="Failure Store", description="hello db")
            dbs = node(icon="database", label="Signals", description="hello db")

        # Outline mode (default) - use 'with' for children
        with node(
            icon="genetics",
            label="Swarm Agents",
            description="All agents implement this interface.",
            show_as="group",
            grid=(3, 2),
        ) as agents:
            node(icon="psychology", label="Head", description="This thing will parse user input and translate it to machine readable inputs.")
            node(icon="psychology", label="Email2PDF")
            node(icon="psychology", label="PDF2Markdown")
            node(icon="psychology", label="Classification", description="Detects if invoice")
            with node(icon="psychology", label="Annotate", description="Tagging of invoice infos"):
                node(icon="psychology", label="sub agent 1")
                node(icon="psychology", label="sub agent 2")
            writer = node(icon="psychology", label="Writer")

        node(icon="cloud_download", label="IMAP service")
        node(icon="extension", label="docling lib")
        node(icon="graph_5", label="LLM (4b)")
        node(icon="graph_5", label="LLM (1.7b)")
        fs = node(icon="folder_open", label="Filesystem")

        # Edges
        user >> cli
        agents >> env | "uses"
        writer >> fs | "writes"

    print("Diagram saved to example.svg")


if __name__ == "__main__":
    main()
    sandwich_example()
