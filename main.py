"""Example usage of specplot."""

from specplot import diagram, node


def main():
    """Create an example architecture diagram."""
    with diagram(filename="example"):
        # Simple node - no 'with' needed when no children
        user = node(icon="person", label="User")

        # Group with nested nodes - use 'with' for children
        with node(
            icon="cloud",
            label="Swarm Environment",
            description="Contains essentials for agents to work",
        ) as env:
            cli = node(icon="terminal", label="Command Line Interface")
            goal = node(icon="assignment", label="Overall Goal")
            db = node(icon="database", label="Data Store", description="hello db")
            dbf = node(icon="database", label="Failure Store", description="hello db")
            dbs = node(icon="database", label="Signals", description="hello db")

        # Outline mode (default) - use 'with' for children
        with node(
            icon="hive",
            label="AI Agents",
            description="Our intelligent agents",
            show_as="group",
            grid=(3, 2),
        ) as agents:
            node(icon="psychology", label="Head")
            node(icon="psychology", label="Email2PDF")
            node(icon="psychology", label="PDF2Markdown")
            node(icon="psychology", label="Classification", description="Detects if invoice")
            node(icon="psychology", label="Annotate", description="Tagging of invoice infos")
            writer = node(icon="psychology", label="Writer")

        fs = node(icon="folder_open", label="Filesystem")

        # Edges
        user >> cli
        agents >> env | "uses"
        writer >> fs | "writes"

    print("Diagram saved to example.svg")


if __name__ == "__main__":
    main()
