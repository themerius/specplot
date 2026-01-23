"""Example usage of specplot."""

from specplot import diagram, node


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

        # todo: how to steer left to right and top to bottom?
        # automatically? or should user give hints?
        # goal: compact, good readable overview (embedable in document/paper)
        # todo: always show description place (just empty)

        # Edges
        user >> cli
        agents >> env | "uses"
        writer >> fs | "writes"

    print("Diagram saved to example.svg")


if __name__ == "__main__":
    main()
