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
            label="Cloud Environment",
            description="On-prem hosted infrastructure",
            show_as="group",
            grid=(2, 1),
        ) as env:
            web = node(icon="storage", label="Web Server")
            db = node(icon="database", label="Database", description="hello db")

        # Outline mode (default) - use 'with' for children
        with node(
            icon="hive",
            label="AI Agents",
            description="Our intelligent agents",
        ) as agents:
            node(icon="psychology", label="Reader Agent")
            node(icon="psychology", label="Annotator Agent")
            writer = node(icon="psychology", label="Writer Agent")

        # Edges
        user >> web | "HTTPS"
        agents >> env
        writer >> db | "Write"

    print("Diagram saved to example.svg")


if __name__ == "__main__":
    main()
