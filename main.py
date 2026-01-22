"""Example usage of specplot."""

from specplot import GROUP, OUTLINE, diagram, node


def main():
    """Create an example architecture diagram."""
    with diagram(filename="example"):
        # User node
        with node(icon="person", label="User") as user:
            pass

        # Cloud environment group
        with node(
            icon="cloud",
            label="Cloud Environment",
            description="On-prem hosted infrastructure",
            show_as=GROUP,
            grid=(2, 1),
        ) as env:
            with node(icon="storage", label="Web Server") as web:
                pass
            with node(icon="database", label="Database", description="hello db") as db:
                pass

        # Agents as outline
        with node(
            icon="hive",
            label="AI Agents",
            description="Our intelligent agents",
            show_as=OUTLINE,
        ) as agents:
            with node(icon="psychology", label="Reader Agent"):
                pass
            with node(icon="psychology", label="Annotator Agent"):
                pass
            with node(icon="psychology", label="Writer Agent") as writer:
                pass

        # Create edges
        user >> web | "HTTPS"
        agents >> env
        writer >> db | "Write"

    print("Diagram saved to example.svg")


if __name__ == "__main__":
    main()
