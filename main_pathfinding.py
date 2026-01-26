"""Examples demonstrating the pathfinding edge routing feature."""

from specplot import diagram, node, PathfindingConfig


def basic_comparison():
    """Compare traditional routing vs pathfinding on the same diagram."""
    # Traditional routing (default)
    with diagram(filename="output/pathfinding_off"):
        user = node(icon="person", label="User")
        web = node(icon="web", label="Web Server")
        api = node(icon="api", label="API Gateway")
        db = node(icon="database", label="Database")
        cache = node(icon="memory", label="Cache")

        user >> web >> api >> db
        api >> cache >> db

    print("Traditional routing: pathfinding_off.svg")

    # With pathfinding enabled
    with diagram(filename="output/pathfinding_on", pathfinding=True):
        user = node(icon="person", label="User")
        web = node(icon="web", label="Web Server")
        api = node(icon="api", label="API Gateway")
        db = node(icon="database", label="Database")
        cache = node(icon="memory", label="Cache")

        user >> web >> api >> db
        api >> cache >> db

    print("With pathfinding: pathfinding_on.svg")


def orthogonal_paths():
    """Demonstrate orthogonal (right-angle) path style."""
    with diagram(filename="output/orthogonal_paths", pathfinding=True, path_style="orthogonal"):
        # Create a grid-like layout
        frontend = node(icon="web", label="Frontend")

        with node(icon="cloud", label="Backend", show_as="group", grid=(2, 2)):
            auth = node(icon="lock", label="Auth")
            users = node(icon="group", label="Users")
            orders = node(icon="shopping_cart", label="Orders")
            inventory = node(icon="inventory", label="Inventory")

        db = node(icon="database", label="Database")

        frontend >> auth
        frontend >> users
        auth >> db
        users >> db
        orders >> db
        inventory >> db

    print("Orthogonal paths: orthogonal_paths.svg")


def smooth_paths():
    """Demonstrate smooth Bezier curve paths (default style)."""
    with diagram(filename="output/smooth_paths", pathfinding=True, path_style="smooth"):
        frontend = node(icon="web", label="Frontend")

        with node(icon="cloud", label="Backend", show_as="group", grid=(2, 2)):
            auth = node(icon="lock", label="Auth")
            users = node(icon="group", label="Users")
            orders = node(icon="shopping_cart", label="Orders")
            inventory = node(icon="inventory", label="Inventory")

        db = node(icon="database", label="Database")

        frontend >> auth
        frontend >> users
        auth >> db
        users >> db
        orders >> db
        inventory >> db

    print("Smooth paths: smooth_paths.svg")


def custom_config():
    """Demonstrate custom pathfinding configuration."""
    # Tighter grid with higher obstacle avoidance
    config = PathfindingConfig(
        grid_spacing=10.0,  # Finer grid for more precise routing
        proximity_penalty_weight=1.0,  # Strong penalty for paths near nodes
        path_style="smooth",
        simplification_tolerance=3.0,  # Less simplification for smoother curves
    )

    with diagram(filename="output/custom_config", pathfinding=config):
        a = node(icon="web", label="Service A")
        b = node(icon="api", label="Service B")
        c = node(icon="cloud", label="Service C")
        d = node(icon="database", label="Database")
        e = node(icon="memory", label="Cache")

        a >> b >> d
        a >> c >> d
        b >> e
        c >> e

    print("Custom config: custom_config.svg")


def complex_microservices():
    """A more complex microservices architecture with pathfinding."""
    with diagram(
        filename="output/microservices_pathfinding",
        pathfinding=True,
        layout=(("LR",), ("LR", "LR", "LR"), ("LR",))
    ):
        # Row 1: Entry point
        gateway = node(icon="router", label="API Gateway", pos=1)

        # Row 2: Services in three columns
        with node(icon="lock", label="Auth", show_as="outline", pos=2):
            node(label="JWT validation")
            node(label="OAuth2 flows")
            node(label="Session management")

        with node(icon="person", label="User Service", show_as="outline", pos=3):
            node(label="Profile CRUD")
            node(label="Preferences")
            node(label="Notifications")

        with node(icon="shopping_cart", label="Order Service", show_as="outline", pos=4):
            node(label="Cart management")
            node(label="Checkout flow")
            node(label="Order history")

        # Row 3: Data layer
        with node(icon="database", label="Data Layer", show_as="group", grid=(1, 3), pos=5):
            postgres = node(icon="database", label="PostgreSQL")
            redis = node(icon="memory", label="Redis")
            elastic = node(icon="search", label="Elasticsearch")

        # Edges - pathfinding will route around obstacles
        gateway >> node(icon="lock", label="Auth", pos=2)
        gateway >> node(icon="person", label="User Service", pos=3)
        gateway >> node(icon="shopping_cart", label="Order Service", pos=4)

    print("Microservices with pathfinding: microservices_pathfinding.svg")


def nested_groups():
    """Demonstrate pathfinding with nested group structures."""
    with diagram(filename="output/nested_groups_pathfinding", pathfinding=True):
        client = node(icon="devices", label="Client Apps")

        with node(
            icon="cloud",
            label="Cloud Infrastructure",
            description="AWS deployment",
            show_as="group",
            grid=(1, 2)
        ):
            with node(
                icon="dns",
                label="Compute",
                show_as="group",
                grid=(2, 1)
            ):
                lambda_fn = node(icon="function", label="Lambda")
                ecs = node(icon="view_module", label="ECS")

            with node(
                icon="storage",
                label="Storage",
                show_as="group",
                grid=(2, 1)
            ):
                s3 = node(icon="folder", label="S3")
                dynamo = node(icon="database", label="DynamoDB")

        external = node(icon="public", label="External APIs")

        client >> lambda_fn
        client >> ecs
        lambda_fn >> s3
        lambda_fn >> dynamo
        ecs >> dynamo
        lambda_fn >> external | "webhook"

    print("Nested groups with pathfinding: nested_groups_pathfinding.svg")


def outline_connections():
    """Demonstrate pathfinding with outline item connections."""
    with diagram(filename="output/outline_pathfinding", pathfinding=True):
        # Source with outline items
        with node(
            icon="source",
            label="Data Sources",
            description="Input data streams",
            show_as="outline"
        ):
            kafka = node(label="Kafka topics")
            api_input = node(label="REST API")
            files = node(label="S3 files")

        # Processing
        processor = node(
            icon="settings",
            label="Stream Processor",
            description="Apache Flink cluster"
        )

        # Destinations with outline
        with node(
            icon="output",
            label="Destinations",
            description="Output sinks",
            show_as="outline"
        ):
            warehouse = node(label="Data warehouse")
            alerts = node(label="Alert system")
            dashboard = node(label="Real-time dashboard")

        kafka >> processor
        api_input >> processor
        files >> processor
        processor >> warehouse
        processor >> alerts
        processor >> dashboard

    print("Outline connections with pathfinding: outline_pathfinding.svg")


def grid_comparison():
    """Show how different grid spacings affect routing."""
    # Coarse grid (faster, less precise)
    coarse = PathfindingConfig(grid_spacing=25.0, path_style="smooth")
    with diagram(filename="output/grid_coarse", pathfinding=coarse):
        a = node(icon="web", label="A")
        b = node(icon="api", label="B")
        c = node(icon="cloud", label="C")
        d = node(icon="database", label="D")
        a >> b >> d
        a >> c >> d

    print("Coarse grid (25px): grid_coarse.svg")

    # Fine grid (slower, more precise)
    fine = PathfindingConfig(grid_spacing=10.0, path_style="smooth")
    with diagram(filename="output/grid_fine", pathfinding=fine):
        a = node(icon="web", label="A")
        b = node(icon="api", label="B")
        c = node(icon="cloud", label="C")
        d = node(icon="database", label="D")
        a >> b >> d
        a >> c >> d

    print("Fine grid (10px): grid_fine.svg")


def debug_visualization():
    """Demonstrate debug mode showing virtual nodes."""
    # Debug mode renders the pathfinding grid visually:
    # - Gray dots: regular grid nodes (available for routing)
    # - Red dots: blocked nodes (inside obstacles)
    # - Orange dots: boundary nodes (near obstacles)
    # - Green/teal/blue circles: snapping points on node borders
    #   (green = center, blue = edges)

    config = PathfindingConfig(debug=True, grid_spacing=15.0)
    with diagram(filename="output/debug_grid", pathfinding=config):
        a = node(icon="web", label="Frontend")
        b = node(icon="api", label="API Gateway")
        c = node(icon="database", label="Database")
        d = node(icon="memory", label="Cache")

        a >> b >> c
        b >> d >> c

    print("Debug visualization: debug_grid.svg")

    # Debug with orthogonal paths
    config_ortho = PathfindingConfig(debug=True, path_style="orthogonal")
    with diagram(filename="output/debug_orthogonal", pathfinding=config_ortho):
        a = node(icon="web", label="Service A")
        b = node(icon="cloud", label="Service B")
        c = node(icon="api", label="Service C")
        db = node(icon="database", label="Database")

        a >> b >> db
        a >> c >> db

    print("Debug orthogonal: debug_orthogonal.svg")


if __name__ == "__main__":
    print("=== Pathfinding Examples ===\n")

    print("1. Basic comparison (traditional vs pathfinding):")
    basic_comparison()
    print()

    print("2. Path styles:")
    orthogonal_paths()
    smooth_paths()
    print()

    print("3. Custom configuration:")
    custom_config()
    print()

    print("4. Complex architectures:")
    complex_microservices()
    print()

    print("5. Nested groups:")
    nested_groups()
    print()

    print("6. Outline connections:")
    outline_connections()
    print()

    print("7. Grid spacing comparison:")
    grid_comparison()
    print()

    print("8. Debug visualization:")
    debug_visualization()
    print()

    print("=== All examples generated! ===")
