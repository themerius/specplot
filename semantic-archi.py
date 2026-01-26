"""Example usage of specplot."""

from specplot import diagram, node, PathfindingConfig


def print_diagrams():

    with diagram(
        filename="semantic-auth-flow",
        layout=(("LR",), ("LR",), ("LR",)),
        pathfinding=True
    ):
        # pos=1: top LR zone
        user = node(icon="person", label="User", pos=1)
        swarm = node(icon="hive", label="Agent Swarm", pos=1)

        # pos=2,3,4: middle TB columns
        iap = node(icon="identity_platform", label="IaP", description="Using Authelia as identity-aware proxy.", pos=2)

        with node(icon="home", label="Semantic Studio", grid=(3, 1), pos=3) as studio:
            node(icon="dashboard", label="UI")
            node(icon="api", label="API")
            node(icon="database", label="SDA Store")


        # Edges
        user >> iap | "oauth flow"
        swarm >> iap | "api key"
        iap >> studio | "trusted header sso"

    print("Generated aufh flow")


    config = PathfindingConfig(debug=False)
    with diagram(
        filename="semantic-data-flow",
        layout=(("LR",), ("LR",)),
        pathfinding=config
    ):

        user = node(icon="person", label="User", pos=1)
        it = node(icon="person", label="IT Support", pos=1)

        with node(icon="home", label="Semantic Studio", grid=(4, 1), show_as="group", pos=2) as studio:
            ui = node(icon="dashboard", label="UI")
            cron = node(icon="schedule", label="Event scheduler")
            api = node(icon="api", label="API")
            sdas = node(icon="database", label="SDA Store")
            ui >> api >> sdas
            cron >> api | "request job"

        with node(icon="hive", label="Agent Swarm", grid=(3, 1), show_as="group", pos=2) as swarm:
            with node(label="Service", show_as="outline") as svc:
                registry = node(label="Registry")
                node(label="KPI Reporter")
                hello = node(label="Hello", description="reports health status, kpi report, and check for jobs")
            agents = node(icon="cognition", label="Agents", description="Micro-Agent pipline e.g. invoice document processing")
            # agents add artifact download in exchange to a valid license code (e.g. our gcp oauth json)

        user >> studio | "request job"
        it >> swarm | "deploy"

        registry >> api | "register"
        hello >> api | "ack"
        agents >> api | "get/push data"

        svc >> agents | "initiate"

        print("generated data flow")


    with diagram(
        filename="semantic-swarm-architecture",
        layout=(("LR",), ("LR",)),
        pathfinding=config
    ):
        print("generated swarm architecture")


    with diagram(
        filename="semantic-aware-dsl",
        layout=(("LR",), ("LR",)),
        pathfinding=config
    ):
        print("generated aware architecture")



if __name__ == "__main__":
    print_diagrams()

