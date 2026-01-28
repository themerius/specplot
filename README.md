# specplot

> Think in outlines. Ship as diagrams.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![Hero Example](docs/hero.svg)

## What is specplot?

A Python library that transforms outline-based thinking into professional architecture diagrams. Write your system design as structured code, render it as publication-ready SVG.

```python
from specplot import diagram, node

with diagram(filename="architecture"):
    user = node(icon="person", label="User")

    with node(icon="cloud", label="Web App", show_as="group", grid=(1, 2)):
        frontend = node(icon="web", label="Frontend")
        api = node(icon="api", label="API")

    db = node(icon="database", label="PostgreSQL")

    user >> frontend >> api >> db
```

**Key Features:**

- **Outline-first** — `show_as="outline"` for bullet lists, `show_as="group"` for nested boxes
- **Pythonic DSL** — Context managers, `>>` for edges, `|` for labels
- **Publication-ready** — Clean SVG output for papers, docs, presentations
- **Smart routing** — A* pathfinding routes edges around obstacles
- **Material icons** — 2000+ icons from Google Material Symbols

## Quick Start

```bash
pip install specplot
```

```python
from specplot import diagram, node

with diagram(filename="my_system"):
    user = node(icon="person", label="User")
    api = node(icon="api", label="API")
    db = node(icon="database", label="Database")
    user >> api >> db | "SQL"
```

Run your script — SVG appears in the current directory.

## Examples

### Microservices Architecture

Map your distributed system with groups and grids.

![Microservices Example](docs/example_microservices.svg)

<details>
<summary>View code</summary>

```python
from specplot import diagram, node

with diagram(filename="microservices"):
    web = node(icon="web", label="Web Client")
    mobile = node(icon="smartphone", label="Mobile App")

    gateway = node(icon="api", label="API Gateway",
                  description="Auth, rate limiting, routing")

    with node(icon="cloud", label="Services", show_as="group", grid=(2, 3)):
        users = node(icon="person", label="User Service")
        products = node(icon="inventory_2", label="Product Service")
        orders = node(icon="receipt_long", label="Order Service")
        payments = node(icon="payments", label="Payment Service")
        inventory = node(icon="warehouse", label="Inventory Service")
        notify = node(icon="notifications", label="Notification Service")

    with node(icon="storage", label="Data Layer", show_as="group", grid=(1, 3)):
        userdb = node(icon="database", label="Users DB")
        productdb = node(icon="database", label="Products DB")
        orderdb = node(icon="database", label="Orders DB")

    queue = node(icon="sync_alt", label="Message Queue",
                description="Async event processing")

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
```

</details>

### Layered Architecture

Express clean architecture with outline mode.

![Layered Example](docs/example_layered.svg)

<details>
<summary>View code</summary>

```python
from specplot import diagram, node

with diagram(filename="layered"):
    with node(icon="web", label="Presentation",
             description="UI components and controllers",
             show_as="outline"):
        node(label="React Components")
        node(label="REST Controllers")
        node(label="GraphQL Resolvers")

    with node(icon="account_tree", label="Application",
             description="Use cases and orchestration",
             show_as="outline") as app:
        node(label="Command Handlers")
        node(label="Query Handlers")
        node(label="Event Handlers")

    with node(icon="hub", label="Domain",
             description="Business logic and rules",
             show_as="outline") as domain:
        node(label="Entities")
        node(label="Value Objects")
        node(label="Domain Services")
        node(label="Repository Interfaces")

    with node(icon="dns", label="Infrastructure",
             description="External concerns",
             show_as="outline") as infra:
        node(label="Database Repositories")
        node(label="External API Clients")
        node(label="Message Brokers")
        node(label="Caching")

    app >> domain | "depends on"
    infra >> domain | "implements"
```

</details>

### Data Pipeline

Visualize data flows with mixed display modes.

![Pipeline Example](docs/example_pipeline.svg)

<details>
<summary>View code</summary>

```python
from specplot import diagram, node

with diagram(filename="pipeline"):
    with node(icon="source", label="Data Sources", show_as="group", grid=(1, 3)):
        api_src = node(icon="api", label="REST APIs")
        db_src = node(icon="database", label="Databases")
        files = node(icon="folder", label="File Storage")

    ingest = node(icon="input", label="Ingestion Layer",
                 description="Apache Kafka / Kinesis")

    with node(icon="memory", label="Processing",
             description="Spark / Flink jobs",
             show_as="outline") as proc:
        node(label="Validation & Cleaning")
        node(label="Feature Engineering")
        node(label="Aggregations")

    lake = node(icon="waves", label="Data Lake",
               description="S3 / Delta Lake")

    with node(icon="psychology", label="ML Pipeline", show_as="group", grid=(1, 2)):
        train = node(icon="model_training", label="Training")
        serve = node(icon="cloud_upload", label="Model Serving")

    with node(icon="analytics", label="Outputs", show_as="group", grid=(1, 2)):
        dash = node(icon="dashboard", label="Dashboards")
        alerts = node(icon="notifications", label="Alerts")

    api_src >> ingest
    db_src >> ingest
    files >> ingest
    ingest >> proc | "stream"
    proc >> lake | "batch"
    lake >> train | "features"
    train >> serve | "deploy"
    serve >> dash | "predictions"
    serve >> alerts | "anomalies"
```

</details>

## Why specplot?

- **Outline-first design** — Structure architecture the way you think: hierarchical, iterative, refactorable
- **Code is the diagram** — Version control, automation, and seamless AI collaboration
- **LLM-native** — Both Python input and SVG output are text formats AI can read, write, and reason about
- **Publication-ready** — Clean vector graphics for papers, docs, and high-stakes presentations

[Read the full philosophy →](docs/PHILOSOPHY.md)

[Compare with alternatives →](docs/COMPARISON.md)

## Documentation

See [docs/README.md](docs/README.md) for the full reference documentation.

## Support

If specplot helps your work, consider:

- Starring this repository
- Citing it in your publications

### Citation

```bibtex
@software{specplot,
  author = {Hodapp, Sven},
  title = {specplot: Outline-based architecture diagrams},
  year = {2026},
  url = {https://github.com/themerius/specplot}
}
```

## License

[MIT](LICENSE)
