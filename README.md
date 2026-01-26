# specplot

Architecture diagramming tool with Python DSL. Generate professional SVG diagrams for documentation, papers, and presentations.

## Installation

```bash
uv add specplot
```

## Quick Start

```python
from specplot import diagram, node

with diagram(filename="architecture"):
    # Simple node - no 'with' needed
    user = node(icon="person", label="User")

    # Group with nested nodes
    with node(icon="cloud", label="Cloud", show_as="group", grid=(1, 2)):
        db = node(icon="database", label="Database")
        web = node(icon="dns", label="Web Server")

    # Edges with labels
    user >> web | "HTTPS"
    web >> db | "SQL"
```

## Features

### Nodes

Nodes are the basic building blocks. Each node has:
- **icon**: Material Symbols icon name (e.g., "database", "person", "cloud")
- **label**: Main text displayed in the header
- **description**: Optional smaller text below the header (auto-truncated)
- **show_as**: `"group"` or `"outline"` for child display mode
- **grid**: Layout grid for group mode as `(rows, cols)` tuple

```python
# Simple node - no context manager needed
db = node(icon="database", label="PostgreSQL", description="Primary database")

# Node with grouped children - use 'with' for nesting
with node(icon="cloud", label="AWS", show_as="group", grid=(2, 2)):
    node(icon="storage", label="S3")
    node(icon="dns", label="Route53")
    node(icon="security", label="IAM")
    node(icon="computer", label="EC2")

# Node with outline children (bullet list) - default mode
with node(icon="api", label="API") as api:
    node(label="GET /users")
    node(label="POST /users")
    delete = node(label="DELETE /users/:id")
```

### Edges

Connect nodes with edges using the `>>` operator:

```python
user >> web           # Arrow from user to web
web >> db | "SQL"     # Arrow with label
```

Edge styles:
- `>>` - Right arrow (->)
- `<<` - Left arrow (<-)
- Use `edge()` function for more control

### Icons

Uses [Material Symbols](https://fonts.google.com/icons) icon names. Common icons:
- `person`, `group` - Users
- `database`, `storage` - Data
- `cloud`, `dns`, `computer` - Infrastructure
- `api`, `code`, `terminal` - Development
- `security`, `lock`, `vpn_key` - Security
- `smart_toy`, `psychology` - AI/ML

### Smart Edge Routing

Enable intelligent edge routing that automatically finds paths around obstacles:

```python
from specplot import diagram, node, PathfindingConfig

# Enable pathfinding with default settings
with diagram(filename="routed", pathfinding=True):
    ...

# Or customize the routing behavior
config = PathfindingConfig(
    path_style="smooth",      # "smooth" (curved) or "orthogonal" (right angles)
    diagonal_penalty=1.5,     # Higher = prefer horizontal/vertical paths
    turn_penalty=2.0,         # Higher = prefer straighter paths
)

with diagram(filename="custom-routing", pathfinding=config):
    user = node(icon="person", label="User")

    with node(icon="cloud", label="Backend", show_as="group", grid=(2, 1)):
        api = node(icon="api", label="API")
        db = node(icon="database", label="Database")
        api >> db

    user >> api  # Edge automatically routes around the group boundary
```

Without pathfinding, edges use simple bezier curves that may cross through nodes. With pathfinding enabled, edges intelligently navigate around obstacles, creating cleaner diagrams.

### Themes

Customize colors with a theme:

```python
from specplot import Theme, DiagramRenderer

theme = Theme(
    background="#1e1e1e",
    node_fill="#2d2d2d",
    text_color="#ffffff",
    edge_color="#888888",
)

renderer = DiagramRenderer(theme=theme)
```

## Example

```python
from specplot import diagram, node

with diagram(filename="system"):
    user = node(icon="person", label="User")

    with node(
        icon="cloud",
        label="Cloud Environment",
        description="On-prem hosted",
        show_as="group",
        grid=(1, 2),
    ) as env:
        db = node(icon="storage", label="Database")
        web = node(icon="dns", label="Web Server")

    with node(
        icon="smart_toy",
        label="AI Agents",
        description="Intelligent assistants",
    ) as agents:
        node(icon="psychology", label="Reader Agent")
        writer = node(icon="psychology", label="Writer Agent")

    user >> web | "HTTPS"
    agents >> env
    writer >> db | "Write"
```

## API Reference

### `diagram(filename, title=None, **kwargs)`
Context manager that creates and renders a diagram.

### `node(icon, label, description=None, show_as="outline", grid=None)`
Creates a node. Can be used with or without context manager:
- Without `with`: for simple nodes with no children
- With `with`: when the node has nested children

### `edge(source, target, style="->", label=None)`
Explicitly create an edge between nodes.

### show_as values
- `"group"` - Display children as nested node boxes
- `"outline"` - Display children as bullet list (default)

## License

MIT
