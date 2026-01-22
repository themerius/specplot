# specplot

Architecture diagramming tool with Python DSL. Generate professional SVG diagrams for documentation, papers, and presentations.

## Installation

```bash
uv add specplot
```

## Quick Start

```python
from specplot import diagram, node, GROUP, OUTLINE

with diagram(filename="architecture"):
    # Create a simple node
    user = node(icon="person", label="User")

    # Create a group with nested nodes
    with node(icon="cloud", label="Cloud", show_as=GROUP, grid=(1, 2)) as cloud:
        db = node(icon="database", label="Database")
        web = node(icon="dns", label="Web Server")

    # Create edges with labels
    user >> web | "HTTPS"
    web >> db | "SQL"
```

## Features

### Nodes

Nodes are the basic building blocks. Each node has:
- **icon**: Material Symbols icon name (e.g., "database", "person", "cloud")
- **label**: Main text displayed in the header
- **description**: Optional smaller text below the header (auto-truncated)
- **show_as**: How to display children - `GROUP` or `OUTLINE`
- **grid**: Layout grid for GROUP mode as `(rows, cols)` tuple

```python
# Simple node
db = node(icon="database", label="PostgreSQL", description="Primary database")

# Node with grouped children
with node(icon="cloud", label="AWS", show_as=GROUP, grid=(2, 2)) as aws:
    node(icon="storage", label="S3")
    node(icon="dns", label="Route53")
    node(icon="security", label="IAM")
    node(icon="computer", label="EC2")

# Node with outline children (bullet list)
with node(icon="api", label="API", show_as=OUTLINE) as api:
    node(label="GET /users")
    node(label="POST /users")
    node(label="DELETE /users/:id")
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
from specplot import diagram, node, GROUP, OUTLINE

with diagram(filename="system"):
    user = node(icon="person", label="User")

    with node(
        icon="cloud",
        label="Cloud Environment",
        description="On-prem hosted",
        show_as=GROUP,
        grid=(1, 2),
    ) as env:
        db = node(icon="storage", label="Database")
        web = node(icon="dns", label="Web Server")

    with node(
        icon="smart_toy",
        label="AI Agents",
        description="Intelligent assistants",
        show_as=OUTLINE,
    ) as agents:
        reader = node(icon="psychology", label="Reader Agent")
        writer = node(icon="psychology", label="Writer Agent")

    user >> web | "HTTPS"
    agents >> env
    writer >> db | "Write"
```

## API Reference

### `diagram(filename, title=None, **kwargs)`
Context manager that creates and renders a diagram.

### `node(icon, label, description=None, show_as=OUTLINE, grid=None)`
Context manager or function that creates a node. Returns the Node object.

### `edge(source, target, style="->", label=None)`
Explicitly create an edge between nodes.

### Constants
- `GROUP` - Display children as nested nodes
- `OUTLINE` - Display children as bullet list

## License

MIT
