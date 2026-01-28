# specplot Reference Documentation

## Installation

```bash
pip install git+https://github.com/themerius/specplot.git@main
```

Or with uv:

```bash
uv add git+https://github.com/themerius/specplot.git@main
```

## Core Concepts

### Diagram

The `diagram()` context manager creates a diagram and renders it to SVG on exit.

```python
from specplot import diagram, node

with diagram(filename="my_diagram"):
    # nodes and edges here
```

**Parameters:**
- `filename` — Output path (without `.svg` extension)
- `title` — Optional diagram title
- `layout` — Zone layout tuple (see Layout section)
- `pathfinding` — Enable smart edge routing (`True`, `False`, or `PathfindingConfig`)

### Node

Nodes are the building blocks. Use `node()` to create them.

```python
# Simple node
db = node(icon="database", label="PostgreSQL")

# Node with description
api = node(icon="api", label="API Gateway",
          description="Handles auth and routing")

# Node with children (group mode)
with node(icon="cloud", label="Services", show_as="group"):
    node(icon="web", label="Web")
    node(icon="api", label="API")

# Node with children (outline mode)
with node(icon="list", label="Components", show_as="outline"):
    node(label="Authentication")
    node(label="Authorization")
    node(label="Logging")
```

**Parameters:**
- `icon` — Material Symbols icon name
- `label` — Display text
- `description` — Optional description (2 lines max)
- `show_as` — Child display mode: `"group"` or `"outline"`
- `grid` — Grid layout for group children: `(rows, cols)`
- `pos` — Zone position for zone-based layouts

### ShowAs Modes

**`show_as="group"`** — Children rendered as separate boxes inside the parent:

```python
with node(label="Backend", show_as="group", grid=(1, 3)):
    node(icon="api", label="API")
    node(icon="dns", label="Service")
    node(icon="database", label="DB")
```

**`show_as="outline"`** — Children rendered as a bullet list:

```python
with node(label="Requirements", show_as="outline"):
    node(label="User authentication")
    node(label="Data validation")
    node(label="Error handling")
```

### Edges

Connect nodes with the `>>` operator:

```python
user = node(icon="person", label="User")
api = node(icon="api", label="API")
db = node(icon="database", label="Database")

user >> api >> db  # Chain connections
```

Add labels with `|`:

```python
api >> db | "SQL queries"
```

### Edge Styles

**Operator syntax** (common cases):

```python
a >> b              # Arrow right: A ──→ B
b << a              # Arrow left:  A ←── B
a >> b | "label"    # With label:  A ──→ B (label)
```

**Explicit `edge()` function** (all styles):

```python
from specplot import edge

edge(a, b, style="->")              # Arrow right (default)
edge(a, b, style="<-")              # Arrow left
edge(a, b, style="--")              # Line (no arrow)
edge(a, b, style="..")              # Dotted line
edge(a, b, style="..>")             # Dotted arrow right
edge(a, b, style="<..")             # Dotted arrow left
edge(a, b, style="->", label="HTTP") # With label
```

**Style reference:**

| Style | Code | Result |
|-------|------|--------|
| Arrow right | `a >> b` or `style="->"` | `───→` |
| Arrow left | `a << b` or `style="<-"` | `←───` |
| Line | `style="--"` | `────` |
| Dotted | `style=".."` | `┈┈┈┈` |
| Dotted arrow right | `style="..>"` | `┈┈┈→` |
| Dotted arrow left | `style="<.."` | `←┈┈┈` |

### Layout

#### Default Layout

Without explicit layout, nodes flow left-to-right.

#### Zone Layout

Define zones for complex layouts:

```python
with diagram(filename="zones", layout=(
    ("LR",),           # Row 1: single left-to-right zone
    ("TB", "TB", "TB"), # Row 2: three top-to-bottom zones
    ("LR",),           # Row 3: single left-to-right zone
)):
    node(icon="person", label="User", pos=1)
    node(icon="api", label="API", pos=2)
    node(icon="database", label="DB", pos=4)
```

### Pathfinding

Intelligent edge routing is **enabled by default**. Edges automatically find paths around obstacles.

To disable pathfinding (use simple bezier curves):

```python
with diagram(filename="simple", pathfinding=False):
    # edges use direct bezier curves
```

Custom configuration:

```python
from specplot import PathfindingConfig

config = PathfindingConfig(
    enabled=True,
    grid_spacing=15.0,
    path_style="smooth",  # or "orthogonal"
    debug=False,
)

with diagram(filename="custom", pathfinding=config):
    ...
```

## Icons

specplot uses [Material Symbols](https://fonts.google.com/icons). Common icons:

| Icon | Name |
|------|------|
| 👤 | `person` |
| 🗄️ | `database` |
| ☁️ | `cloud` |
| 🌐 | `web` |
| 📡 | `api` |
| 📦 | `inventory_2` |
| ⚙️ | `settings` |
| 🔒 | `lock` |
| 📊 | `analytics` |
| 💾 | `storage` |

Browse all icons at [fonts.google.com/icons](https://fonts.google.com/icons).

## Complete Example

```python
from specplot import diagram, node

with diagram(filename="architecture"):
    # Users
    user = node(icon="person", label="User")
    admin = node(icon="admin_panel_settings", label="Admin")

    # Application layer
    with node(icon="cloud", label="Application",
             show_as="group", grid=(1, 2)):
        web = node(icon="web", label="Web App")
        api = node(icon="api", label="REST API")

    # Services
    with node(icon="dns", label="Services", show_as="outline"):
        node(label="Authentication")
        node(label="Authorization")
        node(label="Business Logic")

    # Data layer
    db = node(icon="database", label="PostgreSQL")
    cache = node(icon="memory", label="Redis")

    # Connections
    user >> web
    admin >> api
    web >> api >> db
    api >> cache | "session"
```

## API Reference

### specplot module

```python
from specplot import diagram, node, edge, PathfindingConfig
```

### diagram(filename, title=None, layout=None, pathfinding=True)

Context manager that creates and renders a diagram.

### node(icon=None, label="", description=None, show_as="outline", grid=None, pos=None)

Creates a node. Returns `NodeContext` that can be used with or without `with` statement.

### edge(source, target, style=EdgeStyle.ARROW_RIGHT, label=None)

Explicit edge creation (alternative to `>>` operator).

### PathfindingConfig

Configuration for edge routing:
- `enabled: bool` — Toggle pathfinding
- `grid_spacing: float` — Grid cell size (default: 15.0)
- `path_style: str` — `"smooth"` or `"orthogonal"`
- `debug: bool` — Render grid visualization
