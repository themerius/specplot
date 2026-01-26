# specplot

Architecture diagramming library with Python DSL that outputs SVG.

## Project Structure

```
specplot/
├── __init__.py      # Public API exports
├── models.py        # Data models (Node, Edge, Diagram, ShowAs, EdgeStyle)
├── icons.py         # Material Symbols icon fetcher with caching
├── layout.py        # Layout algorithms, fixed-width nodes, grid positioning
├── renderer.py      # SVG rendering with drawsvg, themes, edge curves
├── pathfinding.py   # A* grid-based edge routing with obstacle avoidance
└── dsl.py           # Python DSL (context managers, >> operator for edges)
```

## Key Concepts

- **Nodes**: Fixed-width boxes with icon, label, optional description
- **ShowAs modes**: `"group"` (nested child nodes) or `"outline"` (bullet list) - use string literals
- **Edges**: Connect nodes with `>>` operator, labels with `| "text"`
- **Grid layout**: `grid=(rows, cols)` for arranging children
- **DSL flexibility**: Context managers optional for nodes without children (`node(...)` vs `with node(...):`)

## Running

```bash
uv run python main.py  # generates example.svg
```

## Design Decisions

- Fixed node width (200px) for consistent appearance
- **Descriptions**: Fixed 2-line box (38px height) with word-boundary-aware wrapping
- **Text truncation**: Labels 20 chars, outline items 24 chars; descriptions wrap to 2 lines then truncate
- **Text width estimation**: Uses `char_width_avg` (4.8px) for 11px sans-serif font
- Edges from OUTLINE children connect from parent node's edge at the item's y-position
- OUTLINE parent nodes connect edges from header area (not center)
- Smooth bezier curves with horizontal exit/entry (`control_offset = max(50, dx * 0.5)`)
- Material Symbols icons fetched from Google CDN, cached locally (coordinate system detection for transform)
- Edges rendered on top of nodes (arrowheads visible)

## SVG Limitations

- No native text-overflow with ellipsis - must pre-calculate truncation
- No native text wrapping - manually split into multiple `<text>` elements
- `foreignObject` with HTML/CSS works in browsers but fails in vector editors

## Pathfinding System

### Overview

The pathfinding system (`pathfinding.py`) routes edges around obstacles using A* search on a virtual grid. It replaces simple bezier curves with intelligent paths that avoid crossing through nodes.

### Architecture

1. **VirtualGrid**: NetworkX graph overlaid on the diagram
   - Grid spacing configurable (default 15px)
   - Nodes mark cells as obstacles with configurable margin
   - Edges connect to "snapping points" on node borders

2. **Snapping Points**: Connection points on node boundaries
   - Distributed along each side (left, right, top, bottom)
   - Gaussian-weighted preference for center positions
   - Support `no_entry` flag for restricted areas (e.g., group header zones on right side)

3. **EdgeRouter** (`renderer.py`): Orchestrates edge routing
   - Groups edges by node+side for distributed point assignment
   - Uses A* to find paths between snapping points
   - Applies Douglas-Peucker simplification for smooth rendering

### Key Implementation Details

#### Snapping Point Generation (`_compute_snapping_points`)

For each node, snapping points are created on all four sides:
- **Left/Bottom**: Fully available for connections
- **Right**: Points in header zone of GROUP nodes marked `no_entry=True`
- **Top**: Available for all nodes (header zone protected by obstacle grid)

Points use Gaussian weighting (`sigma = side_length / 6`) to prefer center positions.

#### Obstacle Marking

Nodes are marked as obstacles in the grid with margins:
- Regular nodes: Full bounds + margin
- GROUP nodes: Header zone marked separately as obstacle
- OUTLINE nodes: Treated as regular nodes (children are list items, not boxes)

#### Path Finding

A* search with custom heuristics:
- `diagonal_penalty`: Prefers orthogonal movement (default 1.5x)
- `turn_penalty`: Discourages direction changes (default 2.0)
- `proximity_penalty_weight`: Penalizes paths near obstacles

#### Path Styles

- **smooth**: Douglas-Peucker simplification + rounded corners at waypoints
- **orthogonal**: Strict horizontal/vertical segments only

### NodeContext Unwrapping

The DSL returns `NodeContext` wrappers from `node()`. When creating edges:
- Context manager (`with node(...) as x`) returns the actual `Node`
- Direct call (`x = node(...)`) returns `NodeContext`

`Node.__rshift__`, `__lshift__`, `__sub__` use duck-typing to unwrap:
```python
if hasattr(other, '_node'):
    other = other._node
```

### Configuration

```python
PathfindingConfig(
    enabled=True,              # Toggle pathfinding
    grid_spacing=15.0,         # Grid cell size
    path_style="smooth",       # "smooth" or "orthogonal"
    diagonal_penalty=1.5,      # Prefer orthogonal paths
    turn_penalty=2.0,          # Prefer straight paths
    node_margin=1.0,           # Obstacle margin (grid cells)
    layout_spacing_multiplier=2.0,  # Extra space for routing
    debug=False,               # Render grid points
)
```

### Debug Mode

Set `debug=True` to render virtual grid points in the SVG:
- Blue dots: Regular grid points
- Red dots: Blocked/obstacle points
- Green dots: Snapping points on node borders
