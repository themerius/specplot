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
