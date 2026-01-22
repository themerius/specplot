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
- **ShowAs modes**: `GROUP` (nested child nodes) or `OUTLINE` (bullet list)
- **Edges**: Connect nodes with `>>` operator, labels with `| "text"`
- **Grid layout**: `grid=(rows, cols)` for arranging children

## Running

```bash
uv run python main.py  # generates example.svg
```

## Design Decisions

- Fixed node width (200px) for consistent appearance
- Text truncation: labels 20 chars, descriptions 28 chars, outline items 24 chars
- Edges from OUTLINE children connect from parent node
- Smooth bezier curves with horizontal exit/entry
- Material Symbols icons fetched from Google CDN, cached locally
- Edges rendered on top of nodes (arrowheads visible)
