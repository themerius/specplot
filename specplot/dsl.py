"""Python DSL for creating diagrams."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from .models import Diagram, Edge, EdgeStyle, Node, OutlineItem, ShowAs
from .pathfinding import PathfindingConfig
from .renderer import render_to_svg

if TYPE_CHECKING:
    from collections.abc import Generator

# Type alias for show_as parameter
ShowAsLiteral = Literal["group", "outline"]

# Context stack for nested node creation
_diagram_stack: list[Diagram] = []
_node_stack: list[Node] = []


def _current_diagram() -> Diagram | None:
    """Get the current diagram context."""
    return _diagram_stack[-1] if _diagram_stack else None


def _current_parent() -> Node | None:
    """Get the current parent node context."""
    return _node_stack[-1] if _node_stack else None


def _parse_show_as(value: ShowAsLiteral | ShowAs) -> ShowAs:
    """Convert string literal to ShowAs enum."""
    if isinstance(value, ShowAs):
        return value
    if value == "group":
        return ShowAs.GROUP
    return ShowAs.OUTLINE


@contextmanager
def diagram(
        filename: str = "diagram",
        title: str | None = None,
        layout: tuple[tuple[str, ...], ...] | None = None,
        pathfinding: bool | PathfindingConfig = True,
        path_style: Literal["smooth", "orthogonal"] = "smooth",
        **kwargs: Any,
) -> Generator[Diagram]:
    """Create a diagram context.

    Usage:
        with diagram(filename="my_diagram"):
            user = node(icon="person", label="User")
            db = node(icon="database", label="Database")
            user >> db

        # With zone layout:
        with diagram(filename="zones", layout=(("LR",), ("TB", "TB"), ("LR",))):
            node(icon="person", label="User", pos=1)
            node(icon="api", label="API", pos=2)
            node(icon="database", label="DB", pos=3)

        # With pathfinding enabled:
        with diagram(filename="smart", pathfinding=True):
            a = node(icon="web", label="A")
            b = node(icon="database", label="B")
            a >> b

        # With custom pathfinding config:
        config = PathfindingConfig(grid_spacing=20, path_style="orthogonal")
        with diagram(filename="custom", pathfinding=config):
            ...

    Args:
        filename: Output filename (without extension)
        title: Optional diagram title
        layout: Zone layout as tuple of tuples, e.g., (("LR",), ("TB", "TB"), ("LR",))
        pathfinding: Enable intelligent edge routing (True, False, or PathfindingConfig)
        path_style: Path style when pathfinding=True ("smooth" or "orthogonal")
        **kwargs: Additional diagram options

    Yields:
        The Diagram object
    """
    # Handle pathfinding configuration
    pathfinding_config = None
    if pathfinding is True:
        pathfinding_config = PathfindingConfig(enabled=True, path_style=path_style)
    elif isinstance(pathfinding, PathfindingConfig):
        pathfinding_config = pathfinding

    d = Diagram(
        filename=filename,
        title=title,
        layout=layout,
        pathfinding_config=pathfinding_config,
        **kwargs,
    )
    _diagram_stack.append(d)

    try:
        yield d
    finally:
        _diagram_stack.pop()
        # Render on exit
        render_to_svg(d, filename)


class NodeContext:
    """A node that can be used with or without context manager.

    Usage:
        # Without context manager (no children)
        db = node(icon="database", label="Database")

        # With context manager (has children)
        with node(icon="cloud", label="Cloud", show_as="group") as cloud:
            web = node(icon="web", label="Web")
    """

    def __init__(self, node: Node):
        self._node = node
        self._entered = False

    def __enter__(self) -> Node:
        """Enter context manager - push node onto stack for children."""
        _node_stack.append(self._node)
        self._entered = True
        return self._node

    def __exit__(self, *args: Any) -> None:
        """Exit context manager - pop node from stack."""
        if self._entered:
            _node_stack.pop()

    # Delegate all attribute access to the underlying node
    def __getattr__(self, name: str) -> Any:
        return getattr(self._node, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._node, name, value)

    # Support >> operator for edges
    def __rshift__(self, other: Node | NodeContext | OutlineItem) -> Edge:
        target = other._node if isinstance(other, NodeContext) else other
        return self._node >> target

    def __lshift__(self, other: Node | NodeContext | OutlineItem) -> Edge:
        target = other._node if isinstance(other, NodeContext) else other
        return self._node << target

    def __repr__(self) -> str:
        return repr(self._node)


def node(
        icon: str | None = "draft",
        label: str = "",
        description: str | None = None,
        show_as: ShowAsLiteral | ShowAs = "outline",
        grid: tuple[int, int] | None = None,
        pos: int | None = None,
        **kwargs: Any,
) -> NodeContext:
    """Create a node, optionally as a context for child nodes.

    Can be used with or without context manager:

        # Simple node (no children)
        db = node(icon="database", label="Database")

        # Node with children - use 'with' syntax
        with node(icon="cloud", label="Cloud", show_as="group") as cloud:
            web = node(icon="web", label="Web Server")
            api = node(icon="api", label="API")

        # Outline mode (default)
        with node(icon="api", label="Agents", show_as="outline") as agents:
            node(icon="robot", label="Reader")
            writer = node(icon="robot", label="Writer")

        # With zone position
        node(icon="person", label="User", pos=1)

    Args:
        icon: Material Symbols icon name
        label: Node label text
        description: Optional description text
        show_as: "group" or "outline" for child display mode
        grid: Grid layout as (rows, cols) tuple
        pos: Zone position (1-indexed) for zone-based layouts
        **kwargs: Additional node options

    Returns:
        NodeContext that can be used with or without 'with' statement
    """
    n = Node(
        icon=icon,
        label=label,
        description=description,
        show_as=_parse_show_as(show_as),
        grid=grid,
        pos=pos,
        **kwargs,
    )

    current_diag = _current_diagram()
    current_parent = _current_parent()

    if current_parent:
        # Nested node
        n._parent = current_parent
        n._diagram = current_diag
        current_parent.children.append(n)
    elif current_diag:
        # Top-level node
        n._diagram = current_diag
        current_diag.nodes.append(n)

    return NodeContext(n)


def edge(
        source: Node | NodeContext | OutlineItem,
        target: Node | NodeContext | OutlineItem,
        style: EdgeStyle | str = EdgeStyle.ARROW_RIGHT,
        label: str | None = None,
) -> Edge:
    """Create an edge between two nodes or outline items.

    Usage:
        edge(user, web, style="->", label="HTTP")
        edge(api, db, style="--")  # undirected

    Args:
        source: Source node or outline item
        target: Target node or outline item
        style: Edge style ("->", "<-", "--", "..", "..>", "<..")
        label: Optional edge label

    Returns:
        The Edge object
    """
    # Unwrap NodeContext if needed
    if isinstance(source, NodeContext):
        source = source._node
    if isinstance(target, NodeContext):
        target = target._node

    # Convert string style to enum
    if isinstance(style, str):
        style_map = {
            "->": EdgeStyle.ARROW_RIGHT,
            "<-": EdgeStyle.ARROW_LEFT,
            "--": EdgeStyle.LINE,
            "..": EdgeStyle.DOTTED,
            "..>": EdgeStyle.DOTTED_ARROW_RIGHT,
            "<..": EdgeStyle.DOTTED_ARROW_LEFT,
        }
        style = style_map.get(style, EdgeStyle.ARROW_RIGHT)

    e = Edge(source=source, target=target, style=style, label=label)

    # Add to current diagram
    diag = _current_diagram()
    if diag:
        diag.edges.append(e)
        e._diagram = diag

    return e


# Keep backward compatibility with enum constants
GROUP = ShowAs.GROUP
OUTLINE = ShowAs.OUTLINE
