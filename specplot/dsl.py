"""Python DSL for creating diagrams."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from .models import Diagram, Edge, EdgeStyle, Node, OutlineItem, ShowAs
from .renderer import render_to_svg

if TYPE_CHECKING:
    from collections.abc import Generator

# Display mode constants
GROUP = ShowAs.GROUP
OUTLINE = ShowAs.OUTLINE

# Context stack for nested node creation
_diagram_stack: list[Diagram] = []
_node_stack: list[Node] = []


def _current_diagram() -> Diagram | None:
    """Get the current diagram context."""
    return _diagram_stack[-1] if _diagram_stack else None


def _current_parent() -> Node | None:
    """Get the current parent node context."""
    return _node_stack[-1] if _node_stack else None


@contextmanager
def diagram(
    filename: str = "diagram",
    title: str | None = None,
    **kwargs: Any,
) -> Generator[Diagram]:
    """Create a diagram context.

    Usage:
        with diagram(filename="my_diagram") as d:
            # Create nodes and edges here
            pass

    Args:
        filename: Output filename (without extension)
        title: Optional diagram title
        **kwargs: Additional diagram options

    Yields:
        The Diagram object
    """
    d = Diagram(filename=filename, title=title, **kwargs)
    _diagram_stack.append(d)

    try:
        yield d
    finally:
        _diagram_stack.pop()
        # Render on exit
        render_to_svg(d, filename)


@contextmanager
def node(
    icon: str | None = None,
    label: str = "",
    description: str | None = None,
    show_as: ShowAs = ShowAs.OUTLINE,
    grid: tuple[int, int] | None = None,
    **kwargs: Any,
) -> Generator[Node]:
    """Create a node, optionally as a context for child nodes.

    Usage:
        # Simple node
        db = node(icon="database", label="Database")

        # Node with children (group)
        with node(icon="cloud", label="Cloud", show_as=GROUP) as cloud:
            web = node(icon="web", label="Web Server")
            api = node(icon="api", label="API")

    Args:
        icon: Material Symbols icon name
        label: Node label text
        description: Optional description text
        show_as: GROUP or OUTLINE for child display mode
        grid: Grid layout as (rows, cols) tuple
        **kwargs: Additional node options

    Yields:
        The Node object
    """
    n = Node(
        icon=icon,
        label=label,
        description=description,
        show_as=show_as,
        grid=grid,
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

    _node_stack.append(n)

    try:
        yield n
    finally:
        _node_stack.pop()


def create_node(
    icon: str | None = None,
    label: str = "",
    description: str | None = None,
    show_as: ShowAs = ShowAs.OUTLINE,
    grid: tuple[int, int] | None = None,
    **kwargs: Any,
) -> Node:
    """Create a node without using context manager.

    This is useful when you don't need to nest children.

    Usage:
        db = create_node(icon="database", label="Database")
    """
    with node(icon=icon, label=label, description=description, show_as=show_as, grid=grid, **kwargs) as n:
        pass
    return n


def outline_item(text: str, children: list[OutlineItem] | None = None) -> OutlineItem:
    """Create an outline item for a node.

    Usage:
        with node(icon="api", label="API") as api:
            api.outline.append(outline_item("GET /users"))
            api.outline.append(outline_item("POST /users"))
    """
    return OutlineItem(text=text, children=children or [])


def edge(
    source: Node | OutlineItem,
    target: Node | OutlineItem,
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


class EdgeBuilder:
    """Helper class for building edges with labels using | operator."""

    def __init__(self, source: Node | OutlineItem, target: Node | OutlineItem, style: EdgeStyle):
        self.source = source
        self.target = target
        self.style = style
        self._edge: Edge | None = None

    def __or__(self, label: str) -> Edge:
        """Add a label to the edge."""
        e = edge(self.source, self.target, self.style, label=label)
        self._edge = e
        return e

    def __repr__(self) -> str:
        return f"EdgeBuilder({self.source} -> {self.target})"


# Monkey-patch Node for better edge syntax
_original_node_rshift = Node.__rshift__


def _new_node_rshift(self: Node, other: Node | OutlineItem | str) -> Edge | EdgeBuilder:
    """Enhanced >> operator that supports labels."""
    if isinstance(other, str):
        # This is actually a label, return edge builder
        raise TypeError("Use (node >> node) | 'label' syntax for labeled edges")

    edge_obj = _original_node_rshift(self, other)
    return edge_obj


Node.__rshift__ = _new_node_rshift  # type: ignore


# Provide simpler function names as aliases
def n(icon: str | None = None, label: str = "", **kwargs: Any) -> Node:
    """Shorthand for create_node()."""
    return create_node(icon=icon, label=label, **kwargs)
