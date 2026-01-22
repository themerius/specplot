"""Data models for specplot diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class ShowAs(Enum):
    """How to display child nodes."""

    OUTLINE = "outline"
    GROUP = "group"


class EdgeStyle(Enum):
    """Edge/arrow styles."""

    ARROW_RIGHT = "->"
    ARROW_LEFT = "<-"
    LINE = "--"
    DOTTED = ".."
    DOTTED_ARROW_RIGHT = "..>"
    DOTTED_ARROW_LEFT = "<.."


@dataclass
class OutlineItem:
    """An item in a node's outline list."""

    text: str
    children: list[OutlineItem] = field(default_factory=list)
    _parent_node: Node | None = field(default=None, repr=False)
    _index: int = field(default=0, repr=False)

    def __rshift__(self, other: Node | OutlineItem) -> Edge:
        """Create an edge from this outline item to another node/item."""
        return Edge(source=self, target=other, style=EdgeStyle.ARROW_RIGHT)

    def __lshift__(self, other: Node | OutlineItem) -> Edge:
        """Create an edge to this outline item from another node/item."""
        return Edge(source=other, target=self, style=EdgeStyle.ARROW_RIGHT)


@dataclass
class Node:
    """A node in the diagram."""

    icon: str | None = None
    label: str = ""
    description: str | None = None
    show_as: ShowAs = ShowAs.OUTLINE
    grid: tuple[int, int] | None = None
    children: list[Node] = field(default_factory=list)
    outline: list[OutlineItem] = field(default_factory=list)
    _parent: Node | None = field(default=None, repr=False)
    _diagram: Diagram | None = field(default=None, repr=False)

    # Layout properties (set during rendering)
    x: float = field(default=0, repr=False)
    y: float = field(default=0, repr=False)
    width: float = field(default=0, repr=False)
    height: float = field(default=0, repr=False)

    def __rshift__(self, other: Node | OutlineItem) -> Edge:
        """Create an edge from this node to another (->)."""
        edge = Edge(source=self, target=other, style=EdgeStyle.ARROW_RIGHT)
        if self._diagram:
            self._diagram.edges.append(edge)
        return edge

    def __lshift__(self, other: Node | OutlineItem) -> Edge:
        """Create an edge to this node from another (<-)."""
        edge = Edge(source=other, target=self, style=EdgeStyle.ARROW_RIGHT)
        if self._diagram:
            self._diagram.edges.append(edge)
        return edge

    def __sub__(self, other: Node | OutlineItem) -> Edge:
        """Create an undirected edge (--)."""
        edge = Edge(source=self, target=other, style=EdgeStyle.LINE)
        if self._diagram:
            self._diagram.edges.append(edge)
        return edge


@dataclass
class Edge:
    """An edge connecting two nodes or outline items."""

    source: Node | OutlineItem
    target: Node | OutlineItem
    style: EdgeStyle = EdgeStyle.ARROW_RIGHT
    label: str | None = None
    _diagram: Diagram | None = field(default=None, repr=False)

    def __or__(self, label: str) -> Edge:
        """Add a label to the edge using | operator."""
        self.label = label
        return self

    def __rshift__(self, other: Node | OutlineItem) -> Edge:
        """Chain edges: a >> b >> c."""
        new_edge = Edge(source=self.target, target=other, style=self.style)
        if self._diagram:
            self._diagram.edges.append(new_edge)
        return new_edge


@dataclass
class Diagram:
    """The root diagram container."""

    filename: str = "diagram"
    title: str | None = None
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    padding: float = 40
    node_spacing: float = 60
    background_color: str = "#ffffff"

    # Theme colors
    node_fill: str = "#f8fafc"
    node_stroke: str = "#64748b"
    node_header_fill: str = "#e2e8f0"
    text_color: str = "#1e293b"
    icon_color: str = "#475569"
    edge_color: str = "#64748b"
    accent_color: str = "#3b82f6"
