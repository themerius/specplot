"""SVG renderer using drawsvg."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import drawsvg as draw

from .icons import get_icon_path
from .layout import (
    LayoutConfig,
    get_node_connection_point,
    get_outline_item_connection_point,
    layout_diagram,
)

if TYPE_CHECKING:
    from .models import Diagram, Edge, Node, OutlineItem


class Theme:
    """Color theme for diagrams."""

    def __init__(
        self,
        background: str = "#ffffff",
        node_fill: str = "#f8fafc",
        node_stroke: str = "#cbd5e1",
        node_header_fill: str = "#e2e8f0",
        group_fill: str = "#f1f5f9",
        group_stroke: str = "#94a3b8",
        text_color: str = "#1e293b",
        text_secondary: str = "#64748b",
        icon_color: str = "#475569",
        edge_color: str = "#64748b",
        accent_color: str = "#3b82f6",
    ):
        self.background = background
        self.node_fill = node_fill
        self.node_stroke = node_stroke
        self.node_header_fill = node_header_fill
        self.group_fill = group_fill
        self.group_stroke = group_stroke
        self.text_color = text_color
        self.text_secondary = text_secondary
        self.icon_color = icon_color
        self.edge_color = edge_color
        self.accent_color = accent_color


DEFAULT_THEME = Theme()


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max characters with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"  # Unicode ellipsis


def wrap_text_two_lines(text: str, max_chars_per_line: int) -> tuple[str, str | None]:
    """Wrap text into up to two lines with word boundary awareness.

    Returns:
        (line1, line2) where line2 is None if text fits on one line
    """
    if len(text) <= max_chars_per_line:
        return text, None

    # Find a good break point (word boundary) near max_chars_per_line
    break_point = max_chars_per_line

    # Look backwards for a space to break at
    while break_point > 0 and text[break_point] != ' ':
        break_point -= 1

    # If no space found, just break at max_chars
    if break_point == 0:
        break_point = max_chars_per_line

    line1 = text[:break_point].rstrip()
    remaining = text[break_point:].lstrip()

    # Truncate second line if needed
    if len(remaining) > max_chars_per_line:
        line2 = remaining[:max_chars_per_line - 1] + "\u2026"
    else:
        line2 = remaining

    return line1, line2 if remaining else None


class EdgeRouter:
    """Routes edges with collision avoidance and crossing minimization.

    This class pre-analyzes all edges in a diagram and assigns connection
    points that minimize visual collisions and crossings.
    """

    def __init__(self, diagram: Diagram, config: LayoutConfig):
        from .models import Node, ShowAs

        self.diagram = diagram
        self.config = config
        self._node_connections: dict[int, dict[str, list[dict]]] = {}
        self._assigned_points: dict[int, tuple[float, float, float, float]] = {}
        # Maps edge id -> (start_x, start_y, end_x, end_y)

        self._analyze_edges()

    def _get_effective_bounds(self, node: Node) -> tuple[float, float, float, float]:
        """Get effective bounds for a node, using parent for outline children."""
        from .models import ShowAs

        if node._parent and node._parent.show_as == ShowAs.OUTLINE:
            parent = node._parent
            return parent.x, parent.y, parent.width, parent.height
        return node.x, node.y, node.width, node.height

    def _get_node_center(self, node: Node) -> tuple[float, float]:
        """Get center point using effective bounds."""
        x, y, w, h = self._get_effective_bounds(node)
        return x + w / 2, y + h / 2

    def _detect_best_side(
        self, src_node: Node, tgt_node: Node, is_source: bool
    ) -> str:
        """Detect best connection side for a node based on the other endpoint."""
        from .models import ShowAs

        src_x, src_y, src_w, src_h = self._get_effective_bounds(src_node)
        tgt_x, tgt_y, tgt_w, tgt_h = self._get_effective_bounds(tgt_node)

        src_cx, src_cy = src_x + src_w / 2, src_y + src_h / 2
        tgt_cx, tgt_cy = tgt_x + tgt_w / 2, tgt_y + tgt_h / 2

        dx = tgt_cx - src_cx
        dy = tgt_cy - src_cy

        # Check overlaps
        horizontal_overlap = src_x < tgt_x + tgt_w and src_x + src_w > tgt_x
        vertical_overlap = src_y < tgt_y + tgt_h and src_y + src_h > tgt_y

        if is_source:
            # Determine exit side
            if horizontal_overlap and not vertical_overlap:
                return "bottom" if dy > 0 else "top"
            elif vertical_overlap and not horizontal_overlap:
                return "right" if dx > 0 else "left"
            else:
                if abs(dx) > abs(dy):
                    return "right" if dx > 0 else "left"
                else:
                    return "bottom" if dy > 0 else "top"
        else:
            # Determine entry side (opposite logic)
            if horizontal_overlap and not vertical_overlap:
                return "top" if dy > 0 else "bottom"
            elif vertical_overlap and not horizontal_overlap:
                return "left" if dx > 0 else "right"
            else:
                if abs(dx) > abs(dy):
                    return "left" if dx > 0 else "right"
                else:
                    return "top" if dy > 0 else "bottom"

    def _analyze_edges(self) -> None:
        """Analyze all edges and group by node and side."""
        from .models import Node, OutlineItem

        # Initialize node connections dict
        def ensure_node(node: Node) -> None:
            node_id = id(node)
            if node_id not in self._node_connections:
                self._node_connections[node_id] = {
                    "left": [], "right": [], "top": [], "bottom": []
                }

        # First pass: determine sides and collect edges
        for edge in self.diagram.edges:
            # Get actual nodes
            if isinstance(edge.source, Node):
                src_node = edge.source
            elif isinstance(edge.source, OutlineItem):
                src_node = edge.source._parent_node
            else:
                continue

            if isinstance(edge.target, Node):
                tgt_node = edge.target
            elif isinstance(edge.target, OutlineItem):
                tgt_node = edge.target._parent_node
            else:
                continue

            if src_node is None or tgt_node is None:
                continue

            ensure_node(src_node)
            ensure_node(tgt_node)

            # Determine sides
            src_side = self._detect_best_side(src_node, tgt_node, is_source=True)
            tgt_side = self._detect_best_side(src_node, tgt_node, is_source=False)

            # Check for same-column stacking (needs special handling)
            src_x, src_y, src_w, src_h = self._get_effective_bounds(src_node)
            tgt_x, tgt_y, tgt_w, tgt_h = self._get_effective_bounds(tgt_node)
            src_cx = src_x + src_w / 2
            tgt_cx = tgt_x + tgt_w / 2

            is_same_column = abs(src_cx - tgt_cx) < min(src_w, tgt_w) / 2
            needs_detour = is_same_column and src_side in ("top", "bottom")

            # Get other endpoint position for sorting
            tgt_cx, tgt_cy = self._get_node_center(tgt_node)
            src_cx, src_cy = self._get_node_center(src_node)

            # Store edge info for source node
            edge_info_src = {
                "edge": edge,
                "edge_id": id(edge),
                "is_source": True,
                "other_node": tgt_node,
                "other_x": tgt_cx,
                "other_y": tgt_cy,
                "side": src_side,
                "needs_detour": needs_detour,
                "actual_node": edge.source if isinstance(edge.source, Node) else src_node,
            }

            # Store edge info for target node
            edge_info_tgt = {
                "edge": edge,
                "edge_id": id(edge),
                "is_source": False,
                "other_node": src_node,
                "other_x": src_cx,
                "other_y": src_cy,
                "side": tgt_side,
                "needs_detour": needs_detour,
                "actual_node": edge.target if isinstance(edge.target, Node) else tgt_node,
            }

            # Handle detour: override sides for same-column
            if needs_detour:
                if src_side == "bottom":
                    # Exit bottom-right, enter top-right
                    edge_info_src["side"] = "bottom"
                    edge_info_src["corner"] = "right"
                    edge_info_tgt["side"] = "top"
                    edge_info_tgt["corner"] = "right"
                else:
                    edge_info_src["side"] = "top"
                    edge_info_src["corner"] = "right"
                    edge_info_tgt["side"] = "bottom"
                    edge_info_tgt["corner"] = "right"

            self._node_connections[id(src_node)][edge_info_src["side"]].append(edge_info_src)
            self._node_connections[id(tgt_node)][edge_info_tgt["side"]].append(edge_info_tgt)

        # Second pass: sort edges on each side to minimize crossings
        self._sort_edges_to_minimize_crossings()

        # Third pass: assign connection points
        self._assign_connection_points()

    def _sort_edges_to_minimize_crossings(self) -> None:
        """Sort edges on each side of each node to minimize crossings."""
        for node_id, sides in self._node_connections.items():
            # For left/right sides: sort by other endpoint's y position
            for side in ["left", "right"]:
                sides[side].sort(key=lambda e: e["other_y"])

            # For top/bottom sides: sort by other endpoint's x position
            for side in ["top", "bottom"]:
                sides[side].sort(key=lambda e: e["other_x"])

    def _assign_connection_points(self) -> None:
        """Assign actual connection points with offsets to avoid collisions."""
        from .models import ShowAs

        for node_id, sides in self._node_connections.items():
            for side, edges in sides.items():
                if not edges:
                    continue

                # Get the node (from first edge)
                first_edge = edges[0]
                if first_edge["is_source"]:
                    node = first_edge["actual_node"]
                else:
                    node = first_edge["actual_node"]

                # Get effective bounds
                x, y, w, h = self._get_effective_bounds(node)

                # Calculate connection points with offsets
                n = len(edges)
                offset_spacing = 15  # Pixels between connection points
                total_spread = (n - 1) * offset_spacing

                for i, edge_info in enumerate(edges):
                    edge_id = edge_info["edge_id"]
                    is_source = edge_info["is_source"]
                    needs_detour = edge_info.get("needs_detour", False)
                    corner = edge_info.get("corner")

                    # Calculate offset from center
                    if n == 1:
                        offset = 0
                    else:
                        offset = -total_spread / 2 + i * offset_spacing

                    # Calculate base connection point
                    if side == "right":
                        cx = x + w
                        cy = y + h / 2 + offset
                    elif side == "left":
                        cx = x
                        cy = y + h / 2 + offset
                    elif side == "top":
                        cx = x + w / 2 + offset
                        cy = y
                    else:  # bottom
                        cx = x + w / 2 + offset
                        cy = y + h

                    # For detour edges, adjust to corner
                    if needs_detour and corner == "right":
                        if side == "bottom":
                            cx = x + w  # Right edge
                            cy = y + h  # Bottom
                        elif side == "top":
                            cx = x + w  # Right edge
                            cy = y  # Top

                    # Handle outline children specially
                    actual_node = edge_info["actual_node"]
                    if actual_node._parent and actual_node._parent.show_as == ShowAs.OUTLINE:
                        parent = actual_node._parent
                        try:
                            idx = parent.children.index(actual_node)
                            # Calculate outline item y position
                            base_y = parent.y + self.config.header_height
                            if parent.description:
                                base_y += self.config.description_height
                            item_y = base_y + idx * self.config.outline_item_height + self.config.outline_item_height / 2
                            cy = item_y
                            if side == "right":
                                cx = parent.x + parent.width
                            elif side == "left":
                                cx = parent.x
                        except ValueError:
                            pass

                    # Store the assigned point
                    if edge_id not in self._assigned_points:
                        self._assigned_points[edge_id] = {}

                    if is_source:
                        self._assigned_points[edge_id]["source"] = (cx, cy, side, needs_detour)
                    else:
                        self._assigned_points[edge_id]["target"] = (cx, cy, side, needs_detour)

    def get_edge_points(self, edge: Edge) -> tuple[
        tuple[float, float, str, bool],
        tuple[float, float, str, bool]
    ] | None:
        """Get assigned connection points for an edge.

        Returns ((sx, sy, src_side, src_detour), (tx, ty, tgt_side, tgt_detour))
        or None if edge wasn't analyzed.
        """
        edge_id = id(edge)
        if edge_id not in self._assigned_points:
            return None

        points = self._assigned_points[edge_id]
        if "source" not in points or "target" not in points:
            return None

        return points["source"], points["target"]


class DiagramRenderer:
    """Renders diagrams to SVG."""

    def __init__(
        self,
        theme: Theme | None = None,
        config: LayoutConfig | None = None,
    ):
        self.theme = theme or DEFAULT_THEME
        self.config = config or LayoutConfig()

    def render(self, diagram: Diagram) -> draw.Drawing:
        """Render a diagram to an SVG Drawing object."""
        # Layout the diagram
        layout_diagram(diagram, self.config)

        # Calculate canvas size (including nested nodes)
        def get_bounds(node: Node) -> tuple[float, float]:
            max_x = node.x + node.width
            max_y = node.y + node.height
            for child in node.children:
                cx, cy = get_bounds(child)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
            return max_x, max_y

        max_x = diagram.padding
        max_y = diagram.padding

        for node in diagram.nodes:
            mx, my = get_bounds(node)
            max_x = max(max_x, mx)
            max_y = max(max_y, my)

        width = max_x + diagram.padding
        height = max_y + diagram.padding

        # Create drawing
        d = draw.Drawing(width, height)

        # Add background
        d.append(
            draw.Rectangle(
                0, 0, width, height,
                fill=self.theme.background,
            )
        )

        # Render nodes first
        for node in diagram.nodes:
            self._render_node(d, node)

        # Create edge router for collision-free edge placement
        edge_router = EdgeRouter(diagram, self.config)

        # Render edges on top (so arrowheads are visible)
        for edge in diagram.edges:
            self._render_edge(d, edge, edge_router)

        return d

    def _render_node(self, d: draw.Drawing, node: Node) -> None:
        """Render a single node."""
        from .models import ShowAs

        x, y = node.x, node.y
        w, h = node.width, node.height

        is_group = node.children and node.show_as == ShowAs.GROUP

        # Node background
        fill = self.theme.group_fill if is_group else self.theme.node_fill
        stroke = self.theme.group_stroke if is_group else self.theme.node_stroke

        d.append(
            draw.Rectangle(
                x, y, w, h,
                fill=fill,
                stroke=stroke,
                stroke_width=1.5,
                rx=6, ry=6,
            )
        )

        # Header background
        header_height = self.config.header_height
        d.append(
            draw.Rectangle(
                x, y, w, header_height,
                fill=self.theme.node_header_fill,
                stroke="none",
                rx=6, ry=6,
            )
        )
        # Cover bottom corners of header
        d.append(
            draw.Rectangle(
                x, y + header_height - 6, w, 6,
                fill=self.theme.node_header_fill,
                stroke="none",
            )
        )

        # Header separator line
        d.append(
            draw.Line(
                x, y + header_height,
                x + w, y + header_height,
                stroke=stroke,
                stroke_width=1,
            )
        )

        # Icon
        icon_x = x + self.config.node_padding
        icon_y = y + (header_height - self.config.icon_size) / 2

        if node.icon:
            icon_path = get_icon_path(node.icon)
            # Detect icon coordinate system by checking for large negative values
            # Google Material Symbols use viewBox "0 -960 960 960"
            # Standard icons use viewBox "0 0 24 24"
            is_material_symbols = "-" in icon_path and any(
                c in icon_path for c in ["-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1"]
            )

            if is_material_symbols:
                # Material Symbols: scale from 960 and offset for negative y
                scale = self.config.icon_size / 960
                # Translate to position, then scale, then offset for negative coords
                icon_group = draw.Group(
                    transform=f"translate({icon_x}, {icon_y + self.config.icon_size}) scale({scale})"
                )
            else:
                # Standard 24x24 icons
                icon_group = draw.Group(
                    transform=f"translate({icon_x}, {icon_y}) scale({self.config.icon_size / 24})"
                )

            icon_group.append(
                draw.Path(
                    d=icon_path,
                    fill=self.theme.icon_color,
                )
            )
            d.append(icon_group)

        # Label (truncated to fit)
        label_x = icon_x + self.config.icon_size + 8
        label_y = y + header_height / 2
        label_text = truncate_text(node.label, self.config.max_label_chars)

        d.append(
            draw.Text(
                label_text,
                self.config.font_size_label,
                label_x, label_y,
                fill=self.theme.text_color,
                font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                font_weight="500",
                dominant_baseline="middle",
            )
        )

        # Description (two-line box with word wrapping)
        current_y = y + header_height

        if node.description:
            # Calculate max chars based on available width
            available_width = w - (self.config.node_padding * 2)
            max_chars = int(available_width / self.config.char_width_avg)

            line1, line2 = wrap_text_two_lines(node.description, max_chars)

            line_height = 16  # Spacing between lines
            # Center the text block vertically in the description area
            if line2:
                # Two lines: position them centered
                text_block_height = line_height * 2
                start_y = current_y + (self.config.description_height - text_block_height) / 2 + line_height / 2
            else:
                # One line: center it vertically
                start_y = current_y + self.config.description_height / 2

            d.append(
                draw.Text(
                    line1,
                    self.config.font_size_description,
                    x + self.config.node_padding,
                    start_y,
                    fill=self.theme.text_secondary,
                    font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                    dominant_baseline="middle",
                )
            )

            if line2:
                d.append(
                    draw.Text(
                        line2,
                        self.config.font_size_description,
                        x + self.config.node_padding,
                        start_y + line_height,
                        fill=self.theme.text_secondary,
                        font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                        dominant_baseline="middle",
                    )
                )

            current_y += self.config.description_height

            # Separator line after description (only if there are children)
            if node.children:
                d.append(
                    draw.Line(
                        x, current_y,
                        x + w, current_y,
                        stroke=stroke,
                        stroke_width=0.5,
                        stroke_dasharray="3,2",
                    )
                )

        # Render children
        if node.children:
            if node.show_as == ShowAs.GROUP:
                # Render child nodes
                for child in node.children:
                    self._render_node(d, child)
            else:
                # Render as outline items
                for i, child in enumerate(node.children):
                    item_y = current_y + i * self.config.outline_item_height
                    self._render_outline_item(d, node, child.label, item_y, level=0)

        # Render outline items (legacy)
        if node.outline:
            for i, item in enumerate(node.outline):
                item_y = current_y + i * self.config.outline_item_height
                self._render_outline_item(d, node, item.text, item_y, level=0)

    def _render_outline_item(
        self,
        d: draw.Drawing,
        node: Node,
        text: str,
        y: float,
        level: int = 0,
    ) -> None:
        """Render an outline item (single line, truncated)."""
        x = node.x + self.config.node_padding + level * 12
        text_y = y + self.config.outline_item_height / 2

        # Bullet point and truncated text
        bullet = "\u2022" if level == 0 else "\u25e6"  # ● or ○
        # Account for bullet and space in max chars
        truncated_text = truncate_text(text, self.config.max_outline_chars - 2)

        d.append(
            draw.Text(
                f"{bullet} {truncated_text}",
                self.config.font_size_outline,
                x, text_y,
                fill=self.theme.text_secondary,
                font_family="JetBrains Mono, Consolas, monospace",
                dominant_baseline="middle",
            )
        )

    def _render_edge(self, d: draw.Drawing, edge: Edge, edge_router: EdgeRouter) -> None:
        """Render an edge using pre-computed connection points from EdgeRouter."""
        from .models import EdgeStyle, Node, OutlineItem, ShowAs

        # Get pre-computed connection points from router
        points = edge_router.get_edge_points(edge)
        if points is None:
            return

        (sx, sy, src_side, src_detour), (tx, ty, tgt_side, tgt_detour) = points
        needs_detour = src_detour or tgt_detour

        # Style settings
        is_dotted = edge.style in (
            EdgeStyle.DOTTED,
            EdgeStyle.DOTTED_ARROW_RIGHT,
            EdgeStyle.DOTTED_ARROW_LEFT,
        )
        has_arrow_end = edge.style in (
            EdgeStyle.ARROW_RIGHT,
            EdgeStyle.DOTTED_ARROW_RIGHT,
        )
        has_arrow_start = edge.style in (
            EdgeStyle.ARROW_LEFT,
            EdgeStyle.DOTTED_ARROW_LEFT,
        )

        path = draw.Path(
            stroke=self.theme.edge_color,
            stroke_width=1.5,
            fill="none",
        )

        if is_dotted:
            path = draw.Path(
                stroke=self.theme.edge_color,
                stroke_width=1.5,
                fill="none",
                stroke_dasharray="5,5",
            )

        # Calculate control points based on connection sides
        dx = abs(tx - sx)
        dy = abs(ty - sy)

        if needs_detour:
            # Same-column vertical: route around via side with clear visual path
            detour_offset = 50

            # Get the rightmost x coordinate for the detour
            mid_x = max(sx, tx) + detour_offset

            # Control points: go right then curve to target
            c1x, c1y = mid_x, sy
            c2x, c2y = mid_x, ty

            path.M(sx, sy)
            path.C(c1x, c1y, c2x, c2y, tx, ty)

        elif src_side in ("left", "right"):
            # Horizontal exit/entry - use horizontal control points
            control_offset = max(50, dx * 0.5)

            if src_side == "right":
                c1x, c1y = sx + control_offset, sy
                c2x, c2y = tx - control_offset, ty
            else:
                c1x, c1y = sx - control_offset, sy
                c2x, c2y = tx + control_offset, ty

            path.M(sx, sy)
            path.C(c1x, c1y, c2x, c2y, tx, ty)

        else:
            # Vertical exit/entry - use vertical control points
            control_offset = max(50, dy * 0.5)

            if src_side == "bottom":
                c1x, c1y = sx, sy + control_offset
                c2x, c2y = tx, ty - control_offset
            else:
                c1x, c1y = sx, sy - control_offset
                c2x, c2y = tx, ty + control_offset

            path.M(sx, sy)
            path.C(c1x, c1y, c2x, c2y, tx, ty)

        d.append(path)

        # Draw arrowheads
        arrow_size = 8

        if has_arrow_end:
            # Angle from last control point to endpoint
            angle = math.atan2(ty - c2y, tx - c2x)
            self._draw_arrowhead(d, tx, ty, angle, arrow_size)

        if has_arrow_start:
            # Angle from first control point to start (reversed)
            angle = math.atan2(sy - c1y, sx - c1x)
            self._draw_arrowhead(d, sx, sy, angle, arrow_size)

        # Draw label if present (positioned on the curve)
        if edge.label:
            # Calculate bezier midpoint at t=0.5
            t = 0.5
            # Bezier formula: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
            mid_x = (1-t)**3 * sx + 3*(1-t)**2*t * c1x + 3*(1-t)*t**2 * c2x + t**3 * tx
            mid_y = (1-t)**3 * sy + 3*(1-t)**2*t * c1y + 3*(1-t)*t**2 * c2y + t**3 * ty

            # Label background
            label_width = len(edge.label) * 7 + 12
            d.append(
                draw.Rectangle(
                    mid_x - label_width / 2,
                    mid_y - 9,
                    label_width,
                    18,
                    fill=self.theme.background,
                    stroke=self.theme.edge_color,
                    stroke_width=1,
                    rx=4, ry=4,
                )
            )

            d.append(
                draw.Text(
                    edge.label,
                    11,
                    mid_x, mid_y,
                    fill=self.theme.text_secondary,
                    font_family="JetBrains Mono, Consolas, monospace",
                    text_anchor="middle",
                    dominant_baseline="middle",
                )
            )

    def _draw_arrowhead(
        self,
        d: draw.Drawing,
        x: float,
        y: float,
        angle: float,
        size: float,
    ) -> None:
        """Draw an arrowhead at the given position and angle."""
        # Triangle arrowhead points
        p1_x = x - size * math.cos(angle - math.pi / 6)
        p1_y = y - size * math.sin(angle - math.pi / 6)
        p2_x = x - size * math.cos(angle + math.pi / 6)
        p2_y = y - size * math.sin(angle + math.pi / 6)

        d.append(
            draw.Lines(
                x, y,
                p1_x, p1_y,
                p2_x, p2_y,
                x, y,
                fill=self.theme.edge_color,
                stroke="none",
            )
        )


def render_to_svg(diagram: Diagram, filename: str | None = None) -> str:
    """Render a diagram to SVG.

    Args:
        diagram: The diagram to render
        filename: Optional filename to save to (without extension)

    Returns:
        SVG content as string
    """
    renderer = DiagramRenderer()
    drawing = renderer.render(diagram)

    if filename:
        drawing.save_svg(f"{filename}.svg")

    return drawing.as_svg()
