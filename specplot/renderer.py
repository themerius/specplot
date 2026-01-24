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

        # Render edges on top (so arrowheads are visible)
        for edge in diagram.edges:
            self._render_edge(d, edge)

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

    def _render_edge(self, d: draw.Drawing, edge: Edge) -> None:
        """Render an edge between nodes with smart routing."""
        from .models import EdgeStyle, Node, OutlineItem, ShowAs

        def get_connection_point(
            node: Node, direction: str
        ) -> tuple[float, float]:
            """Get connection point for a node.

            For outline children: connects from parent's edge at the outline item's y-position.
            For outline parents: connects from header area (not center of entire node).
            For regular nodes: connects from the node's edge.
            """
            if node._parent and node._parent.show_as == ShowAs.OUTLINE:
                # Outline child: use parent's x but this item's y position
                parent = node._parent
                try:
                    idx = parent.children.index(node)
                    return get_outline_item_connection_point(
                        parent, idx, direction, self.config
                    )
                except ValueError:
                    pass

            # For nodes with OUTLINE children, connect from header area for left/right
            if node.children and node.show_as == ShowAs.OUTLINE:
                header_y = node.y + self.config.header_height / 2
                if direction == "right":
                    return node.x + node.width, header_y
                elif direction == "left":
                    return node.x, header_y
                # For top/bottom, use normal node connection points

            # Regular node - all 4 sides
            return get_node_connection_point(node, direction)

        def get_effective_bounds(node: Node) -> tuple[float, float, float, float]:
            """Get effective bounds for a node, using parent for outline children.

            Returns (x, y, width, height).
            """
            if node._parent and node._parent.show_as == ShowAs.OUTLINE:
                # Outline child: use parent's bounds
                parent = node._parent
                return parent.x, parent.y, parent.width, parent.height
            return node.x, node.y, node.width, node.height

        def get_node_center_from_bounds(x: float, y: float, w: float, h: float) -> tuple[float, float]:
            """Get center point from bounds."""
            return x + w / 2, y + h / 2

        def detect_best_directions(
            src: Node, tgt: Node
        ) -> tuple[str, str]:
            """Detect best connection directions based on relative positions.

            Returns (source_direction, target_direction).
            """
            # Use effective bounds (parent bounds for outline children)
            src_x, src_y, src_w, src_h = get_effective_bounds(src)
            tgt_x, tgt_y, tgt_w, tgt_h = get_effective_bounds(tgt)

            src_cx, src_cy = get_node_center_from_bounds(src_x, src_y, src_w, src_h)
            tgt_cx, tgt_cy = get_node_center_from_bounds(tgt_x, tgt_y, tgt_w, tgt_h)

            dx = tgt_cx - src_cx
            dy = tgt_cy - src_cy

            # Check if nodes are in same column (vertically stacked)
            horizontal_overlap = (
                src_x < tgt_x + tgt_w and src_x + src_w > tgt_x
            )
            # Check if nodes are in same row (horizontally adjacent)
            vertical_overlap = (
                src_y < tgt_y + tgt_h and src_y + src_h > tgt_y
            )

            # Determine dominant direction
            if horizontal_overlap and not vertical_overlap:
                # Vertically stacked - use top/bottom
                if dy > 0:
                    return "bottom", "top"
                else:
                    return "top", "bottom"
            elif vertical_overlap and not horizontal_overlap:
                # Horizontally adjacent - use left/right
                if dx > 0:
                    return "right", "left"
                else:
                    return "left", "right"
            else:
                # Diagonal or overlapping - pick based on larger delta
                if abs(dx) > abs(dy):
                    # More horizontal
                    if dx > 0:
                        return "right", "left"
                    else:
                        return "left", "right"
                else:
                    # More vertical
                    if dy > 0:
                        return "bottom", "top"
                    else:
                        return "top", "bottom"

        def is_same_column_stacked(src: Node, tgt: Node) -> bool:
            """Check if nodes are vertically stacked in same column."""
            src_x, _, src_w, _ = get_effective_bounds(src)
            tgt_x, _, tgt_w, _ = get_effective_bounds(tgt)
            src_cx = src_x + src_w / 2
            tgt_cx = tgt_x + tgt_w / 2
            # Consider same column if centers are within half a node width
            return abs(src_cx - tgt_cx) < min(src_w, tgt_w) / 2

        # Get source and target nodes
        if isinstance(edge.source, Node):
            source_node = edge.source
        elif isinstance(edge.source, OutlineItem):
            source_node = edge.source._parent_node
        else:
            return

        if isinstance(edge.target, Node):
            target_node = edge.target
        elif isinstance(edge.target, OutlineItem):
            target_node = edge.target._parent_node
        else:
            return

        if source_node is None or target_node is None:
            return

        # Detect best connection directions
        src_dir, tgt_dir = detect_best_directions(source_node, target_node)

        # Check for same-column stacking (needs curved detour)
        needs_detour = is_same_column_stacked(source_node, target_node) and src_dir in ("top", "bottom")

        # Get connection points
        sx, sy = get_connection_point(
            edge.source if isinstance(edge.source, Node) else source_node, src_dir
        )
        tx, ty = get_connection_point(
            edge.target if isinstance(edge.target, Node) else target_node, tgt_dir
        )

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

        # Calculate control points based on connection directions
        dx = abs(tx - sx)
        dy = abs(ty - sy)

        if needs_detour:
            # Same-column vertical: route around via side with clear visual path
            # Use effective bounds for outline children
            src_x, src_y, src_w, src_h = get_effective_bounds(source_node)
            tgt_x, tgt_y, tgt_w, tgt_h = get_effective_bounds(target_node)

            detour_offset = 50  # How far to go sideways

            if src_dir == "bottom":
                # Going down: exit bottom-right, curve right, enter top-right
                mid_x = max(src_x + src_w, tgt_x + tgt_w) + detour_offset

                # Exit from bottom-right area of source
                sx = src_x + src_w  # Right edge
                sy = src_y + src_h  # Bottom edge

                # Enter from top-right area of target
                tx = tgt_x + tgt_w  # Right edge
                ty = tgt_y  # Top edge

                # Control points: go right then curve down
                c1x, c1y = mid_x, sy
                c2x, c2y = mid_x, ty
            else:
                # Going up: exit top-right, curve right, enter bottom-right
                mid_x = max(src_x + src_w, tgt_x + tgt_w) + detour_offset

                sx = src_x + src_w
                sy = src_y  # Top edge

                tx = tgt_x + tgt_w
                ty = tgt_y + tgt_h  # Bottom edge

                c1x, c1y = mid_x, sy
                c2x, c2y = mid_x, ty

            path.M(sx, sy)
            path.C(c1x, c1y, c2x, c2y, tx, ty)

        elif src_dir in ("left", "right"):
            # Horizontal exit/entry - use horizontal control points
            control_offset = max(50, dx * 0.5)

            if src_dir == "right":
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

            if src_dir == "bottom":
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
