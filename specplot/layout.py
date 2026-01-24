"""Layout algorithms for specplot diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .models import Diagram, Node


@dataclass
class Zone:
    """A zone in the layout grid."""

    direction: Literal["LR", "TB"]
    row: int
    col: int
    zone_number: int  # 1-indexed
    nodes: list[Node] = field(default_factory=list)
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0


def parse_layout(layout_spec: tuple[tuple[str, ...], ...]) -> list[Zone]:
    """Parse tuple-of-tuples layout spec into Zone objects.

    Args:
        layout_spec: Layout like (("LR",), ("TB", "TB", "TB"), ("LR",))

    Returns:
        List of Zone objects with 1-indexed zone numbers
    """
    zones = []
    zone_number = 1

    for row_idx, row in enumerate(layout_spec):
        for col_idx, direction in enumerate(row):
            if direction not in ("LR", "TB"):
                raise ValueError(f"Invalid direction '{direction}', must be 'LR' or 'TB'")
            zones.append(Zone(
                direction=direction,  # type: ignore
                row=row_idx,
                col=col_idx,
                zone_number=zone_number,
            ))
            zone_number += 1

    return zones


def bucket_nodes_by_zone(nodes: list[Node], zones: list[Zone]) -> None:
    """Assign nodes to zones based on their pos attribute.

    Modifies zones in place, adding nodes to their nodes list.

    Args:
        nodes: List of nodes with pos attributes
        zones: List of Zone objects to populate
    """
    zone_map = {z.zone_number: z for z in zones}
    max_zone = len(zones)

    for node in nodes:
        if node.pos is None:
            if max_zone == 1:
                # Single zone: default to zone 1 for backwards compatibility
                node.pos = 1
            else:
                raise ValueError(
                    f"Node '{node.label}' has no pos attribute. "
                    f"Multi-zone layouts require explicit pos (1-{max_zone})."
                )

        if node.pos < 1 or node.pos > max_zone:
            raise ValueError(
                f"Node '{node.label}' has pos={node.pos}, but valid range is 1-{max_zone}."
            )

        zone_map[node.pos].nodes.append(node)


@dataclass
class LayoutConfig:
    """Configuration for layout calculations."""

    # Fixed node width for consistent appearance
    node_width: float = 200
    node_padding: float = 12
    header_height: float = 40
    description_height: float = 38  # Two-line description box
    outline_item_height: float = 22
    group_padding: float = 16
    group_header_height: float = 44  # Header + optional description
    node_spacing_h: float = 80  # Horizontal spacing between nodes
    node_spacing_v: float = 30  # Vertical spacing in grids
    icon_size: float = 20
    font_size_label: float = 14
    font_size_description: float = 11
    font_size_outline: float = 11
    char_width_avg: float = 4.8  # Average character width for 11px sans-serif font (conservative)
    max_label_chars: int = 20  # Max chars for label
    max_description_chars: int = 28  # Max chars for description (1 line)
    max_outline_chars: int = 24  # Max chars per outline item


def estimate_text_width(text: str, char_width: float) -> float:
    """Estimate width of text based on character count."""
    return len(text) * char_width


def calculate_node_dimensions(
    node: Node,
    config: LayoutConfig,
) -> tuple[float, float]:
    """Calculate the width and height needed for a node.

    Uses fixed width for consistent appearance.

    Returns:
        (width, height) tuple
    """
    from .models import ShowAs

    # Use fixed width for regular nodes
    width = config.node_width

    # Start with header height
    height = config.header_height

    # Add description height if present (single line, fixed height)
    if node.description:
        height += config.description_height

    # Handle children
    if node.children:
        if node.show_as == ShowAs.GROUP:
            # Use pre-calculated child dimensions (from recursive calc_dims)
            child_dims = []
            for child in node.children:
                # If dimensions were already calculated, use them
                if child.width > 0 and child.height > 0:
                    child_dims.append((child.width, child.height))
                else:
                    cw, ch = calculate_node_dimensions(child, config)
                    child_dims.append((cw, ch))

            grid = node.grid or (1, len(node.children))
            rows, cols = grid

            # Calculate grid cell sizes
            col_widths = [0.0] * cols
            row_heights = [0.0] * rows

            for i, (cw, ch) in enumerate(child_dims):
                row = i // cols
                col = i % cols
                if row < rows:
                    col_widths[col] = max(col_widths[col], cw)
                    row_heights[row] = max(row_heights[row], ch)

            group_content_width = (
                sum(col_widths) + config.node_spacing_h * (cols - 1)
            )
            group_content_height = (
                sum(row_heights) + config.node_spacing_v * (rows - 1)
            )

            # Group width accommodates children plus padding
            width = group_content_width + config.group_padding * 2
            height += group_content_height + config.group_padding

        else:  # OUTLINE mode
            # Outline items add height (one line each, fixed height)
            height += len(node.children) * config.outline_item_height

    # Handle outline items (legacy support)
    if node.outline:
        height += len(node.outline) * config.outline_item_height

    return width, height


def layout_diagram(diagram: Diagram, config: LayoutConfig | None = None) -> None:
    """Calculate positions for all nodes in a diagram.

    This modifies nodes in-place, setting their x, y, width, height attributes.
    Supports zone-based layouts when diagram.layout is specified.
    """
    from .models import ShowAs

    if config is None:
        config = LayoutConfig()

    # First pass: calculate dimensions for all nodes (recursively)
    def calc_dims(node: Node) -> tuple[float, float]:
        # First calculate children dimensions recursively
        for child in node.children:
            calc_dims(child)
        # Then calculate this node's dimensions
        w, h = calculate_node_dimensions(node, config)
        node.width = w
        node.height = h
        return w, h

    for node in diagram.nodes:
        calc_dims(node)

    # Parse layout and bucket nodes
    layout_spec = diagram.layout or (("LR",),)
    zones = parse_layout(layout_spec)
    bucket_nodes_by_zone(diagram.nodes, zones)

    # Calculate zone dimensions based on their nodes and direction
    for zone in zones:
        if not zone.nodes:
            zone.width = 0
            zone.height = 0
            continue

        if zone.direction == "LR":
            # Horizontal: width = sum of node widths + spacing, height = max node height
            zone.width = sum(n.width for n in zone.nodes) + config.node_spacing_h * (len(zone.nodes) - 1)
            zone.height = max(n.height for n in zone.nodes)
        else:  # TB
            # Vertical: width = max node width, height = sum of node heights + spacing
            zone.width = max(n.width for n in zone.nodes)
            zone.height = sum(n.height for n in zone.nodes) + config.node_spacing_v * (len(zone.nodes) - 1)

    # Position zones in the grid (row by row)
    # First, calculate max columns per row and dimensions
    rows_in_layout = {}
    for zone in zones:
        if zone.row not in rows_in_layout:
            rows_in_layout[zone.row] = []
        rows_in_layout[zone.row].append(zone)

    # Calculate row heights and column widths
    row_heights = {}
    for row_idx, row_zones in rows_in_layout.items():
        row_heights[row_idx] = max((z.height for z in row_zones if z.height > 0), default=0)

    # For each row, calculate column widths
    # But we need to align columns across rows for a grid effect
    # Actually, zones in different rows may have different column counts
    # So we calculate width per row and center each row

    # Calculate total width per row
    row_widths = {}
    for row_idx, row_zones in rows_in_layout.items():
        non_empty = [z for z in row_zones if z.width > 0]
        if non_empty:
            row_widths[row_idx] = sum(z.width for z in non_empty) + config.node_spacing_h * (len(non_empty) - 1)
        else:
            row_widths[row_idx] = 0

    # Find max row width for centering
    max_row_width = max(row_widths.values()) if row_widths else 0

    # Position zones row by row
    current_y = diagram.padding
    for row_idx in sorted(rows_in_layout.keys()):
        row_zones = rows_in_layout[row_idx]
        row_height = row_heights[row_idx]
        row_width = row_widths[row_idx]

        # Skip empty rows
        if row_height == 0:
            continue

        # Center this row horizontally
        current_x = diagram.padding + (max_row_width - row_width) / 2

        for zone in row_zones:
            if zone.width == 0:
                continue

            zone.x = current_x
            zone.y = current_y
            current_x += zone.width + config.node_spacing_h

        current_y += row_height + config.node_spacing_v

    # Position nodes within zones
    for zone in zones:
        if not zone.nodes:
            continue

        if zone.direction == "LR":
            # Horizontal layout: nodes side by side, vertically centered
            x = zone.x
            for node in zone.nodes:
                node.x = x
                # Vertically center within zone
                node.y = zone.y + (zone.height - node.height) / 2
                x += node.width + config.node_spacing_h
        else:  # TB
            # Vertical layout: nodes stacked, horizontally centered
            y = zone.y
            for node in zone.nodes:
                # Horizontally center within zone
                node.x = zone.x + (zone.width - node.width) / 2
                node.y = y
                y += node.height + config.node_spacing_v

    # Position children within their parents
    def position_children(parent: Node) -> None:
        if not parent.children:
            return

        if parent.show_as == ShowAs.GROUP:
            grid = parent.grid or (1, len(parent.children))
            rows, cols = grid

            # Calculate child dimensions for grid
            child_dims = [(c.width, c.height) for c in parent.children]

            col_widths = [0.0] * cols
            row_heights_grid = [0.0] * rows

            for i, (cw, ch) in enumerate(child_dims):
                row = i // cols
                col = i % cols
                if row < rows:
                    col_widths[col] = max(col_widths[col], cw)
                    row_heights_grid[row] = max(row_heights_grid[row], ch)

            # Starting position inside parent
            start_y = parent.y + config.group_header_height
            if parent.description:
                start_y += config.description_height

            start_x = parent.x + config.group_padding

            # Position each child in the grid
            current_y = start_y
            for row in range(rows):
                current_x = start_x
                for col in range(cols):
                    idx = row * cols + col
                    if idx < len(parent.children):
                        child = parent.children[idx]
                        child.x = current_x
                        child.y = current_y
                        # Recursively position grandchildren
                        position_children(child)
                    current_x += col_widths[col] + config.node_spacing_h
                current_y += row_heights_grid[row] + config.node_spacing_v

    for node in diagram.nodes:
        position_children(node)


def get_node_connection_point(
    node: Node, direction: str = "right"
) -> tuple[float, float]:
    """Get the connection point for edges on a node.

    Args:
        node: The node to get connection point for
        direction: 'left', 'right', 'top', or 'bottom'

    Returns:
        (x, y) coordinates of the connection point
    """
    if direction == "right":
        return node.x + node.width, node.y + node.height / 2
    elif direction == "left":
        return node.x, node.y + node.height / 2
    elif direction == "top":
        return node.x + node.width / 2, node.y
    elif direction == "bottom":
        return node.x + node.width / 2, node.y + node.height
    else:
        return node.x + node.width / 2, node.y + node.height / 2


def get_outline_item_connection_point(
    node: Node, item_index: int, direction: str = "right", config: LayoutConfig | None = None
) -> tuple[float, float]:
    """Get connection point for an outline item."""
    if config is None:
        config = LayoutConfig()

    # Calculate y position of the outline item
    base_y = node.y + config.header_height
    if node.description:
        base_y += config.description_height

    item_y = base_y + item_index * config.outline_item_height + config.outline_item_height / 2

    if direction == "right":
        return node.x + node.width, item_y
    else:
        return node.x, item_y
