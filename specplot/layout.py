"""Layout algorithms for specplot diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Diagram, Node


@dataclass
class LayoutConfig:
    """Configuration for layout calculations."""

    # Fixed node width for consistent appearance
    node_width: float = 200
    node_padding: float = 12
    header_height: float = 40
    description_height: float = 20  # Single line description
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

    # Second pass: position top-level nodes
    # Simple horizontal layout for now
    x = diagram.padding
    y = diagram.padding
    max_height = 0.0

    for node in diagram.nodes:
        node.x = x
        node.y = y
        max_height = max(max_height, node.height)
        x += node.width + config.node_spacing_h

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
            row_heights = [0.0] * rows

            for i, (cw, ch) in enumerate(child_dims):
                row = i // cols
                col = i % cols
                if row < rows:
                    col_widths[col] = max(col_widths[col], cw)
                    row_heights[row] = max(row_heights[row], ch)

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
                current_y += row_heights[row] + config.node_spacing_v

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
