"""Grid-based A* pathfinding for edge routing using NetworkX."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import networkx as nx

if TYPE_CHECKING:
    from .models import Diagram, Node


@dataclass
class PathfindingConfig:
    """Configuration for pathfinding-based edge routing."""

    enabled: bool = True
    grid_spacing: float = 15.0
    distance_weight: float = 1.0
    proximity_penalty_weight: float = 0.5
    path_style: Literal["smooth", "orthogonal"] = "smooth"
    # Margin around obstacles for proximity penalty
    obstacle_margin: float = 30.0
    # Douglas-Peucker simplification tolerance for smooth paths
    simplification_tolerance: float = 5.0


@dataclass
class VirtualNode:
    """A point in the virtual routing grid."""

    x: float
    y: float
    grid_row: int
    grid_col: int
    is_blocked: bool = False
    is_boundary: bool = False
    attached_node: Node | None = None
    attachment_side: str | None = None  # "left", "right", "top", "bottom"
    snapping_weight: float = 1.0  # Lower is better for snapping


@dataclass
class RoutedPath:
    """A computed path between two connection points."""

    points: list[tuple[float, float]]
    source_side: str
    target_side: str
    # For smooth paths, these are the Bezier control points
    control_points: list[tuple[float, float]] | None = None


class VirtualGrid:
    """Routing grid with NetworkX graph for A* pathfinding."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        config: PathfindingConfig,
    ):
        """Initialize the grid.

        Args:
            bounds: (min_x, min_y, max_x, max_y) diagram bounds
            config: Pathfinding configuration
        """
        self.bounds = bounds
        self.config = config
        self.nodes: dict[tuple[int, int], VirtualNode] = {}
        self.graph: nx.Graph = nx.Graph()
        self._node_obstacles: list[tuple[float, float, float, float]] = []
        self._snapping_points: dict[int, dict[str, list[VirtualNode]]] = {}

    @classmethod
    def generate(
        cls,
        diagram: Diagram,
        config: PathfindingConfig,
    ) -> VirtualGrid:
        """Build grid after layout, mark obstacles, compute snapping weights.

        Args:
            diagram: Laid-out diagram
            config: Pathfinding configuration

        Returns:
            Populated VirtualGrid ready for pathfinding
        """
        # Compute diagram bounds with padding
        bounds = cls._compute_bounds(diagram, padding=config.grid_spacing * 2)
        grid = cls(bounds, config)

        # Create grid of virtual nodes
        grid._create_grid()

        # Collect all node bounds (including nested)
        grid._collect_obstacles(diagram)

        # Mark blocked cells and boundaries
        grid._mark_obstacles_and_boundaries()

        # Build the NetworkX graph
        grid._build_graph()

        # Compute snapping points for each node
        grid._compute_snapping_points(diagram)

        return grid

    @staticmethod
    def _compute_bounds(
        diagram: Diagram,
        padding: float,
    ) -> tuple[float, float, float, float]:
        """Compute diagram bounds including all nodes."""
        if not diagram.nodes:
            return (0, 0, 200, 200)

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        def update_bounds(node: Node) -> None:
            nonlocal min_x, min_y, max_x, max_y
            min_x = min(min_x, node.x)
            min_y = min(min_y, node.y)
            max_x = max(max_x, node.x + node.width)
            max_y = max(max_y, node.y + node.height)
            for child in node.children:
                update_bounds(child)

        for node in diagram.nodes:
            update_bounds(node)

        return (
            min_x - padding,
            min_y - padding,
            max_x + padding,
            max_y + padding,
        )

    def _create_grid(self) -> None:
        """Create the grid of virtual nodes."""
        min_x, min_y, max_x, max_y = self.bounds
        spacing = self.config.grid_spacing

        rows = int((max_y - min_y) / spacing) + 1
        cols = int((max_x - min_x) / spacing) + 1

        for row in range(rows):
            for col in range(cols):
                x = min_x + col * spacing
                y = min_y + row * spacing
                vnode = VirtualNode(
                    x=x,
                    y=y,
                    grid_row=row,
                    grid_col=col,
                )
                self.nodes[(row, col)] = vnode

    def _collect_obstacles(self, diagram: Diagram) -> None:
        """Collect all node bounding boxes as obstacles."""
        from .models import ShowAs

        def collect_node(node: Node, is_group_child: bool = False) -> None:
            # Add node bounds as obstacle
            self._node_obstacles.append((
                node.x,
                node.y,
                node.x + node.width,
                node.y + node.height,
            ))

            # For groups, recurse into children
            if node.children and node.show_as == ShowAs.GROUP:
                for child in node.children:
                    collect_node(child, is_group_child=True)

        for node in diagram.nodes:
            collect_node(node)

    def _mark_obstacles_and_boundaries(self) -> None:
        """Mark grid cells as blocked or boundary based on obstacles."""
        spacing = self.config.grid_spacing

        for (row, col), vnode in self.nodes.items():
            x, y = vnode.x, vnode.y

            for ox1, oy1, ox2, oy2 in self._node_obstacles:
                # Check if point is inside the obstacle (with small inset)
                inset = spacing * 0.3
                if (ox1 + inset < x < ox2 - inset and
                    oy1 + inset < y < oy2 - inset):
                    vnode.is_blocked = True
                    break

                # Check if point is on the boundary (within one grid cell)
                is_near_left = abs(x - ox1) < spacing and oy1 - spacing < y < oy2 + spacing
                is_near_right = abs(x - ox2) < spacing and oy1 - spacing < y < oy2 + spacing
                is_near_top = abs(y - oy1) < spacing and ox1 - spacing < x < ox2 + spacing
                is_near_bottom = abs(y - oy2) < spacing and ox1 - spacing < x < ox2 + spacing

                if is_near_left or is_near_right or is_near_top or is_near_bottom:
                    vnode.is_boundary = True

    def _build_graph(self) -> None:
        """Create NetworkX graph with weighted edges."""
        spacing = self.config.grid_spacing

        # Add all non-blocked nodes to the graph
        for (row, col), vnode in self.nodes.items():
            if not vnode.is_blocked:
                self.graph.add_node((row, col), vnode=vnode)

        # Add edges to neighbors (4-directional + 4 diagonals)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonal
        ]

        for (row, col), vnode in self.nodes.items():
            if vnode.is_blocked:
                continue

            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                neighbor = self.nodes.get((nr, nc))

                if neighbor and not neighbor.is_blocked:
                    # Calculate edge weight
                    is_diagonal = dr != 0 and dc != 0
                    base_distance = spacing * (1.414 if is_diagonal else 1.0)
                    weight = self._calculate_edge_weight(
                        vnode, neighbor, base_distance
                    )
                    self.graph.add_edge((row, col), (nr, nc), weight=weight)

    def _calculate_edge_weight(
        self,
        from_node: VirtualNode,
        to_node: VirtualNode,
        base_distance: float,
    ) -> float:
        """Calculate edge weight with distance and proximity penalty."""
        weight = base_distance * self.config.distance_weight

        # Add proximity penalty for nodes near obstacles
        margin = self.config.obstacle_margin
        min_dist = float("inf")

        mid_x = (from_node.x + to_node.x) / 2
        mid_y = (from_node.y + to_node.y) / 2

        for ox1, oy1, ox2, oy2 in self._node_obstacles:
            # Distance to obstacle edge
            dx = max(ox1 - mid_x, 0, mid_x - ox2)
            dy = max(oy1 - mid_y, 0, mid_y - oy2)
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = min(min_dist, dist)

        if min_dist < margin:
            penalty = self.config.proximity_penalty_weight * (margin - min_dist) / margin
            weight *= (1 + penalty)

        return weight

    def _compute_snapping_points(self, diagram: Diagram) -> None:
        """Compute weighted snapping points for each node."""
        from .models import ShowAs

        def process_node(node: Node) -> None:
            node_id = id(node)
            self._snapping_points[node_id] = {
                "left": [],
                "right": [],
                "top": [],
                "bottom": [],
            }

            # Find boundary nodes adjacent to each side
            for (row, col), vnode in self.nodes.items():
                if vnode.is_blocked:
                    continue

                x, y = vnode.x, vnode.y
                spacing = self.config.grid_spacing

                # Check proximity to each side
                # Left side
                if abs(x - node.x) < spacing * 1.5 and node.y <= y <= node.y + node.height:
                    distance_from_center = abs(y - (node.y + node.height / 2))
                    sigma = node.height / 4
                    weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))
                    vnode_copy = VirtualNode(
                        x=x, y=y, grid_row=row, grid_col=col,
                        is_boundary=True, attached_node=node,
                        attachment_side="left", snapping_weight=1 - weight,
                    )
                    self._snapping_points[node_id]["left"].append(vnode_copy)

                # Right side
                if abs(x - (node.x + node.width)) < spacing * 1.5 and node.y <= y <= node.y + node.height:
                    distance_from_center = abs(y - (node.y + node.height / 2))
                    sigma = node.height / 4
                    weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))
                    vnode_copy = VirtualNode(
                        x=x, y=y, grid_row=row, grid_col=col,
                        is_boundary=True, attached_node=node,
                        attachment_side="right", snapping_weight=1 - weight,
                    )
                    self._snapping_points[node_id]["right"].append(vnode_copy)

                # Top side
                if abs(y - node.y) < spacing * 1.5 and node.x <= x <= node.x + node.width:
                    distance_from_center = abs(x - (node.x + node.width / 2))
                    sigma = node.width / 4
                    weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))
                    vnode_copy = VirtualNode(
                        x=x, y=y, grid_row=row, grid_col=col,
                        is_boundary=True, attached_node=node,
                        attachment_side="top", snapping_weight=1 - weight,
                    )
                    self._snapping_points[node_id]["top"].append(vnode_copy)

                # Bottom side
                if abs(y - (node.y + node.height)) < spacing * 1.5 and node.x <= x <= node.x + node.width:
                    distance_from_center = abs(x - (node.x + node.width / 2))
                    sigma = node.width / 4
                    weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))
                    vnode_copy = VirtualNode(
                        x=x, y=y, grid_row=row, grid_col=col,
                        is_boundary=True, attached_node=node,
                        attachment_side="bottom", snapping_weight=1 - weight,
                    )
                    self._snapping_points[node_id]["bottom"].append(vnode_copy)

            # Sort snapping points by weight (lower is better)
            for side in self._snapping_points[node_id]:
                self._snapping_points[node_id][side].sort(key=lambda v: v.snapping_weight)

            # Process children for groups
            if node.children and node.show_as == ShowAs.GROUP:
                for child in node.children:
                    process_node(child)

        for node in diagram.nodes:
            process_node(node)

    def find_path(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[float, float]] | None:
        """Find path using A* with NetworkX.

        Args:
            start: Grid coordinates (row, col) of start
            end: Grid coordinates (row, col) of end

        Returns:
            List of (x, y) world coordinates, or None if no path
        """
        if start not in self.graph or end not in self.graph:
            return None

        try:
            # A* with Euclidean heuristic
            def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
                va = self.nodes[a]
                vb = self.nodes[b]
                return math.sqrt((va.x - vb.x) ** 2 + (va.y - vb.y) ** 2)

            path = nx.astar_path(
                self.graph,
                start,
                end,
                heuristic=heuristic,
                weight="weight",
            )

            # Convert to world coordinates
            return [(self.nodes[p].x, self.nodes[p].y) for p in path]

        except nx.NetworkXNoPath:
            return None

    def select_snapping_point(
        self,
        node: Node,
        side: str,
        occupied: set[tuple[int, int]],
        target_y: float | None = None,
    ) -> VirtualNode | None:
        """Select best snapping point for a node side.

        Args:
            node: The node to snap to
            side: Which side ("left", "right", "top", "bottom")
            occupied: Set of already-occupied grid positions
            target_y: Optional target y coordinate (for outline item connections)

        Returns:
            Best available VirtualNode for snapping
        """
        node_id = id(node)
        if node_id not in self._snapping_points:
            return None

        candidates = self._snapping_points[node_id].get(side, [])

        # Filter out occupied positions
        available = [
            v for v in candidates
            if (v.grid_row, v.grid_col) not in occupied
        ]

        if not available:
            # Fall back to any candidate if all occupied
            available = candidates

        if not available:
            return None

        # If target_y is specified, prefer points closer to it
        if target_y is not None:
            available.sort(key=lambda v: abs(v.y - target_y) + v.snapping_weight * 10)
        else:
            # Just use weight
            available.sort(key=lambda v: v.snapping_weight)

        return available[0] if available else None

    def get_nearest_grid_point(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int] | None:
        """Get nearest non-blocked grid point to world coordinates."""
        spacing = self.config.grid_spacing
        min_x, min_y, _, _ = self.bounds

        # Calculate nearest grid position
        col = round((x - min_x) / spacing)
        row = round((y - min_y) / spacing)

        # Search in expanding radius for non-blocked point
        for radius in range(10):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    pos = (row + dr, col + dc)
                    if pos in self.nodes and not self.nodes[pos].is_blocked:
                        return pos

        return None


def simplify_path_douglas_peucker(
    points: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    """Simplify path using Douglas-Peucker algorithm.

    Args:
        points: List of (x, y) points
        tolerance: Maximum perpendicular distance to keep

    Returns:
        Simplified list of points
    """
    if len(points) <= 2:
        return points

    # Find point with maximum distance from line between first and last
    first = points[0]
    last = points[-1]

    max_dist = 0
    max_idx = 0

    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], first, last)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # If max distance is greater than tolerance, recursively simplify
    if max_dist > tolerance:
        left = simplify_path_douglas_peucker(points[: max_idx + 1], tolerance)
        right = simplify_path_douglas_peucker(points[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [first, last]


def _perpendicular_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    """Calculate perpendicular distance from point to line."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def compute_bezier_control_points(
    points: list[tuple[float, float]],
    tension: float = 0.3,
) -> list[tuple[float, float]]:
    """Compute smooth Bezier control points for a path.

    Uses Catmull-Rom spline converted to cubic Bezier for smooth curves.

    Args:
        points: Simplified path points
        tension: Curve tension (0 = angular, 1 = very smooth)

    Returns:
        List of control points for SVG path
    """
    if len(points) < 2:
        return points

    if len(points) == 2:
        # Simple line - use midpoint for control
        mx = (points[0][0] + points[1][0]) / 2
        my = (points[0][1] + points[1][1]) / 2
        return [points[0], (mx, my), (mx, my), points[1]]

    result = [points[0]]

    for i in range(1, len(points)):
        p0 = points[max(0, i - 2)]
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[min(len(points) - 1, i + 1)]

        # Catmull-Rom to Bezier conversion
        c1x = p1[0] + (p2[0] - p0[0]) * tension / 3
        c1y = p1[1] + (p2[1] - p0[1]) * tension / 3
        c2x = p2[0] - (p3[0] - p1[0]) * tension / 3
        c2y = p2[1] - (p3[1] - p1[1]) * tension / 3

        result.extend([(c1x, c1y), (c2x, c2y), p2])

    return result


def extract_orthogonal_waypoints(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Extract waypoints for orthogonal (right-angle) paths.

    Converts a free-form path to one that only uses horizontal
    and vertical segments.

    Args:
        points: Original path points

    Returns:
        List of waypoints with only H/V segments
    """
    if len(points) < 2:
        return points

    result = [points[0]]
    current = points[0]

    for next_point in points[1:]:
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]

        # Determine dominant direction
        if abs(dx) >= abs(dy):
            # Horizontal first, then vertical
            mid = (next_point[0], current[1])
        else:
            # Vertical first, then horizontal
            mid = (current[0], next_point[1])

        # Only add intermediate point if it's different
        if mid != current and mid != next_point:
            result.append(mid)
        result.append(next_point)
        current = next_point

    return result
