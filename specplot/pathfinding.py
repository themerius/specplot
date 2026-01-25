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
    # Debug mode: render virtual nodes in the diagram
    debug: bool = False


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
        """Compute snapping points exactly on each node's border.

        Creates virtual nodes positioned exactly on the node edges,
        with positions distributed along the side for multi-edge support.
        """
        from .models import ShowAs

        spacing = self.config.grid_spacing

        def process_node(node: Node) -> None:
            node_id = id(node)
            self._snapping_points[node_id] = {
                "left": [],
                "right": [],
                "top": [],
                "bottom": [],
            }

            # Create snapping points exactly on borders
            # Number of points per side based on side length
            num_points_v = max(3, int(node.height / spacing))
            num_points_h = max(3, int(node.width / spacing))

            # Left side - points exactly at x = node.x
            for i in range(num_points_v):
                # Distribute points along the height
                t = (i + 0.5) / num_points_v  # 0.5 offset to avoid corners
                y = node.y + t * node.height
                x = node.x  # Exactly on the border

                # Find nearest grid point for pathfinding
                grid_pos = self.get_nearest_grid_point(x - spacing, y)
                if grid_pos is None:
                    grid_pos = (0, 0)

                # Weight based on distance from center (Gaussian)
                center_y = node.y + node.height / 2
                distance_from_center = abs(y - center_y)
                sigma = node.height / 6  # sigma = side_length / 6
                weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))

                vnode = VirtualNode(
                    x=x, y=y,
                    grid_row=grid_pos[0], grid_col=grid_pos[1],
                    is_boundary=True, attached_node=node,
                    attachment_side="left", snapping_weight=1 - weight,
                )
                self._snapping_points[node_id]["left"].append(vnode)

            # Right side - points exactly at x = node.x + node.width
            for i in range(num_points_v):
                t = (i + 0.5) / num_points_v
                y = node.y + t * node.height
                x = node.x + node.width  # Exactly on the border

                grid_pos = self.get_nearest_grid_point(x + spacing, y)
                if grid_pos is None:
                    grid_pos = (0, 0)

                center_y = node.y + node.height / 2
                distance_from_center = abs(y - center_y)
                sigma = node.height / 6
                weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))

                vnode = VirtualNode(
                    x=x, y=y,
                    grid_row=grid_pos[0], grid_col=grid_pos[1],
                    is_boundary=True, attached_node=node,
                    attachment_side="right", snapping_weight=1 - weight,
                )
                self._snapping_points[node_id]["right"].append(vnode)

            # Top side - points exactly at y = node.y
            for i in range(num_points_h):
                t = (i + 0.5) / num_points_h
                x = node.x + t * node.width
                y = node.y  # Exactly on the border

                grid_pos = self.get_nearest_grid_point(x, y - spacing)
                if grid_pos is None:
                    grid_pos = (0, 0)

                center_x = node.x + node.width / 2
                distance_from_center = abs(x - center_x)
                sigma = node.width / 6
                weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))

                vnode = VirtualNode(
                    x=x, y=y,
                    grid_row=grid_pos[0], grid_col=grid_pos[1],
                    is_boundary=True, attached_node=node,
                    attachment_side="top", snapping_weight=1 - weight,
                )
                self._snapping_points[node_id]["top"].append(vnode)

            # Bottom side - points exactly at y = node.y + node.height
            for i in range(num_points_h):
                t = (i + 0.5) / num_points_h
                x = node.x + t * node.width
                y = node.y + node.height  # Exactly on the border

                grid_pos = self.get_nearest_grid_point(x, y + spacing)
                if grid_pos is None:
                    grid_pos = (0, 0)

                center_x = node.x + node.width / 2
                distance_from_center = abs(x - center_x)
                sigma = node.width / 6
                weight = math.exp(-(distance_from_center ** 2) / (2 * sigma ** 2))

                vnode = VirtualNode(
                    x=x, y=y,
                    grid_row=grid_pos[0], grid_col=grid_pos[1],
                    is_boundary=True, attached_node=node,
                    attachment_side="bottom", snapping_weight=1 - weight,
                )
                self._snapping_points[node_id]["bottom"].append(vnode)

            # Sort snapping points by position (for consistent distribution)
            # Left/right: sort by y position
            self._snapping_points[node_id]["left"].sort(key=lambda v: v.y)
            self._snapping_points[node_id]["right"].sort(key=lambda v: v.y)
            # Top/bottom: sort by x position
            self._snapping_points[node_id]["top"].sort(key=lambda v: v.x)
            self._snapping_points[node_id]["bottom"].sort(key=lambda v: v.x)

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

    def select_distributed_snapping_point(
        self,
        node: Node,
        side: str,
        edge_index: int,
        total_edges: int,
    ) -> VirtualNode | None:
        """Select snapping point using Gaussian distribution for multiple edges.

        Distributes edges along the side using sigma-based positions:
        - 1 edge: center (0 sigma)
        - 2 edges: -1 sigma, +1 sigma
        - 3 edges: -1 sigma, 0, +1 sigma
        - 4 edges: -1.5 sigma, -0.5 sigma, +0.5 sigma, +1.5 sigma
        - etc.

        Args:
            node: The node to snap to
            side: Which side ("left", "right", "top", "bottom")
            edge_index: Index of this edge (0-based, sorted by source position)
            total_edges: Total number of edges on this side

        Returns:
            VirtualNode at the distributed position
        """
        node_id = id(node)
        if node_id not in self._snapping_points:
            return None

        candidates = self._snapping_points[node_id].get(side, [])
        if not candidates:
            return None

        # Calculate the target position based on Gaussian distribution
        # sigma = side_length / 6, so we have ~3 sigma on each side of center
        if side in ("left", "right"):
            side_length = node.height
            center = node.y + node.height / 2
        else:
            side_length = node.width
            center = node.x + node.width / 2

        sigma = side_length / 6

        # Calculate the sigma offset for this edge
        if total_edges == 1:
            sigma_offset = 0
        else:
            # Distribute edges symmetrically around center
            # For n edges, use positions: -(n-1)/2, ..., -0.5, 0.5, ..., (n-1)/2 for even
            # Or: -(n-1)/2, ..., -1, 0, 1, ..., (n-1)/2 for odd
            half = (total_edges - 1) / 2
            sigma_offset = (edge_index - half)

        # Calculate target coordinate
        target_offset = sigma_offset * sigma
        if side in ("left", "right"):
            target_pos = center + target_offset
            # Find closest snapping point by y position
            candidates_sorted = sorted(candidates, key=lambda v: abs(v.y - target_pos))
        else:
            target_pos = center + target_offset
            # Find closest snapping point by x position
            candidates_sorted = sorted(candidates, key=lambda v: abs(v.x - target_pos))

        return candidates_sorted[0] if candidates_sorted else None

    def get_snapping_points_for_side(
        self,
        node: Node,
        side: str,
    ) -> list[VirtualNode]:
        """Get all snapping points for a node side."""
        node_id = id(node)
        if node_id not in self._snapping_points:
            return []
        return self._snapping_points[node_id].get(side, [])

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


def compute_path_center(
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    """Compute the center point of a path using arc length.

    Finds the point that is equidistant (by path length) from both endpoints.

    Args:
        points: List of path points

    Returns:
        (x, y) coordinates of the path center
    """
    if len(points) < 2:
        return points[0] if points else (0, 0)

    if len(points) == 2:
        return (
            (points[0][0] + points[1][0]) / 2,
            (points[0][1] + points[1][1]) / 2,
        )

    # Calculate cumulative arc lengths
    arc_lengths = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        segment_length = math.sqrt(dx * dx + dy * dy)
        arc_lengths.append(arc_lengths[-1] + segment_length)

    total_length = arc_lengths[-1]
    if total_length == 0:
        return points[0]

    target_length = total_length / 2

    # Find the segment containing the center
    for i in range(1, len(arc_lengths)):
        if arc_lengths[i] >= target_length:
            # Interpolate within this segment
            segment_start = arc_lengths[i - 1]
            segment_end = arc_lengths[i]
            segment_length = segment_end - segment_start

            if segment_length == 0:
                return points[i - 1]

            t = (target_length - segment_start) / segment_length
            x = points[i - 1][0] + t * (points[i][0] - points[i - 1][0])
            y = points[i - 1][1] + t * (points[i][1] - points[i - 1][1])
            return (x, y)

    # Fallback to last point
    return points[-1]


def compute_bezier_path_center(
    control_points: list[tuple[float, float]],
) -> tuple[float, float]:
    """Compute the center point of a Bezier path.

    Samples the Bezier curve and finds the point at t=0.5 of total arc length.

    Args:
        control_points: Bezier control points (start, c1, c2, end, c1, c2, end, ...)

    Returns:
        (x, y) coordinates of the path center
    """
    if len(control_points) < 4:
        return compute_path_center(list(control_points))

    # Sample the Bezier curve to get approximate points
    sample_points = []
    num_segments = (len(control_points) - 1) // 3

    for seg in range(num_segments):
        base = seg * 3
        if base + 3 >= len(control_points):
            break

        p0 = control_points[base]
        c1 = control_points[base + 1]
        c2 = control_points[base + 2]
        p3 = control_points[base + 3]

        # Sample this segment
        samples_per_segment = 10
        for i in range(samples_per_segment + 1):
            t = i / samples_per_segment
            # Cubic Bezier formula
            mt = 1 - t
            x = mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * p3[0]
            y = mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * p3[1]
            sample_points.append((x, y))

    return compute_path_center(sample_points)
