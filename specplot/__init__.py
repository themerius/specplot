"""specplot - Architecture diagramming with Python DSL.

Example usage:
    from specplot import diagram, node, edge, GROUP, OUTLINE

    with diagram(filename="architecture"):
        user = node(icon="person", label="User")

        with node(icon="cloud", label="Cloud", show_as=GROUP) as cloud:
            web = node(icon="web", label="Web Server")
            db = node(icon="database", label="Database")

        user >> web
        web >> db | "SQL"
"""

from .dsl import (
    GROUP,
    OUTLINE,
    NodeContext,
    diagram,
    edge,
    node,
)
from .models import (
    Diagram,
    Edge,
    EdgeStyle,
    Node,
    OutlineItem,
    ShowAs,
)
from .pathfinding import (
    PathfindingConfig,
)
from .renderer import (
    DEFAULT_THEME,
    DiagramRenderer,
    Theme,
    render_to_svg,
)

__version__ = "0.1.0"

__all__ = [
    # DSL functions
    "diagram",
    "node",
    "edge",
    "NodeContext",
    # Constants (backward compatibility)
    "GROUP",
    "OUTLINE",
    # Models
    "Diagram",
    "Node",
    "Edge",
    "OutlineItem",
    "EdgeStyle",
    "ShowAs",
    # Pathfinding
    "PathfindingConfig",
    # Rendering
    "render_to_svg",
    "DiagramRenderer",
    "Theme",
    "DEFAULT_THEME",
    # Version
    "__version__",
]
