# Comparison with Alternatives

## Overview

| Tool | Approach | Output | Hierarchical | Version Control | LLM-Friendly |
|------|----------|--------|--------------|-----------------|--------------|
| **specplot** | Python DSL | SVG | Native (outline/group) | Excellent | Excellent |
| diagrams | Python DSL | PNG/SVG | Limited | Excellent | Good |
| Mermaid | Markdown DSL | SVG | Limited | Good | Good |
| PlantUML | Text DSL | PNG/SVG | Limited | Good | Moderate |
| Structurizr | Java/DSL | Multiple | C4 Model | Good | Moderate |
| draw.io | GUI | XML/SVG | Manual | Poor | Poor |

## Detailed Comparison

### diagrams

[diagrams](https://github.com/mingrammer/diagrams) is an excellent Python library that inspired specplot.

**Strengths:**
- Beautiful cloud provider icons
- Simple, intuitive API
- Good for cloud infrastructure diagrams

**Where specplot differs:**
- Native outline/group hierarchy for nested architectures
- Publication-quality SVG with clean vector output
- Designed for architecture *specs*, not just infrastructure
- Flexible layout system with zones and grids

### Mermaid

[Mermaid](https://mermaid.js.org/) is widely adopted, especially in Markdown-based documentation.

**Strengths:**
- Embedded directly in Markdown
- GitHub renders it natively
- Good for simple flowcharts and sequences

**Where specplot differs:**
- Full programming language (Python) for complex logic
- Hierarchical outline mode for nested structures
- Better control over layout and styling
- Professional typography and spacing

### PlantUML

[PlantUML](https://plantuml.com/) is a veteran in text-based diagramming.

**Strengths:**
- Mature, feature-rich
- Wide IDE support
- Many diagram types (UML, C4, etc.)

**Where specplot differs:**
- Cleaner Python syntax vs PlantUML's custom DSL
- Modern, minimal visual style
- Better suited for AI-assisted workflows

### Structurizr

[Structurizr](https://structurizr.com/) implements the C4 model as code.

**Strengths:**
- Full C4 model support
- Multiple export formats
- Architecture decision records

**Where specplot differs:**
- Lightweight, single-purpose library
- No cloud service required
- Outline mode for rapid ideation
- Simpler learning curve

### draw.io / Lucidchart / Visio

GUI-based diagramming tools.

**Strengths:**
- Intuitive drag-and-drop
- Real-time collaboration
- Rich formatting options

**Where specplot differs:**
- Diagrams live in your codebase
- Changes can be tracked in git
- Reproducible, automated generation
- No context switching from IDE

## When to Use specplot

specplot is ideal when you need:

- Architecture diagrams that evolve with your code
- Publication-quality output for papers or documentation
- Hierarchical views of complex systems
- AI-assisted diagram generation or modification
- Version-controlled, reviewable architecture specs

## When to Use Something Else

Consider alternatives when:

- You need real-time visual collaboration → draw.io, Miro
- You want embedded Markdown diagrams → Mermaid
- You need cloud provider icons → diagrams
- You need full UML compliance → PlantUML
- You want C4 model enforcement → Structurizr
