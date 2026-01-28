# Philosophy

## Why specplot?

specplot was born from frustration with existing architecture diagramming tools. Every option seemed to lack something essential — the flexibility to iterate quickly, the formalism to maintain consistency, or the output quality needed for professional publications.

## The Power of Outlining

Outlining is one of the most powerful techniques for structured thinking:

- **Hierarchical by nature** — Complex systems decompose naturally into nested components
- **Iterative refinement** — Start rough, add detail progressively
- **Living documentation** — Outlines evolve with your understanding
- **Parallel structure** — Reveals patterns and inconsistencies

Software architecture is inherently hierarchical. Systems contain subsystems. Services contain modules. Modules contain functions. The outline metaphor maps directly to how architects think about structure.

## Code as Diagram

When your diagram *is* code:

- **Version control** — Track changes, branch, merge, review
- **Automation** — Generate diagrams in CI/CD pipelines
- **Refactoring** — Rename, extract, restructure with confidence
- **Testing** — Validate diagram consistency programmatically
- **Collaboration** — Pull requests for architecture changes

## LLM-Native Design

Both the Python DSL and SVG output are text-based formats that AI systems can read, write, and reason about:

- **Generate** — Describe an architecture, get working diagram code
- **Explain** — AI can read and summarize existing diagrams
- **Modify** — "Add a cache layer between API and database"
- **Convert** — Transform between diagram formats

## Design Principles

1. **Simplicity over features** — A focused DSL beats a kitchen sink
2. **Text over binary** — Everything should be readable and diffable
3. **Standards over proprietary** — SVG is universal, open, and future-proof
4. **Code over configuration** — Python gives you full programming power when needed

## Inspiration

specplot draws inspiration from:

- [diagrams](https://github.com/mingrammer/diagrams) — Diagram as Code philosophy
- [Mermaid](https://mermaid.js.org/) — Text-based diagram generation
- [C4 Model](https://c4model.com/) — Hierarchical architecture documentation
- [Structurizr](https://structurizr.com/) — Architecture as code approach
