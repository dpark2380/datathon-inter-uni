## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## superpowers vs. ponytail

Both `ponytail` and `superpowers` are active in this project (project-scoped plugins). They pull in opposite directions — ponytail is YAGNI/minimal-process, superpowers wants brainstorm → spec → plan → TDD → subagent-driven execution for everything.

Rule for this project: **ponytail wins for datathon work.** This is a time-boxed datathon — EDA, sentiment/theme extraction scripts, and one-off analysis notebooks should stay in ponytail's lazy/minimal mode, not go through superpowers' full methodology. Only reach for superpowers' heavier skills (TDD, subagent-driven-development, formal planning) if we build something reusable/production-like that other parts of the project will depend on (e.g. a shared data-loading or scoring pipeline used across multiple notebooks).

## api-and-interface-design (deferred)

Not relevant yet — the project is notebook/script-based analysis with no public interface to design. Don't apply this skill to internal analysis code.

Trigger it later only if the deliverable grows a live component with a real interface, e.g. a small API serving sentiment predictions to a dashboard, or a module boundary other people's code calls into. Until then, this stays deferred per ponytail (no interface contracts for code nobody else calls).
