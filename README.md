# swe-coding-agent

A multi-turn, ReAct-style coding agent. Give it a GitHub issue or a plain-English
task and it plans the work, reads and edits files, runs tests and shell commands
in a Docker sandbox, and submits a patch. Built on Python and the Microsoft Agent
Framework, with Redis-backed trajectory memory and an evaluation harness that
scores the full agent against a single-shot baseline on success rate, average
turns, and token cost.

**Status: Work in Progress** — currently on M0 (scaffolding). See
[planning.md](planning.md) for the milestone plan.
