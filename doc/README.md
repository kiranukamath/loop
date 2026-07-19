# Loop — learning docs

This folder captures the teaching content for each phase of Loop, written up
after the fact so you can re-read the concepts without scrolling back through
a chat transcript. See [`PLAN.md`](../PLAN.md) for the live progress tracker
and the original phase specs; these docs are the "walkthrough" companion —
what was built, why, the concepts behind it, and the design tradeoffs made
along the way.

| Doc | Phase | Capability |
|---|---|---|
| [phase-10-production-hardening.md](phase-10-production-hardening.md) | 10 | Resilience, guardrails, cost budgeting |
| [phase-11-session-history.md](phase-11-session-history.md) | 11 | Checkpoint replay / observability |
| [phase-12-mcp-interop.md](phase-12-mcp-interop.md) | 12 | Model Context Protocol — Loop as MCP server + client |
| [phase-13-multiagent-orchestration.md](phase-13-multiagent-orchestration.md) | 13 | Supervisor pattern · parallel fan-out/fan-in (Send) · Command handoffs |

Phases 0–9 were built in earlier sessions — their teaching content lives in
the conversation history from those sessions and in the code comments
throughout `loop/`. Docs for phases 10–11 are written here because they
closed out the full 12-phase v1+production-track plan (0–11) in one sitting.
Phase 12 is the start of the v2 "frontier track" — see `PLAN.md` for the full
roadmap. The underlying MCP *concepts* (protocol, roles/primitives,
transports) live in [`../docs/README.md`](../docs/README.md#phase-12--mcp--interoperability);
this file is the "what we built" walkthrough. Same split for Phase 13: the
multi-agent *concepts* (fixed workflow vs. multi-agent, Send/map-reduce,
Command handoffs + supervisor, reducers + bounded agency) live in
[`../docs/README.md`](../docs/README.md#phase-13--multi-agent-orchestration).
