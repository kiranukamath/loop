# Phase 11 — Session history UI

**Capability taught:** reading back from the checkpoint — LangGraph's
`get_state_history()` / `get_state()`, the same mechanism that powers
time-travel debugging and audit logs in production agentic systems.

## Why this phase exists

Since Phase 4/5, every graph run has been checkpointed to SQLite after every
node (`SqliteSaver`, wired in `loop/memory.py`). That checkpoint already
holds the complete plan, every answer, every grade, and the readiness verdict
for every session that's ever run — nothing new needs to be *stored*. Phase
11 is purely about *reading it back out* and surfacing it as a UI, without
adding a single new table or duplicating any data.

This mirrors a pattern you already know from `git log`: every commit is a
full, independently-inspectable snapshot. You don't need a separate "history
service" bolted onto the side of your repo — the commits themselves *are*
the history. LangGraph's checkpoints work the same way.

---

## The checkpoint mental model

`compiled_graph.get_state_history(config)` returns a generator of
`StateSnapshot` objects, **newest first**. Each one has:

- **`.values`** — the full state dict at that point in the run
- **`.next`** — which node runs next from here (empty tuple `()` if the run
  is finished)
- **`.created_at`** — an ISO timestamp
- **`.metadata`** — step number, source, parent info

### Why the newest snapshot already has everything

Every node function returns only the *delta* it changed — e.g. `grader()`
returns `{"grades": [new_grade]}`, not the whole state. But the *checkpoint*
LangGraph writes after that node stores the **merged** state (old state +
delta, via each field's reducer — plain overwrite for most fields,
`_append_list` for `answers`/`grades`/`flagged_inputs`, `add_messages` for
`messages`). So by the time the graph reaches its last checkpoint, that one
snapshot already has the fully accumulated `plan`, `answers`, `grades`,
`weak_areas`, and `readiness_verdict` from the *entire* run — no need to walk
every intermediate snapshot and manually replay the deltas.

This is exactly why `get_session_history()` in `api.py` only looks at
`next(get_state_history(config), None)` — the first item the generator
yields — rather than iterating the whole history.

```python
history = _graph.get_state_history(config)
snapshot = next(history, None)   # newest = fully accumulated state
```

---

## 11a — Two new API endpoints

### `GET /sessions` — listing

LangGraph has **no built-in "list every thread" API** — a checkpointer only
knows how to save/load state for a thread_id you already have. So listing
requires going one layer lower and querying the checkpointer's own storage
directly.

For `SqliteSaver`, that's a `checkpoints` table. Inspecting a live one
(`PRAGMA table_info(checkpoints)`) shows columns `thread_id`, `checkpoint_ns`,
`checkpoint_id`, `checkpoint` (a serialized blob), `metadata` — **no
timestamp column**. The `checkpoint_id` itself, though, turned out to be a
UUID6 — a time-ordered UUID variant that's lexicographically sortable by
creation time. So:

```sql
SELECT thread_id, MAX(checkpoint_id) AS latest
FROM checkpoints
WHERE checkpoint_ns = ''
GROUP BY thread_id
ORDER BY latest DESC
```

...gives the most-recently-active thread_id first, using plain string
comparison — no need to decode the UUID6's embedded timestamp. For the
actual ISO timestamp shown in the UI, `graph.get_state(config)` (not
`get_state_history` — no need for the full generator when you only want the
latest snapshot) exposes `.created_at` directly on the `StateSnapshot`.

Response shape:

```json
{
  "sessions": [
    {"thread_id": "...", "started_at": "...", "verdict": "ready",
     "sessions_completed": 2, "total_sessions": 6}
  ],
  "persistence": "sqlite"
}
```

**MemorySaver case:** if `_graph.checkpointer` isn't a `SqliteSaver`
(i.e. `DB_PATH` is unset — the dev-box default), there is no durable storage
to query at all. Rather than pretend, the endpoint returns
`{"sessions": [], "persistence": "none"}` — an honest signal the UI uses to
show *"running on in-memory storage"* instead of a misleading empty list.

### `GET /sessions/{thread_id}/history` — detail

Reads the newest snapshot (as above), then reshapes it:

- **`plan`** — `role_summary`, `total_sessions`, `key_gaps`
- **`sessions`** — one entry per graded question, built by joining
  `state["answers"]` and `state["grades"]` on `question_id` (a dict-keyed
  join, not a nested loop — both lists are small and question_id is unique
  per session), with the question's title/prompt looked up via the existing
  `tools.get_question_by_id()`
- **`readiness_verdict`** — as stored, or `None` if the session never
  reached that gate
- **404** if the thread_id has no checkpoint at all (`snapshot is None` or
  `not snapshot.values`)

### Simplification vs. the original plan text

`PLAN.md`'s sample response included a per-session `weak_areas_after` field —
the weak-areas snapshot as it stood right after *that specific* session.
Getting that exactly right would mean walking the *entire* checkpoint
history (not just the newest snapshot) and diffing `weak_areas` between
session boundaries — real complexity for a detail few users would notice.
The shipped version exposes one top-level `weak_areas` (the final,
fully-accumulated list) instead. Documented here and in `PLAN.md` rather than
silently dropped — a reasonable scope cut for a v1 learning project, not an
oversight.

---

## 11b — the history page

`loop/static/sessions.html` — vanilla JS, no build step, same dark Tailwind
theme as `index.html` (Phase 7). Two views toggled by hiding/showing
`<section>` elements, exactly like `index.html`'s gate panels:

- **List view** — one card per session: verdict badge (ready / not ready /
  incomplete), timestamp, "N of M sessions completed", a "View details"
  button. Shows an amber persistence notice when `persistence: "none"`, or a
  friendly empty state when there are zero sessions.
- **Detail view** — fetched on demand via `GET /sessions/{id}/history`:
  a plan card, one card per graded question (modality-free — the response
  doesn't currently carry modality, just title/prompt/answer/grade) with a
  CSS width-based score bar, collapsible question+answer text, and
  strengths/improvements/feedback; a readiness-verdict card, or a graceful
  "not reached yet" message for an in-progress session.

`index.html` gained a `📋 History` link in its header pointing at
`/static/sessions.html` — no routing framework involved; FastAPI's
`StaticFiles` mount already serves any file dropped into `loop/static/`.

## Verification

Manually verified against the real `db/loop_state.sqlite` left over from an
earlier phase's demo run (`uv run python -m loop.graph`) — no synthetic data
needed. The list view correctly showed an in-progress session (1 of 6
sessions completed, "Incomplete" badge); the detail view rendered the real
prep plan, the one graded system-design question with its actual score,
strengths, and improvements, and correctly showed the "not reached a
readiness verdict yet" message since that particular run never reached
Gate 3. No console errors; back-navigation between list and detail views
confirmed working.

---

## What to take away

- **A checkpoint-backed graph gives you audit/replay for free** — the same
  data that powers HITL resume (Phase 5) also powers a full session history
  view. No new persistence layer, no data duplication — just a different
  *read* over storage you already have.
- **`get_state()` vs. `get_state_history()`** — reach for the plain
  `get_state()` when you only need the current/latest snapshot (cheaper,
  simpler); reach for `get_state_history()` when the *concept* of
  chronological replay matters, even if in practice you only consume the
  first item.
- **When an API doesn't exist at the level you want ("list every thread"),
  drop one layer down to what the component is actually built on** (here:
  direct SQL against the checkpointer's own table) — but keep that access
  read-only and scoped to what you already own writing to, exactly the same
  caution you'd apply to querying another service's database table directly.
