# loop
An agentic interview coach built on LangGraph — plans, mock-interviews, grades, and adapts across sessions.

## Configuration

**Session cap:** by default each run is limited to **2 interview questions** (`MAX_SESSIONS=2` in `loop/config.py`). The planner still generates a full curriculum; only the first N sessions are executed before the readiness verdict. To change it, set `MAX_SESSIONS=<n>` in your `.env` file or edit the default in [`loop/config.py`](loop/config.py).
