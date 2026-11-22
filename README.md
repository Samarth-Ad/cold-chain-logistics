# Cold Chain Logistics

The dispatch assistant uses LangGraph with an OpenRouter model. Fleet telemetry is
accessed through a local MCP server; weather and SOP retrieval remain agent tools.
The model receives named, bounded telemetry operations and never receives a tool
that accepts SQL text. The MCP server alone owns the parameterized read queries.

## Setup

1. Install Python 3.12, SQL Server ODBC Driver 18, and `uv`.
2. Run `uv sync` from the project root.
3. Copy `.env.example` to `.env` and set `SQL_AGENT_USER`,
   `SQL_AGENT_PASSWORD`, `OPEN_ROUTER_API_KEY`, and `AGENT_LLM_NAME`.
   The database user needs read access to `FDE_VIEWS.VW_ACTIVE_FLEET`.
4. Start the SQL Server and Qdrant services as needed for your deployment, and
   ingest SOP documents into Qdrant before asking compliance questions.

## Run

Run `uv run streamlit run src/ui.py` for the chat interface, or
`uv run python -m src.orchestrator` for the CLI. Both start the MCP server as a
local stdio subprocess, discover its tools, and call them through the MCP client.
Run from the project root so the app can find `.env` and its data files.

`uv run python mcp_demo.py` is a small standalone example of an MCP session.
`uv run python -m src.agent_tools` lists the tools the production agent sees.
To compare the source CSV with the database through MCP for the Los Angeles
example, run `uv run python scripts/compare_nearby_readings.py` after SQL Server
is available. The script prints timestamps, temperatures and distances from both
sources without asking the model to interpret them. Add `--csv-only` to check
the checked-in data without SQL Server.

The telemetry MCP server currently exposes:

- `find_fleet_readings_near`: readings within a radius of supplied coordinates,
  sorted by actual distance. Use this for city or GPS questions.
- `list_fleet_telemetry`: recent readings with location, temperature, cargo and
  route risk, optionally filtered by temperature or risk classification.
- `list_temperature_readings_above`: up to ten readings above a threshold or
  within a temperature range.
- `count_temperature_readings_above`: count readings above a threshold.

Each tool call opens a short-lived local MCP session. The server uses fixed,
parameterized, read-only SQL statements; the agent cannot supply SQL syntax.
The view called `VW_ACTIVE_FLEET` reads a historical table without a live-status
filter. Its rows are evidence of past readings, not proof of active shipments now.
