"""Tools available to the dispatch agent."""

from src.mcp_client import get_mcp_tools
from src.tools import fetch_corridor_conditions, search_compliance_sop


def get_agent_tools():
    """Discover telemetry MCP tools and add the weather and SOP tools."""
    return [*get_mcp_tools(), fetch_corridor_conditions, search_compliance_sop]


def main():
    """List the tools the agent can call, including server-discovered tools."""
    for tool in get_agent_tools():
        print(f"{tool.name}: {tool.description}")


if __name__ == "__main__":
    main()
