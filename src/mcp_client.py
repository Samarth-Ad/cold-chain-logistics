"""Discover and invoke telemetry tools through the local MCP server."""

import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "src.mcp_server"],
    cwd=PROJECT_ROOT,
    env={
        **get_default_environment(),
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(PROJECT_ROOT), os.getenv("PYTHONPATH")) if part
        ),
    },
)


async def _list_tools():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return (await session.list_tools()).tools


async def _call_tool(name: str, arguments: dict) -> str:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)

    if result.isError:
        return "Telemetry lookup failed. Check the MCP server and database connection."
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, default=str)
    return "\n".join(block.text for block in result.content if block.type == "text")


def get_mcp_tools() -> list[StructuredTool]:
    """Expose server-discovered MCP tools to the synchronous LangGraph agent."""
    discovered = asyncio.run(_list_tools())
    tools = []

    for definition in discovered:
        def make_invoker(name):
            def invoke_mcp(**kwargs):
                return asyncio.run(_call_tool(name, kwargs))

            return invoke_mcp

        tools.append(
            StructuredTool.from_function(
                func=make_invoker(definition.name),
                name=definition.name,
                description=definition.description or definition.name,
                args_schema=definition.inputSchema,
            )
        )

    return tools
