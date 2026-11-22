import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from src.mcp_client import SERVER


async def main():
    load_dotenv()
    model = ChatOpenAI(
        model=os.environ["AGENT_LLM_NAME"],
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url=os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    messages = [
        SystemMessage(
            content="Use telemetry for readings. Summarize only the returned historical data."
        ),
        HumanMessage(content=input("Ask about fleet temperatures: ")),
    ]

    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Discover the MCP tool and describe it to the model.
            await session.initialize()
            listed = await session.list_tools()
            tools_by_name = {tool.name: tool for tool in listed.tools}

            model_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools_by_name.values()
            ]

            model_with_tools = model.bind_tools(model_tools)
            max_tool_rounds = 3

            for round_number in range(max_tool_rounds + 1):
                reply = await model_with_tools.ainvoke(messages)

                if not reply.tool_calls:
                    print("Answer:", reply.content)
                    return

                if round_number == max_tool_rounds:
                    print("Stopped after three tool-call rounds.")
                    return

                messages.append(reply)

                for call in reply.tool_calls:
                    print("Model requested:", call["name"], call["args"])

                    if call["name"] not in tools_by_name:
                        payload = {"error": "Unknown tool."}
                    else:
                        result = await session.call_tool(call["name"], arguments=call["args"])
                        if result.isError or result.structuredContent is None:
                            payload = {"error": "Telemetry lookup failed."}
                        else:
                            payload = result.structuredContent

                    print("Tool result:", payload)
                    messages.append(
                        ToolMessage(
                            content=json.dumps(payload),
                            tool_call_id=call["id"],
                        )
                    )


if __name__ == "__main__":
    asyncio.run(main())
