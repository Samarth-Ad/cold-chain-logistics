import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import (BaseMessage, HumanMessage, SystemMessage)

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from src.agent_tools import get_agent_tools

# setup the project root and load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment before importing modules that depend on it.
load_dotenv(PROJECT_ROOT / ".env")


# Agent State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# TOOL REGISTRY
FDE_TOOLS = get_agent_tools()

if not FDE_TOOLS:
    raise ValueError("No tools found. Please check the tool registry.")

print(f"Registered {len(FDE_TOOLS)} agent tools:")
for agent_tool in FDE_TOOLS:
    print(f"  - {agent_tool.name}")


# LLM INITIALIZATION
AGENT_LLM_SETTING = os.getenv("AGENT_LLM_NAME", "OLLAMA").strip().upper()

def initialize_llm()->ChatOpenAI:

    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    base_url = os.getenv("OPEN_ROUTER_BASE_URL","https://openrouter.ai/api/v1",)
    model_name = os.getenv("AGENT_LLM_NAME")

    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY is missing from .env")

    if not model_name:
        raise ValueError("AGENT_LLM_NAME is missing from .env")

    print("\nInitializing OpenRouter LLM...")
    print(f"Model: {model_name}")
    print(f"Endpoint: {base_url}")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_retries=2,
    )

llm = initialize_llm()

llm_with_tools = llm.bind_tools(FDE_TOOLS)

# Langgraph Nodes
def reasoning_node(state: AgentState):
    """Let the LLM decide whether to call tools or answer."""

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }

# Graph Initialization
def build_agent():

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("reasoner", reasoning_node)
    graph_builder.add_node("tools", ToolNode(FDE_TOOLS))

    graph_builder.add_edge(START, "reasoner")

    graph_builder.add_conditional_edges(
        "reasoner",
        tools_condition,
    )

    graph_builder.add_edge("tools", "reasoner")

    return graph_builder.compile(
        checkpointer=MemorySaver()
    )

print("\nCompiling FDE LangGraph orchestrator...")

fde_agent = build_agent()

# CLI ENTRYPOINT
def main():

    print("\n" + "=" * 60)
    print("FDE SUPPLY CHAIN ORCHESTRATOR ONLINE")
    print("=" * 60)

    print(f"LLM Provider: OpenRouter")
    print(f"LLM Model: {os.getenv('AGENT_LLM_NAME')}")
    print(f"Embedding Provider: {os.getenv('EMBEDDING_MODEL_PROVIDER')}")
    print(f"Registered Tools: {len(FDE_TOOLS)}")

    print("=" * 60)

    prompt_path = (
        PROJECT_ROOT
        / "src"
        / "prompts"
        / "system_prompt.txt"
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt not found: {prompt_path}"
        )

    system_prompt = SystemMessage(
        content=prompt_path.read_text(encoding="utf-8")
    )

    thread_config = {
        "configurable": {
            "thread_id": "fde-cli-session"
        },
        "recursion_limit": 25,
    }

    first_turn = True

    while True:

        user_input = input("\nDispatcher > ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Shutting down FDE Agent...")
            break

        if not user_input:
            continue

        messages = []

        if first_turn:
            messages.append(system_prompt)
            first_turn = False

        messages.append(
            HumanMessage(content=user_input)
        )

        try:
            events = fde_agent.stream(
                {"messages": messages},
                config=thread_config,
                stream_mode="updates",
            )

            for event in events:

                for node_name, node_state in event.items():

                    if node_name == "tools":
                        print("\n[System] Executing tools...")

                        for message in node_state["messages"]:
                            print(f"  Tool: {message.name}")

                    elif node_name == "reasoner":

                        latest_message = node_state["messages"][-1]

                        if latest_message.content:
                            print(
                                f"\nFDE Agent:\n{latest_message.content}"
                            )

        except Exception as exc:
            print(
                f"\nExecution Error: "
                f"{type(exc).__name__}: {exc}"
            )
            
if __name__ == "__main__":
    main()
            
            
            
            