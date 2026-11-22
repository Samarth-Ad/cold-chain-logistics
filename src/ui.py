
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# --------------------------------------------------
# 1. PROJECT INITIALIZATION
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.orchestrator import fde_agent


# --------------------------------------------------
# 2. SYSTEM PROMPT
# --------------------------------------------------

PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "system_prompt.txt"

if not PROMPT_PATH.exists():
    raise FileNotFoundError(f"System prompt missing: {PROMPT_PATH}")

SYSTEM_PROMPT = SystemMessage(
    content=PROMPT_PATH.read_text(encoding="utf-8")
)


# --------------------------------------------------
# 3. STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cold Chain Logistics | FDE",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧊 Cold Chain Logistics")
st.caption("Field Data Engineering · Intelligent Dispatch Assistant")


# --------------------------------------------------
# 4. SESSION MANAGEMENT
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

if "prompt_initialized" not in st.session_state:
    st.session_state.prompt_initialized = False

thread_config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    },
    "recursion_limit": 25
}


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("FDE Command Center")

    st.markdown("---")

    st.caption("Active Session")
    st.code(st.session_state.thread_id[:8])

    st.markdown("---")

    if st.button("New Conversation", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.ui_messages = []
        st.session_state.prompt_initialized = False
        st.rerun()

    st.markdown("---")

    st.caption("Connected Services")
    st.write("🟢 LangGraph Agent")
    st.write("🟢 MCP Telemetry")
    st.write("🟢 Qdrant Vector Store")
    st.write("🟢 OpenRouter LLM")


# --------------------------------------------------
# 6. RENDER PREVIOUS CONVERSATION
# --------------------------------------------------

for entry in st.session_state.ui_messages:

    with st.chat_message(entry["role"]):

        if entry["role"] == "assistant":
            for trace in entry.get("traces", []):

                if trace["type"] == "tool_input":
                    with st.expander(
                        f"⚡ Tool: {trace['name']}"
                    ):
                        st.json(trace["args"])

                elif trace["type"] == "tool_output":
                    with st.expander(
                        f"📤 Output: {trace['name']}"
                    ):
                        st.code(trace["content"])

        st.markdown(entry["content"])


# --------------------------------------------------
# 7. CHAT EXECUTION
# --------------------------------------------------

if user_input := st.chat_input(
    "Ask about fleet telemetry, route disruptions or SOP compliance..."
):

    st.session_state.ui_messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        final_response = ""
        current_traces = []

        try:
            with st.status(
                "🧠 Initializing FDE Agent...",
                expanded=True
            ) as status:

                # Load the system prompt only once per thread.
                if not st.session_state.prompt_initialized:
                    messages = [SYSTEM_PROMPT, HumanMessage(content=user_input)]
                    st.session_state.prompt_initialized = True
                else:
                    messages = [HumanMessage(content=user_input)]

                events = fde_agent.stream(
                    {"messages": messages},
                    config=thread_config,
                    stream_mode="updates"
                )

                for event in events:

                    for node_name, node_state in event.items():

                        # ----------------------------------
                        # REASONING NODE
                        # ----------------------------------

                        if node_name == "reasoner":

                            latest_msg = node_state["messages"][-1]

                            if latest_msg.tool_calls:

                                status.update(
                                    label="🧠 Agent planning tool execution..."
                                )

                                for tool_call in latest_msg.tool_calls:

                                    tool_name = tool_call["name"]
                                    tool_args = tool_call["args"]

                                    st.write(f"⚡ Calling `{tool_name}`")

                                    with st.expander(
                                        f"Tool Arguments: {tool_name}"
                                    ):
                                        st.json(tool_args)

                                    current_traces.append({
                                        "type": "tool_input",
                                        "name": tool_name,
                                        "args": tool_args
                                    })

                            if latest_msg.content:

                                final_response = latest_msg.content

                                status.update(
                                    label="📝 Generating response..."
                                )

                        # ----------------------------------
                        # TOOL EXECUTION NODE
                        # ----------------------------------

                        elif node_name == "tools":

                            status.update(
                                label="🔧 Executing agent tools..."
                            )

                            for msg in node_state.get("messages", []):

                                if isinstance(msg, ToolMessage):

                                    with st.expander(
                                        f"Tool Output: {msg.name}"
                                    ):
                                        st.code(
                                            str(msg.content),
                                            language="text"
                                        )

                                    current_traces.append({
                                        "type": "tool_output",
                                        "name": msg.name,
                                        "content": str(msg.content)
                                    })

                status.update(
                    label="Incident Analysis Complete",
                    state="complete",
                    expanded=False
                )

            # ----------------------------------
            # FINAL RESPONSE
            # ----------------------------------

            if final_response:

                st.markdown(final_response)

                st.session_state.ui_messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "traces": current_traces
                })

            else:
                st.warning("The agent completed without a text response.")

        except Exception as exc:

            st.error(
                f"Agent execution failed: {type(exc).__name__}: {exc}"
            )
