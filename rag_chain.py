# -*- coding: utf-8 -*-
"""RAG Chain with Tools - Streamlit-ready (LCEL + Tools + Memory)"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any

import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

# New memory system from langchain-community
from langchain_community.chat_message_histories import ChatMessageHistory

# ============================================================================
# CONFIG
# ============================================================================
INDEX_NAME = "youtube-qa-index"
TOP_K = 5
MEMORY_WINDOW_SIZE = 20
SESSION_ID_KEY = "langchain_session"

# ============================================================================
# GLOBALS
# ============================================================================
_initialized = False
retriever = None
pc = None
index = None
llm = None
llm_with_tools = None
rag_agent_chain_with_history = None

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
def _setup_env():
    """Set defaults for tracing/endpoint if not present in env."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "memory-and-tools-rag-agent-v3")


# ============================================================================
# RETRIEVER - cached resource for Streamlit
# ============================================================================
@st.cache_resource
def get_retriever():
    """
    Load SentenceTransformer model.
    Kept inside the function to avoid import issues on some platforms.
    """
    import torch  # imported lazily for Streamlit stability

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device="cpu",
    )
    return model


# ============================================================================
# TOOLS (single-argument dict signature required by langchain_core @tool)
# ============================================================================
@tool
def calculator(payload: dict) -> str:
    """
    payload: {"expression": "2+2*3"}
    """
    expression = payload.get("expression", "")
    try:
        # Note: using eval - keep trusted input or replace with a safer evaluator if needed
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time(payload: dict) -> str:
    """payload ignored; returns current datetime string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def word_count(payload: dict) -> str:
    """
    payload: {"text": "some long text"}
    """
    text = payload.get("text", "")
    return f"Word count: {len(text.split())}"


@tool
def convert_case(payload: dict) -> str:
    """
    payload: {"text": "hello", "case_type": "upper"|"lower"|"title"}
    """
    text = payload.get("text", "")
    case_type = (payload.get("case_type") or "").lower()
    if case_type == "upper":
        return text.upper()
    if case_type == "lower":
        return text.lower()
    if case_type == "title":
        return text.title()
    return f"Error: Unknown case_type '{case_type}'. Use 'upper', 'lower', or 'title'."


@tool
def estimate_targets(payload: dict) -> str:
    """
    payload: {"weight_kg": 80, "sex": "male", "activity": "light", "goal": "lose"}
    """
    try:
        weight_kg = float(payload.get("weight_kg", 0))
    except Exception:
        return "Error: weight_kg must be a number."

    sex = (payload.get("sex") or "").lower()
    activity = (payload.get("activity") or "light").lower()
    goal = (payload.get("goal") or "maintain").lower()

    factors = {"sedentary": 28, "light": 31, "moderate": 34, "active": 37}
    factor = factors.get(activity, 31)

    maintenance = weight_kg * factor
    if goal == "lose":
        target = maintenance - 400
        goal_text = "weight loss"
    elif goal == "gain":
        target = maintenance + 400
        goal_text = "muscle gain"
    else:
        target = maintenance
        goal_text = "maintenance"

    protein_low = weight_kg * 1.6
    protein_high = weight_kg * 2.2

    return (
        f"Estimated daily targets for {goal_text}:\n"
        f"- Calories: {int(target)} kcal/day\n"
        f"- Protein: {protein_low:.1f}–{protein_high:.1f} g/day\n"
        "Note: These are simplified estimates—adjust for age, body composition, and training."
    )


# Pack tools into a list for binding
tools = [calculator, get_current_time, word_count, convert_case, estimate_targets]


# ============================================================================
# PINECONE RAG RETRIEVAL HELPERS
# ============================================================================
def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Encode the query and query Pinecone index for top_k matches.
    Returns the raw Pinecone response or {"matches": []} on failure.
    """
    global retriever, index
    if retriever is None or index is None:
        return {"matches": []}

    try:
        vector = retriever.encode(query).tolist()
        res = index.query(vector=vector, top_k=top_k, include_metadata=True, timeout=10)
        return res
    except Exception as e:
        print(f"⚠️ Pinecone retrieval error: {e}")
        return {"matches": []}


def context_string_from_matches(matches: List[dict]) -> str:
    parts: List[str] = []
    for m in matches:
        meta = m.get("metadata", {}) or {}
        passage = meta.get("text") or meta.get("passage_text") or ""
        if passage:
            parts.append(passage)
    return "\n\n".join(parts)


# ============================================================================
# TOOL EXECUTOR (runs tools requested by LLM)
# ============================================================================
def _tool_executor(call: dict) -> ToolMessage:
    """
    call: expected shape depends on LLM tool-calling output.
    We support either {"name": "...", "args": {...}} or the LCEL style tool call object.
    """
    # Try to find the tool name and args in a few common patterns
    tool_name = call.get("name") or (call.get("function") or {}).get("name")
    raw_args = call.get("args") or (call.get("function") or {}).get("arguments") or {}

    # Normalize args: if it's a JSON string, parse it
    if isinstance(raw_args, str):
        try:
            tool_args = json.loads(raw_args)
        except Exception:
            # fall back to empty dict
            tool_args = {}
    elif isinstance(raw_args, dict):
        tool_args = raw_args
    else:
        tool_args = {}

    # Find matching tool by name (tools are decorated; they have .name attribute)
    matching = [t for t in tools if getattr(t, "name", None) == tool_name or getattr(t, "__name__", None) == tool_name]
    if not matching:
        result_text = f"Tool '{tool_name}' not found."
    else:
        try:
            # The tool callable expects a single dict argument
            result_text = matching[0].invoke(tool_args)
        except Exception as e:
            result_text = f"Error in tool '{tool_name}': {e}"

    # Return a ToolMessage so LCEL chain can consume it
    return ToolMessage(content=str(result_text), tool_call_id=call.get("id", "tool_call"))


# ============================================================================
# PROMPT & MESSAGE BUILDING
# ============================================================================
def _build_full_prompt_messages(inputs: dict) -> dict:
    """
    Build the full messages list combining:
      - system prompt
      - conversation history (from RunnableWithMessageHistory)
      - current user message
      - optional RAG context appended as a user message
    """
    user_message = inputs["user_message"]
    history = inputs.get("chat_history", [])  # expecting list of Message objects

    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))

    messages: List[Any] = []

    # System prompt
    messages.append(
        SystemMessage(
            content=(
                "You are a friendly, evidence-based personal trainer and RAG assistant. "
                "Give safe, practical recommendations and explain your reasoning simply.\n\n"
                "Tool rules:\n"
                "- Use the `calculator` tool for arithmetic (e.g. '2 + 2').\n"
                "- Use the `word_count` tool for any word counting requests.\n"
                "- Use the `convert_case` tool for case conversions.\n"
                "- Use the `get_current_time` tool for current date/time requests.\n"
                "- Use ONLY the `estimate_targets` tool for calorie/protein targets.\n\n"
                "When you call a tool, explicitly state that you used it and base the answer on its output."
            )
        )
    )

    # Append history (these are already Message objects compatible with LLM)
    messages.extend(history)

    # User message
    messages.append(HumanMessage(content=user_message))

    # RAG context as an additional human message (if available)
    if context:
        messages.append(HumanMessage(content=f"📚 Relevant context from knowledge base:\n{context}"))

    return {"messages": messages, "rag_context": context, "original_user_message": user_message}


# ============================================================================
# MEMORY: use ChatMessageHistory stored in Streamlit session_state
# ============================================================================
def _get_session_history(session_id: str):
    key = f"{SESSION_ID_KEY}_{session_id}"
    if key not in st.session_state:
        # ChatMessageHistory implements the expected interface for RunnableWithMessageHistory
        st.session_state[key] = ChatMessageHistory()
    return st.session_state[key]


# ============================================================================
# CHAIN INITIALIZATION
# ============================================================================
def initialize_chain():
    """
    Initialize retriever, Pinecone index, LLM, tools, and the LCEL runnable graph.
    Safe to call multiple times; initialization is idempotent.
    """
    global _initialized, retriever, pc, index, llm, llm_with_tools, rag_agent_chain_with_history

    if _initialized:
        return

    _setup_env()

    # Retriever
    retriever = get_retriever()

    # Pinecone client & index
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        print("⚠️ PINECONE_API_KEY not set; Pinecone calls will fail at runtime.")
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)

    # LLM & bind tools
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Build LCEL run graph
    prompt_builder = RunnableLambda(_build_full_prompt_messages)
    messages_extractor = RunnablePassthrough.assign(messages=lambda x: x["messages"])

    def _is_tool_call(response: AIMessage) -> bool:
        # response may be an AIMessage with .tool_calls attribute (list) when the LLM requests tools
        return getattr(response, "tool_calls", None) is not None and len(response.tool_calls) > 0

    tool_execution_loop = RunnableBranch(
        (
            _is_tool_call,
            # If tool call: execute tools -> re-call LLM with tool outputs appended
            RunnableMap({
                "tool_results": RunnableLambda(lambda x: [_tool_executor(c) for c in x.tool_calls]),
                "original_context": RunnablePassthrough(),
            })
            | RunnableMap({
                "messages": lambda x: x["original_context"]["messages"] + [x["original_context"]["llm_response"]] + x["tool_results"]
            })
            | RunnableMap({
                "llm_response": llm,  # second LLM call (final answer)
                "final_response": lambda x: x["llm_response"]
            })
        ),
        # Else: no tool calls, first LLM response is final
        RunnableMap({
            "llm_response": RunnablePassthrough(),
            "final_response": lambda x: x["llm_response"]
        })
    )

    core_rag_chain = (
        prompt_builder
        | RunnableMap({
            "llm_response": messages_extractor | llm_with_tools,
            "messages": lambda x: x["messages"]
        })
        | tool_execution_loop
    )

    # Wrap with history management
    rag_agent_chain_with_history = RunnableWithMessageHistory(
        core_rag_chain,
        _get_session_history,
        input_messages_key="user_message",
        output_messages_key="final_response",
        history_messages_key="chat_history",
        input_keys=["user_message"],
        output_keys=["final_response"],
    )

    _initialized = True
    print("✅ RAG chain initialized")


# ============================================================================
# MAIN CHAT CALL
# ============================================================================
def chat_with_rag_and_tools(user_message: str) -> str:
    """
    Invoke the runnable chain (which uses RunnableWithMessageHistory internally).
    Returns the assistant's content as a string.
    """
    if not _initialized:
        raise RuntimeError("Chain not initialized. Call initialize_chain() first.")

    try:
        result = rag_agent_chain_with_history.invoke(
            {"user_message": user_message},
            config={"configurable": {"session_id": "streamlit_user"}}
        )
        # result["final_response"] is expected to be an AIMessage
        return getattr(result["final_response"], "content", str(result["final_response"]))
    except Exception as e:
        # Log and return error for UI
        print(f"❌ Error during chat invocation: {e}")
        return f"Error: {str(e)}"
