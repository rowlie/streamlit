import streamlit as st
import os
from datetime import datetime

# This import must match your corrected rag_chain.py
from rag_chain import initialize_chain, chat_with_rag_and_tools, SESSION_ID_KEY


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="LCEL RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
)

# Initialize Streamlit session state for UI
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Configuration")

    # Check API keys (Streamlit Cloud: uses secrets)
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")

    if all([api_key, pinecone_key, langsmith_key]):
        st.success("✅ All API keys configured")
        st.info("🌐 **LangSmith tracing is active**")
    else:
        st.error("❌ Missing environment variables. Add them in Streamlit Secrets.")
        st.stop()

    st.divider()

    # Clear chat + LCEL memory
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []

        memory_key = f"{SESSION_ID_KEY}_streamlit_user"
        if memory_key in st.session_state:
            del st.session_state[memory_key]

        st.rerun()

    st.divider()

    st.caption(
        """
### Demo Prompts

**RAG Retrieval**
- *What is the most dangerous type of fat?*

**Tool Use**
- *I am an 80kg male, light activity, and I want to lose weight.  
  What should my calories and protein be?*  
  → (Uses **estimate_targets** tool)

**Conversational Memory**
1. Ask calorie/protein question above  
2. Then ask:  
   *What if I switch to active training?*  
   → Memory remembers weight & sex.

**LCEL Architecture**
- Uses `RunnableBranch` for tool execution  
- Uses `RunnableWithMessageHistory` for memory  
        """
    )


# ============================================================================
# PAGE TITLE
# ============================================================================
st.title("🤖 Body Logic — LCEL Agent with RAG, Tools & Memory")
st.markdown(
    "Ask any fitness question. The agent uses RAG, specialized tools, and conversational memory."
)


# ============================================================================
# INITIALIZE THE RAG + TOOL CHAIN
# ============================================================================
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG chain..."):
            initialize_chain()
        st.session_state.chain_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize chain: {str(e)}")
        st.stop()


# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================================
# USER INPUT
# ============================================================================
if prompt := st.chat_input("Ask your question here..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            with st.spinner("Thinking... (sending to LangSmith)…"):
                response = chat_with_rag_and_tools(prompt)

            placeholder.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

        except Exception as e:
            error_text = f"❌ Error: {str(e)}"
            placeholder.error(error_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_text}
            )
