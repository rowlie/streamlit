import streamlit as st
import os
from datetime import datetime

# Import your chain logic
from rag_chain import initialize_chain, chat_with_rag_and_tools

# Page config
st.set_page_config(
    page_title="BodyLogic - Agentic RAG Coach",
    page_icon="💪",
    layout="wide"
)

# ---------- CUSTOM CSS (no sidebar, background image, centered chat) ----------
st.markdown(
    """
    <style>
    /* Remove default padding and center content a bit */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: auto;
    }

    /* Background image for entire app */
    .stApp {
        background-image: url("https://images.pexels.com/photos/1552249/pexels-photo-1552249.jpeg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Add a dark translucent overlay behind main content for readability */
    .block-container {
        background-color: rgba(0, 0, 0, 0.65);
        border-radius: 12px;
    }

    /* Global text color */
    .stApp, .block-container, .markdown-text-container {
        color: #f9fafb;
    }

    /* Chat messages */
    .stChatMessage {
        font-size: 0.95rem;
    }

    [data-testid="stChatMessage-user"] {
        background-color: rgba(37, 99, 235, 0.9);
        color: #f9fafb;
    }

    [data-testid="stChatMessage-assistant"] {
        background-color: rgba(15, 23, 42, 0.9);
        color: #e5e7eb;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: rgba(15, 23, 42, 0.9);
        color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# ---------- MAIN HEADER ----------
st.title("BodyLogic")
st.markdown(
    "### Achieve your fitness and nutrition goals with an Agentic RAG chatbot with integrated tools and memory."
)

st.markdown("---")

# ---------- INITIALIZE CHAIN ----------
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Warming up your BodyLogic coach..."):
            initialize_chain()
            st.session_state.chain_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize coach: {str(e)}")
        st.stop()

# ---------- CHAT HISTORY ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- CHAT INPUT ----------
if prompt := st.chat_input("Ask BodyLogic anything about your training, nutrition, or habits..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("BodyLogic is thinking..."):
                response = chat_with_rag_and_tools(prompt)
                message_placeholder.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# ---------- FOOTER ----------
st.markdown("---")
st.caption(
    "💡 BodyLogic for educational purposes uses a RAG system using Curated Youtube Content plus tools like calculator, time, word count, case conversion, and calorie/protein target estimation to personalise your coaching."
)
