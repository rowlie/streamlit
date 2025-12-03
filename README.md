# BodyLogic — RAG Agent with Tools and Memory

BodyLogic is a Streamlit app that delivers evidence-based fitness guidance using a Retrieval-Augmented Generation (RAG) pipeline, tool-calling, and lightweight conversational memory. It indexes curated YouTube transcripts into Pinecone with all-mpnet-base-v2 embeddings, then answers user questions with OpenAI's GPT via LangChain Runnables. 

## Features

- RAG on curated YouTube content using Pinecone (768-dim embeddings via all-mpnet-base-v2). 
- Tool-aware agent: calorie/protein estimator with strict usage rules. 
- Conversational memory: sliding window of the last 20 messages to preserve context. 
- Streamlit UI with demo prompts and LangSmith tracing enabled for observability. 

## Architecture

- **Ingestion**: YouTubeTranscriptApi → chunking (500 characters, 50 overlap) → HuggingFace embeddings → Pinecone upsert. 
- **Serving chain** (LangChain RunnableSequence):
  - Retriever step: encodes user query and fetches top-k (5) matches from Pinecone. 
  - Prompt builder: system message + memory + optional knowledge base context message. 
  - First LLM call with bound tools decides whether to call tools. 
  - Tool execution and final LLM response. 
- **Memory**: global Python list capped at 20 messages. 

## Repo Structure

- `rag_chain.py` — Core chain: initialization, retriever, tools, runnable steps, and chat function. 
- `app.py` — Streamlit UI, sidebar checks, demo prompts, and chat loop. 
- `video transcripts.py` — Script/notebook to fetch transcript, chunk, embed, and index into Pinecone. 

## Requirements

- Python 3.10+
- Recommended packages: `streamlit`, `langchain`, `langchain-openai`, `langchain-core`, `langchain-text-splitters`, `sentence-transformers`, `pinecone-client`, `youtube-transcript-api`, `huggingface-hub`, `langchain-huggingface`, `tqdm`, `python-dotenv`. 

## Environment Variables

Set via Streamlit secrets

