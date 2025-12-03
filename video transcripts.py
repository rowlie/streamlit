# %%
from getpass import getpass
import os

pinecone_key = getpass("Enter your Pinecone API key: ")
os.environ["PINECONE_API_KEY"] = pinecone_key

# %%
# --- Install necessary libraries --- this works run the first cell then this
# Uncomment and run this in Colab first
# !pip uninstall -y pinecone pinecone-client
# !pip install youtube-transcript-api langchain pinecone-client \
#               "huggingface-hub>=0.17.0" "sentence-transformers>=2.2.0" \
#               python-dotenv langchain-text-splitters langchain-huggingface tqdm

import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm.auto import tqdm

# --- 1. Configuration ---
VIDEO_URL = "https://www.youtube.com/watch?v=bCI5cZg-PNY"
VIDEO_ID = VIDEO_URL.split("v=")[-1]
OUTPUT_FILE = "transcript.txt"

PINECONE_INDEX_NAME = "youtube-qa-index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # must match the model

# --- 2. Transcript Extraction Function ---
def extract_transcript(video_id):
    print(f"-> Attempting to fetch transcript for Video ID: {video_id}...")
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Pick the first English transcript
        transcript_content = None
        for t in transcript_list:
            if t.language_code.startswith("en"):
                transcript_content = t.fetch()  # Returns list of FetchedTranscriptSnippet
                break

        if not transcript_content:
            raise Exception("No English transcript found.")

        # Concatenate transcript text
        full_text = " ".join([snippet.text for snippet in transcript_content])
        print("-> Transcript successfully fetched!")
        return full_text

    except Exception as e:
        print(f"ERROR: Could not fetch transcript. Details: {e}")
        return None

# --- 3. Text Chunking ---
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    print(f"-> Splitting transcript into chunks (Size: {chunk_size}, Overlap: {chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_text(text)
    print(f"-> Created {len(chunks)} chunks.")
    return chunks

# --- 4. Pinecone Indexing ---
def index_chunks_to_pinecone(text_chunks, index_name, model_name, api_key):
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set.")
        return False

    try:
        print("-> Initializing Pinecone...")
        pc = Pinecone(api_key=api_key)
        existing_indexes = pc.list_indexes()
        index_names = [idx["name"] for idx in existing_indexes]

        if index_name not in index_names:
            print(f"-> Creating index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("-> Waiting for index to be ready...")
            time.sleep(5)

        index = pc.Index(index_name)
        print(f"-> Loading embeddings model '{model_name}'...")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        batch_size = 32
        for i in tqdm(range(0, len(text_chunks), batch_size), desc="Indexing Batches"):
            batch_texts = text_chunks[i:i+batch_size]
            batch_vectors = embeddings.embed_documents(batch_texts)

            # --- THE ONLY CHANGE IS HERE ---
            # We now prepend the unique VIDEO_ID to the chunk ID to ensure appending works.
            vectors_to_upsert = [
                {"id": f"{VIDEO_ID}-chunk-{i+j}", "values": batch_vectors[j], "metadata": {"text": batch_texts[j], "source": VIDEO_URL}}
                for j in range(len(batch_texts))
            ]

            index.upsert(vectors=vectors_to_upsert)

        print(f"-> Successfully indexed {len(text_chunks)} chunks.")
        return True

    except Exception as e:
        print(f"Indexing failed: {e}")
        return False

# --- 5. Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # In Colab, set PINECONE_API_KEY using:
    # from google.colab import auth
    # !export PINECONE_API_KEY="YOUR_KEY_HERE"
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    if not pinecone_key:
        print("FATAL: PINECONE_API_KEY not set in environment.")
    else:
        transcript = extract_transcript(VIDEO_ID)
        if transcript:
            chunks = chunk_text(transcript)
            success = index_chunks_to_pinecone(chunks, PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME, pinecone_key)

            if success:
                # Save chunks locally
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    f.write("\n\n" + "="*80 + "\n\n".join([f"[CHUNK {i+1}]: {c}" for i,c in enumerate(chunks)]))
                print(f"Chunks saved to '{OUTPUT_FILE}'.")

            print(f"\nPipeline completed in {time.time() - start_time:.2f} seconds.")
        else:
            print("Failed to fetch transcript; cannot continue.")


