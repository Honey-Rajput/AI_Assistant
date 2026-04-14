import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.pdf_utils import extract_text_from_pdf
from app.chat_utilis import get_chat_model, ask_chat_model
from app.vectorstore_utils import create_qdrant_index, retrive_similar_documents

# ============================================================
# LOAD ENV
# ============================================================
load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="💬",
    layout="wide"
)

# ============================================================
# SESSION STATE INIT (VERY IMPORTANT)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None

if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

if "processed" not in st.session_state:
    st.session_state.processed = False   # ✅ THIS FIXES YOUR ERROR

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="text-align:center; padding:20px;">
    <h1 style="color:#0A66C2;">💬 AI Document Assistant</h1>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR (UPLOAD + PROCESS)
# ============================================================
with st.sidebar:

    st.header("📁 Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

        if st.button("🚀 Process Documents"):

            with st.spinner("Processing PDFs..."):

                all_chunks = []

                # ============================================================
                # PROCESS EACH PDF
                # ============================================================
                for uploaded_file in uploaded_files:

                    # Extract text
                    full_text = extract_text_from_pdf(uploaded_file)

                    if full_text.strip() == "":
                        st.error(f"No text found in {uploaded_file.name}")
                        continue

                    # Save temp file
                    temp_name = f"temp_{uploaded_file.name}"
                    with open(temp_name, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load PDF
                    loader = PyPDFLoader(temp_name)
                    pages = loader.load()

                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(pages)
                    all_chunks.extend(chunks)

                st.write("✅ Total Chunks Created:", len(all_chunks))

                # ============================================================
                # CREATE QDRANT INDEX
                # ============================================================
                qdrant_client = create_qdrant_index(all_chunks)

                # ✅ STORE IN SESSION (IMPORTANT FIX)
                st.session_state.qdrant_client = qdrant_client

                # ============================================================
                # LOAD CHAT MODEL
                # ============================================================
                chat_model = get_chat_model(api_key=EURI_API_KEY)
                st.session_state.chat_model = chat_model

                # ✅ MARK AS PROCESSED (MAIN FIX)
                st.session_state.processed = True

                st.success("✅ Documents processed successfully!")

# ============================================================
# CHAT UI
# ============================================================
st.subheader("💬 Chat with Your Documents")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# USER INPUT
# ============================================================
if prompt := st.chat_input("Ask something about your documents..."):

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # ============================================================
    # MAIN FIXED CONDITION
    # ============================================================
    if st.session_state.get("processed", False):

        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching..."):

                # Retrieve chunks
                relevant_chunks = retrive_similar_documents(
                    st.session_state.qdrant_client,
                    prompt,
                    k=4
                )

                context = "\n\n".join(relevant_chunks)

                system_prompt = f"""
You are an AI assistant.

Documents:
{context}

User Question: {prompt}

Answer:
"""

                # Get response
                response = ask_chat_model(
                    st.session_state.chat_model,
                    system_prompt
                )

            st.markdown(response)

            # Save response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

    else:
        st.error("⚠️ Please click 'Process Documents' first!")
