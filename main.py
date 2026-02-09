import streamlit as st
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.pdf_utils import extract_text_from_pdf
from app.chat_utilis import get_chat_model, ask_chat_model
from app.vectorstore_utils import create_qdrant_index, retrive_similar_documents


# ============================================================
# Load ENV Keys
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
# SESSION STATE INIT
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None

if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

if "doc_type" not in st.session_state:
    st.session_state.doc_type = "general"

# ✅ Track last uploaded files (MULTI PDF)
if "last_files" not in st.session_state:
    st.session_state.last_files = None


# ============================================================
# Document Type Detector
# ============================================================
def detect_document_type(text):

    text = text.lower()

    keyword_sets = {
        "medical": ["diagnosis", "prescription", "treatment", "hospital"],
        "financial": ["profit", "income", "revenue", "balance sheet"],
        "legal": ["agreement", "court", "contract"]
    }

    scores = {}

    for dtype, keywords in keyword_sets.items():
        scores[dtype] = sum(text.count(word) for word in keywords)

    best_type = max(scores, key=scores.get)

    if scores[best_type] < 2:
        return "general"

    return best_type


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="text-align:center; padding:20px;">
    <h1 style="color:#0A66C2;">💬 AI Document Assistant</h1>
    
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR UPLOAD (MULTIPLE PDFs FIXED)
# ============================================================
with st.sidebar:

    st.header("📁 Upload PDF Documents")

    # ✅ MULTIPLE FILE UPLOAD ENABLED
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

        if st.button("🚀 Process Documents"):

            # ✅ Reset chat if new files uploaded
            current_files = [file.name for file in uploaded_files]

            if current_files != st.session_state.last_files:
                st.session_state.messages = []
                st.session_state.last_files = current_files

            with st.spinner("Processing PDFs..."):

                all_chunks = []
                combined_text = ""

                # ============================================================
                # Loop Through All PDFs
                # ============================================================
                for uploaded_file in uploaded_files:

                    # Extract Full Text
                    full_text = extract_text_from_pdf(uploaded_file)

                    if full_text.strip() == "":
                        st.error(f"❌ No readable text found in {uploaded_file.name}")
                        continue

                    combined_text += full_text + "\n"

                    # Save each PDF temporarily
                    temp_name = f"temp_{uploaded_file.name}"

                    with open(temp_name, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load PDF with metadata
                    loader = PyPDFLoader(temp_name)
                    pages = loader.load()

                    # Chunking
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(pages)
                    all_chunks.extend(chunks)

                # ============================================================
                # Detect Document Type (Combined)
                # ============================================================
                doc_type = detect_document_type(combined_text)
                st.session_state.doc_type = doc_type

                st.info(f"📌 Detected Document Type: {doc_type.upper()}")

                # ============================================================
                # Limit Chunks
                # ============================================================
                MAX_CHUNKS = 3000
                if len(all_chunks) > MAX_CHUNKS:
                    st.warning(f"⚠️ Large PDFs detected. Using first {MAX_CHUNKS} chunks.")
                    all_chunks = all_chunks[:MAX_CHUNKS]

                st.write("✅ Total Chunks Created:", len(all_chunks))

                # ============================================================
                # Upload to Qdrant
                # ============================================================
                qdrant_client = create_qdrant_index(all_chunks)
                st.session_state.qdrant_client = qdrant_client

                # ============================================================
                # Load Chat Model
                # ============================================================
                chat_model = get_chat_model(api_key=EURI_API_KEY)
                st.session_state.chat_model = chat_model

                st.success("✅ All Documents Indexed Successfully!")


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
if prompt := st.chat_input("Ask something about your uploaded documents..."):

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # ============================================================
    # Assistant Response
    # ============================================================
    if st.session_state.qdrant_client and st.session_state.chat_model:

        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching documents..."):

                # Retrieve relevant chunks
                relevant_chunks = retrive_similar_documents(
                    st.session_state.qdrant_client,
                    prompt,
                    k=4
                )

                context = "\n\n".join(relevant_chunks)

                doc_type = st.session_state.doc_type

                # ============================================================
                # PROMPTS
                # ============================================================
                if doc_type == "medical":
                    system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant. 
Answer based only on the documents.

Medical Documents:
{context}

User Question: {prompt}

Answer:"""

                elif doc_type == "financial":
                    system_prompt = f"""You are FinDoc Pro, an intelligent financial assistant.

Financial Documents:
{context}

User Question: {prompt}

Answer:"""

                elif doc_type == "legal":
                    system_prompt = f"""You are LawDoc Pro, an intelligent legal assistant.

Legal Documents:
{context}

User Question: {prompt}

Answer:"""

                else:
                    system_prompt = f"""You are DocChat Pro, an intelligent document assistant.

Documents:
{context}

User Question: {prompt}

Answer:"""

                # Ask AI Model
                response = ask_chat_model(
                    st.session_state.chat_model,
                    system_prompt
                )

            st.markdown(response)

            # Store assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

    else:
        st.error("⚠️ Please upload and process documents first!")
