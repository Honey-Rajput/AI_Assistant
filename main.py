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

# ✅ Track last uploaded file to reset chat properly
if "last_file" not in st.session_state:
    st.session_state.last_file = None


# ============================================================
# Document Type Detector (Improved)
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
    <p>Upload PDFs and chat with them using Qdrant Cloud</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR UPLOAD
# ============================================================
with st.sidebar:

    st.header("📁 Upload PDF Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:

        st.success("✅ File uploaded successfully!")

        if st.button("🚀 Process Document"):

            # ✅ Reset chat if new file uploaded
            if uploaded_file.name != st.session_state.last_file:
                st.session_state.messages = []
                st.session_state.last_file = uploaded_file.name

            with st.spinner("Processing PDF..."):

                # ============================================================
                # Extract Full Text (for doc type detection)
                # ============================================================
                full_text = extract_text_from_pdf(uploaded_file)

                if full_text.strip() == "":
                    st.error("❌ No readable text found (PDF may be scanned).")
                    st.stop()

                # ============================================================
                # Detect Document Type
                # ============================================================
                doc_type = detect_document_type(full_text)
                st.session_state.doc_type = doc_type

                st.info(f"📌 Detected Document Type: {doc_type.upper()}")

                # ============================================================
                # ✅ Save PDF temporarily (Required for PyPDFLoader)
                # ============================================================
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # ============================================================
                # ✅ Load PDF with Page Metadata
                # ============================================================
                loader = PyPDFLoader("temp.pdf")
                pages = loader.load()

                # ============================================================
                # Chunking With Metadata (TOC + Page Index Fix)
                # ============================================================
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(pages)

                # ============================================================
                # Limit Chunks
                # ============================================================
                MAX_CHUNKS = 3000
                if len(chunks) > MAX_CHUNKS:
                    st.warning(f"⚠️ Large PDF detected. Using first {MAX_CHUNKS} chunks.")
                    chunks = chunks[:MAX_CHUNKS]

                st.write("✅ Total Chunks Created:", len(chunks))

                # ============================================================
                # Upload to Qdrant
                # ============================================================
                qdrant_client = create_qdrant_index(chunks)
                st.session_state.qdrant_client = qdrant_client

                # ============================================================
                # Load Chat Model
                # ============================================================
                chat_model = get_chat_model(api_key=EURI_API_KEY)
                st.session_state.chat_model = chat_model

                st.success("✅ Document Indexed Successfully!")


# ============================================================
# CHAT UI
# ============================================================
st.subheader("💬 Chat with Your Document")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# USER INPUT
# ============================================================
if prompt := st.chat_input("Ask something about your uploaded document..."):

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # ============================================================
    # Assistant Response
    # ============================================================
    if st.session_state.qdrant_client and st.session_state.chat_model:

        with st.chat_message("assistant"):

            with st.spinner("🔍 Searching document..."):

                # Retrieve relevant chunks (with page info)
                relevant_chunks = retrive_similar_documents(
                    st.session_state.qdrant_client,
                    prompt,
                    k=4
                )

                context = "\n\n".join(relevant_chunks)

                doc_type = st.session_state.doc_type

                # ============================================================
                # ✅ ORIGINAL PROMPTS RESTORED
                # ============================================================

                if doc_type == "medical":
                    system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant. 
Based on the following medical documents, provide accurate and helpful answers. 
If the information is not in the documents, clearly state that.

Medical Documents:
{context}

User Question: {prompt}

Answer:"""

                elif doc_type == "financial":
                    system_prompt = f"""You are FinDoc Pro, an intelligent financial document assistant.
Based on the following financial documents, answer clearly.
If the answer is not present, say so honestly.

Financial Documents:
{context}

User Question: {prompt}

Answer:"""

                elif doc_type == "legal":
                    system_prompt = f"""You are LawDoc Pro, an intelligent legal document assistant.
Answer only based on the legal document context.

Legal Documents:
{context}

User Question: {prompt}

Answer:"""

                else:
                    system_prompt = f"""You are DocChat Pro, an intelligent document assistant.
Based on the following uploaded documents, answer accurately.
If the answer is not in the documents, clearly mention it.

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
        st.error("⚠️ Please upload and process a document first!")
