# app.py
import os
import json
import streamlit as st
from openai import OpenAI
from pdf1 import OptimizedPDFRAG  # your backend class
from pathlib import Path
from datetime import datetime

# ---------------- CONFIG ----------------
# Put your HF_ROUTER API key here or set env var HF_API_KEY
HF_API_KEY = os.getenv("HF_API_KEY")

PDF_CACHE_DIR = "./pdf_cache"
Path(PDF_CACHE_DIR).mkdir(exist_ok=True)

# OpenAI client (HuggingFace router)
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_API_KEY)

# ---------------- PAGE ----------------
st.set_page_config(page_title="📄 Chat with PDF", page_icon="🤖", layout="wide")
st.markdown(
    "<h1 style='color:#1E88E5;'>🤖 Chat with Your PDF</h1>",
    unsafe_allow_html=True,
)
st.markdown("Upload a PDF in the sidebar, process it and ask questions. The bot will answer using the document context.")

# ---------------- STYLES (full unified CSS) ----------------
st.markdown(
    """
<style>
/* Sidebar (light & readable) */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-right: 2px solid #e6e6e6;
    padding-top: 12px;
}

/* Sidebar header styling */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1E88E5 !important;
    font-weight: 700;
}

/* File uploader */
div[data-testid="stFileUploaderDropzone"] {
    background-color: #fafafa !important;
    border: 2px dashed #1E88E5 !important;
    border-radius: 10px;
    padding: 18px !important;
    color: #000 !important;
}
div[data-testid="stFileUploaderDropzone"] * {
    color: #000 !important;
}
div[data-testid="stFileUploaderFileName"], div[data-testid="stFileUploaderFileName"] * {
    color: #000 !important;
    font-weight: 600;
}

/* Chat container (scrollable) */
.chat-container {
    max-height: 70vh;
    overflow-y: auto;
    padding-right: 12px;
    margin-bottom: 16px;
}

/* User bubble (right) */
.user-msg {
    background-color: #DCF8C6 !important;
    color: #000000 !important;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    display: inline-block;
    max-width: 80%;
    float: right;
    clear: both;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

/* Bot bubble (left) */
.bot-msg {
    background-color: #C5CAE9 !important;
    color: #000000 !important;
    padding: 10px 14px;
    border-radius: 14px;
    margin: 8px 0;
    display: inline-block;
    max-width: 80%;
    float: left;
    clear: both;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

/* clear floats */
.chat-container::after {
    content: "";
    display: table;
    clear: both;
}

/* Process button style */
div.stButton > button {
    background-color: #1E88E5 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 700 !important;
}
div.stButton > button:hover {
    background-color: #1565C0 !important;
}

/* small screens */
@media (max-width: 600px) {
    .user-msg, .bot-msg { max-width: 95% !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- SESSION STATE ----------------
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {time, question, answer}
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 📂 Upload & Process PDF")
    uploaded_file = st.file_uploader("📄 Choose a PDF file", type=["pdf"])

    if uploaded_file:
        pdf_path = os.path.join(PDF_CACHE_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.session_state.current_pdf = pdf_path

    if st.session_state.current_pdf:
        st.write("**Current PDF:**")
        st.markdown(f"- {Path(st.session_state.current_pdf).name}")

    # Controls
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⚡ Process PDF") and st.session_state.current_pdf:
            try:
                with st.spinner("Processing PDF (this can take a moment)..."):
                    rag_system = OptimizedPDFRAG(
                        model_name="all-MiniLM-L6-v2",
                        cache_dir=PDF_CACHE_DIR,
                        max_words=150,
                        overlap=30,
                    )
                    rag_system.process_pdf(st.session_state.current_pdf, force_reprocess=False)
                    st.session_state.rag = rag_system
                    st.session_state.chat_history = []
                    st.success("📚 PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Download chat history
    if st.session_state.chat_history:
        if st.button("⬇️ Download Chat (JSON)"):
            fname = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fname, "w", encoding="utf-8") as fh:
                json.dump(st.session_state.chat_history, fh, ensure_ascii=False, indent=2)
            with open(fname, "rb") as fh:
                st.download_button("Download JSON", data=fh, file_name=fname)
            os.remove(fname)

# ---------------- MAIN: chat area ----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
# render chat history from session_state (top->oldest; we want oldest first)
for item in st.session_state.chat_history:
    q = item.get("question", "")
    a = item.get("answer", "")
    # user (right)
    if q:
        st.markdown(f'<div class="user-msg">👤 <b>You:</b> {q}</div>', unsafe_allow_html=True)
    if a:
        st.markdown(f'<div class="bot-msg">🤖 <b>Bot:</b> {a}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# If no pdf processed, show info
if not st.session_state.rag:
    st.info("Please upload and process a PDF from the sidebar to start chatting.")
else:
    # Chat input (shows at bottom)
    question = st.chat_input("💬 Ask a question about the PDF...")
    if question:
        # 1) Append user message instantly (so it shows without delay)
        # store as dict so we can later update same slot with answer
        entry = {"time": datetime.now().isoformat(), "question": question, "answer": ""}
        st.session_state.chat_history.append(entry)

        # Force immediate display of user message by rerunning UI before heavy work:
        # (We don't use st.experimental_rerun because we want to continue processing in this run.)
        # The appended message will be visible on next script run; to reflect it now we re-render minimal:
        st.rerun()

        # The rest below won't execute because of rerun. The next run will find the last entry with empty answer,
        # then continue to fill it and stream response. So we handle streaming when the last entry has empty answer.

    # If last history item has empty answer and rag exists -> generate response (streaming)
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        if last.get("answer", "") == "" and last.get("question", "") and st.session_state.rag:
            question = last["question"]

            # Create placeholder for streaming
            placeholder = st.empty()
            full_response = ""

            # Stream chunks from backend
            for chunk in st.session_state.rag.generate_response_stream(question):
                full_response += chunk
                placeholder.markdown(
                    f'<div class="bot-msg">🤖 <b>Bot:</b> {full_response}▌</div>',
                    unsafe_allow_html=True
                )

            # Final render without cursor
            placeholder.markdown(
                f'<div class="bot-msg">🤖 <b>Bot:</b> {full_response}</div>',
                unsafe_allow_html=True
            )

            # Save final response in chat history
            st.session_state.chat_history[-1]["answer"] = full_response
            st.rerun()

           
