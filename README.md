# 📄 PdfChat – Chat with Your PDF

PdfChat is an interactive AI-powered application that allows you to **upload a PDF document** and **ask questions** about it in natural language.  
The app uses **semantic search** and **Retrieval-Augmented Generation (RAG)** to provide **accurate, context-based answers** from your document.

---

## 🚀 Features
- **PDF Upload & Parsing** – Extracts text from your PDF.
- **Semantic Search with FAISS** – Finds the most relevant document chunks.
- **Context-Aware Responses** – Uses Hugging Face / OpenAI API for answer generation.
- **Responsive Streamlit UI** – Clean interface with custom CSS.
- **Real-Time Streaming Responses** – Bot types out answers as it processes.
- **Chat History Download** – Save your conversation in JSON format.

---

## 🛠️ Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/) + Custom CSS
- **Backend:** Python
- **AI Model:** Hugging Face / OpenAI (via HuggingFace Router)
- **Vector Search:** FAISS
- **PDF Processing:** PyMuPDF (`fitz`)
- **Embeddings:** SentenceTransformers

---

## 📂 Project Structure
PdfChat/
│── app.py # Streamlit frontend
│── pdf1.py # Backend logic (PDF parsing, FAISS, RAG)
│── requirements.txt # Python dependencies
│── pdf_cache/ # Cached PDF data
│── README.md # Project documentation

yaml
Copy
Edit

---

## ⚡ Installation & Usage

1️⃣ **Clone the repository**
```bash
git clone https://github.com/vaibhav-joswal/PdfChat.git
cd PdfChat
2️⃣ Create a virtual environment

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the app

bash
Copy
Edit
streamlit run app.py
📷 Screenshots

🏆 Key Skills / Expertise
Streamlit UI Development

PDF Text Extraction (PyMuPDF)

Semantic Search (FAISS)

Retrieval-Augmented Generation (RAG)

Hugging Face / OpenAI API Integration

📜 License
This project is licensed under the MIT License – feel free to use and modify.