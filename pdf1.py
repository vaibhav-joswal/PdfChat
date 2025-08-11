
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
import pickle
from pathlib import Path
import re
import hashlib
import logging
from dataclasses import dataclass


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    text: str
    page_num: int
    chunk_id: int

class OptimizedPDFRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir="./cache", max_words=150, overlap=30):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_words = max_words
        self.overlap = overlap
        self._model = None
        self._index = None
        self.document_chunks = []

    self.client = openai.OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_API_KEY")  # Get from environment variable
    )


    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_file_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_cache_path(self, file_hash, suffix):
        return self.cache_dir / f"{file_hash}_{suffix}"

    def extract_text_from_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        text = ""
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text()
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            if page_text:
                text += f"\n--- Page {page_num+1} ---\n{page_text}\n"

        doc.close()
        return text

    def smart_chunk_text(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_words = 0
        chunk_id = 0
        current_page = 1

        for sentence in sentences:
            page_match = re.search(r'--- Page (\d+) ---', sentence)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            sentence_words = len(sentence.split())

            if current_words + sentence_words > self.max_words and current_chunk:
                chunks.append(DocumentChunk(current_chunk.strip(), current_page, chunk_id))
                chunk_id += 1
                current_chunk = sentence
                current_words = sentence_words
            else:
                current_chunk += " " + sentence
                current_words += sentence_words

        if current_chunk.strip():
            chunks.append(DocumentChunk(current_chunk.strip(), current_page, chunk_id))
        
        return chunks

    def generate_embeddings_batch(self, chunks, batch_size=32):
        texts = [chunk.text for chunk in chunks]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    def build_optimized_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        return index

    def semantic_search(self, query, top_k=5):
        if not self._index or not self.document_chunks:
            raise ValueError("Index not built. Process a PDF first.")

        query_embedding = self.model.encode([query])
        distances, indices = self._index.search(np.array(query_embedding).astype("float32"), top_k)
        similarities = np.exp(-distances[0])

        results = []
        for idx, score in zip(indices[0], similarities):
            if idx < len(self.document_chunks):
                results.append((self.document_chunks[idx], score))
        return results

   
    def generate_response_stream(self, question):
        """
        Yields partial chunks of answer for streaming display.
        """
        # 1) Retrieve relevant chunks
        results = self.semantic_search(question, top_k=5)
        chunks = [r[0] for r in results] if results else []
        context = "\n\n".join(f"[Page {c.page_num}]: {c.text}" for c in chunks)

        # 2) Prompt
        system_prompt = (
            "You are an expert, friendly assistant who explains answers in a clear, helpful, and engaging way. "
        "Use the document context as your primary source, but you may also explain concepts with extra details "
        "if it helps the user understand better.\n\n"
        "Guidelines:\n"
        "1. Use the document as your foundation, but elaborate when needed for clarity.\n"
        "2. Always connect your answer to the relevant document sections or page numbers.\n"
        "3. Give step-by-step reasoning or examples if the question needs it.\n"
        "4. Keep a professional yet approachable tone — like a knowledgeable tutor.\n"
        "5. If the answer is missing from the document, guide the user on where or how they could find it.\n"
        "6. Summarize the main takeaway at the end in 1-2 sentences.\n"
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # 3) Stream model response
        try:
            stream = self.client.chat.completions.create(
                model="openai/gpt-oss-120b:novita",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
                stream=True
            )

            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        yield delta.content

        except Exception as e:
            yield f"Error generating response: {e}"


    def process_pdf(self, pdf_path, force_reprocess=False):
        file_hash = self._get_file_hash(pdf_path)
        cache_chunks = self._get_cache_path(file_hash, "chunks.pkl")
        cache_index = self._get_cache_path(file_hash, "index.faiss")

        if not force_reprocess and cache_chunks.exists() and cache_index.exists():
            with open(cache_chunks, 'rb') as f:
                self.document_chunks = pickle.load(f)
            self._index = faiss.read_index(str(cache_index))

            # Print summary of cached chunks loaded
            logger.info(f"Loaded {len(self.document_chunks)} chunks from cache.")
            for chunk in self.document_chunks:
                logger.info(f"Chunk ID: {chunk.chunk_id}, Page: {chunk.page_num}")
            return

        text = self.extract_text_from_pdf(pdf_path)
        self.document_chunks = self.smart_chunk_text(text)

        # Print total chunks and page info when newly created
        logger.info(f"Created {len(self.document_chunks)} chunks from PDF.")
        for chunk in self.document_chunks:
            logger.info(f"Chunk ID: {chunk.chunk_id}, Page: {chunk.page_num}")

        embeddings = self.generate_embeddings_batch(self.document_chunks)
        self._index = self.build_optimized_index(embeddings)

        with open(cache_chunks, 'wb') as f:
            pickle.dump(self.document_chunks, f)
        faiss.write_index(self._index, str(cache_index))


    def ask(self, question, top_k=5):
        results = self.semantic_search(question, top_k)
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]
        avg_conf = float(np.mean(scores)) if scores else 0
        answer = self.generate_response(question, chunks)
        return {"answer": answer, "confidence": avg_conf, "sources": chunks}
