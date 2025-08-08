import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor, execute_values

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()

def _to_pgvector_literal(vec: List[float]) -> str:
    """Convert list of floats to pgvector literal string like '[0.1,0.2,...]'"""
    return "[" + ",".join(map(str, vec)) + "]"

class GooglePDFEmbedder:
    """
    Improved PDF embedder for Google embeddings -> pgvector.
    - preserves page metadata
    - stores metadata as JSONB using psycopg2.extras.Json
    - batch inserts embeddings using execute_values
    """

    def __init__(self,
                 connection_string: str,
                 google_api_key: str,
                 collection_name: str = "faq_embeddings",
                 documents_dir: str = "./documents",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        # embedding dim for Google models (approx)
        self.embedding_dimension = 768

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def setup_database(self):
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR UNIQUE
                );
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    collection_id INTEGER REFERENCES collections(id),
                    embedding VECTOR({self.embedding_dimension}),
                    document TEXT,
                    metadata JSONB
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_hashes (
                    id SERIAL PRIMARY KEY,
                    collection_id INTEGER,
                    file_path VARCHAR,
                    file_hash VARCHAR,
                    last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER,
                    UNIQUE (collection_id, file_path)
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx
                ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            # ensure collection exists
            collection_id = self._get_or_create_collection(cur, self.collection_name)
            conn.commit()
            print("Database setup complete, collection_id =", collection_id)
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def _get_or_create_collection(self, cursor, name: str) -> int:
        cursor.execute("SELECT id FROM collections WHERE name = %s", (name,))
        row = cursor.fetchone()
        if row:
            return row[0]
        cursor.execute("INSERT INTO collections (name) VALUES (%s) RETURNING id", (name,))
        return cursor.fetchone()[0]

    def _calculate_file_hash(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def _file_needs_processing(self, file_path: Path, collection_id: int) -> bool:
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        cur.execute(
            "SELECT file_hash FROM document_hashes WHERE collection_id = %s AND file_path = %s",
            (collection_id, str(file_path))
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return True
        return row[0] != self._calculate_file_hash(file_path)

    def _update_file_hash(self, file_path: Path, collection_id: int, chunk_count: int):
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        file_hash = self._calculate_file_hash(file_path)
        cur.execute("""
            INSERT INTO document_hashes (collection_id, file_path, file_hash, chunk_count)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (collection_id, file_path)
            DO UPDATE SET file_hash = EXCLUDED.file_hash, last_processed = CURRENT_TIMESTAMP, chunk_count = EXCLUDED.chunk_count
        """, (collection_id, str(file_path), file_hash, chunk_count))
        conn.commit()
        conn.close()

    def _remove_old_embeddings(self, file_path: Path, collection_id: int):
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM document_embeddings
            WHERE collection_id = %s AND metadata->>'source' = %s
        """, (collection_id, str(file_path)))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        if deleted:
            print(f"Removed {deleted} old embeddings for {file_path.name}")

    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load PDF and split per-page so page metadata is preserved."""
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # usually one Document per page
        documents: List[Document] = []
        for page_num, page in enumerate(pages):
            # ensure page metadata
            page.metadata.setdefault("page", page_num)
            page.metadata.setdefault("source", str(pdf_path))
            page.metadata.setdefault("filename", pdf_path.name)
            # split each page separately
            chunks = self.text_splitter.split_documents([page])
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "chunk_id": f"{page_num}-{chunk_idx}",
                    "page": page_num,
                    "provider": "google",
                    "document_type": "faq_pdf"
                })
                documents.append(chunk)
        print(f"Loaded {len(pages)} pages -> {len(documents)} chunks from {pdf_path.name}")
        return documents

    def embed_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        collection_id = self._get_or_create_collection(cur, self.collection_name)

        texts = [d.page_content for d in documents]
        print(f"Requesting embeddings for {len(texts)} chunks...")
        embeddings = self.embeddings.embed_documents(texts)  # list of lists

        # prepare values for batch insert: (collection_id, vector_literal, document_text, metadata_json)
        values = []
        for doc, emb in zip(documents, embeddings):
            vec_literal = _to_pgvector_literal(emb)
            values.append((
                collection_id,
                vec_literal,
                doc.page_content,
                psycopg2.extras.Json(doc.metadata)
            ))

        sql = "INSERT INTO document_embeddings (collection_id, embedding, document, metadata) VALUES %s"
        try:
            execute_values(cur, sql, values, template="(%s, %s::vector, %s, %s)")
            conn.commit()
            print(f"Inserted {len(values)} embeddings into collection {collection_id}")
            return len(values)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def process_pdf(self, pdf_path: Path) -> int:
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        collection_id = self._get_or_create_collection(cur, self.collection_name)
        conn.close()

        if not self._file_needs_processing(pdf_path, collection_id):
            print(f"No changes for {pdf_path.name}")
            return 0

        self._remove_old_embeddings(pdf_path, collection_id)
        docs = self.load_pdf(pdf_path)
        count = self.embed_documents(docs)
        self._update_file_hash(pdf_path, collection_id, count)
        return count

    def process_all_pdfs(self) -> Dict[str, int]:
        results = {}
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        collection_id = None
        # ensure collection id created
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        collection_id = self._get_or_create_collection(cur, self.collection_name)
        conn.close()

        for p in pdf_files:
            try:
                count = self.process_pdf(p)
                results[p.name] = count
            except Exception as e:
                results[p.name] = -1
                print(f"Error processing {p.name}: {e}")
        return results

if __name__ == "__main__":
    # Example usage
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CONNECTION_STRING = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"
    embedder = GooglePDFEmbedder(
    connection_string=CONNECTION_STRING,
    google_api_key=GOOGLE_API_KEY,
    collection_name="faq_embeddings",
    documents_dir="./documents"
    )
    embedder.setup_database()
    print("Processing PDFs...")
    print(embedder.process_all_pdfs())