import os
import json
from typing import List, Dict, Any, Optional, Literal

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

def _to_pgvector_literal(vec: List[float]) -> str:
    """Convert list of floats to pgvector literal string like '[0.1,0.2,...]'"""
    return "[" + ",".join(map(str, vec)) + "]"

class GoogleRAGRetriever:
    """
    RAG Retriever using Google embeddings (stored in pgvector) + ChatGoogleGenerativeAI.
    - uses RealDictCursor to get metadata as dict
    - robust metadata handling
    - dedupe and exact-text fallback
    """

    def __init__(self,
                 connection_string: str,
                 google_api_key: str,
                 collection_name: str = "faq_embeddings",
                 llm_model: Optional[str] = None):
        self.connection_string = connection_string
        self.collection_name = collection_name

        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        self.embedding_dimension = 768

        default_model = os.getenv("GOOGLE_LLM_MODEL") or llm_model or "models/gemini-2.0-flash"
        # strip accidental suffix if provided
        if default_model.endswith(":generateContent"):
            default_model = default_model.split(":generateContent", 1)[0]

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=default_model,
                temperature=0.05,
                google_api_key=google_api_key
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google LLM '{default_model}': {e}")

        self.prompt_template = PromptTemplate(
            template=(
                "You are a helpful assistant. Use the context below (from FAQ documents) to answer the question.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
            input_variables=["context", "question"]
        )

    def _get_collection_id(self) -> int:
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        cur.execute("SELECT id FROM collections WHERE name = %s", (self.collection_name,))
        row = cur.fetchone()
        conn.close()
        if not row:
            raise RuntimeError(f"Collection {self.collection_name} not found.")
        return row[0]

    def similarity_search(self, query: str, k: int = 4, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Return up to k best matches (deduped)."""
        # get query embedding
        q_emb = self.embeddings.embed_query(query)
        q_vec = _to_pgvector_literal(q_emb)

        collection_id = self._get_collection_id()
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # fetch more rows (k * 3) and filter/dedupe in Python
        cur.execute("""
            SELECT id, document, metadata, (1 - (embedding <=> %s::vector)) AS similarity_score
            FROM document_embeddings
            WHERE collection_id = %s
            ORDER BY (1 - (embedding <=> %s::vector)) DESC
            LIMIT %s
        """, (q_vec, collection_id, q_vec, max(k * 3, 50)))

        rows = cur.fetchall()
        conn.close()

        results: List[Dict[str, Any]] = []
        seen_snippets = set()
        for r in rows:
            sim = float(r["similarity_score"] or 0.0)
            if sim < similarity_threshold:
                continue

            # metadata comes as dict (RealDictCursor + JSONB) but be safe:
            metadata = r["metadata"] if isinstance(r["metadata"], dict) else (json.loads(r["metadata"]) if r["metadata"] else {})

            snippet = (r["document"] or "").strip()[:250]
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)

            results.append({
                "id": r["id"],
                "content": r["document"],
                "metadata": metadata,
                "similarity_score": sim
            })
            if len(results) >= k:
                break

        # fallback: if no results, try exact-text match (ILIKE)
        if not results:
            conn = psycopg2.connect(self.connection_string)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT id, document, metadata
                FROM document_embeddings
                WHERE collection_id = %s AND document ILIKE %s
                LIMIT %s
            """, (collection_id, f"%{query.split()[0]}%", k))
            for r in cur.fetchall():
                metadata = r["metadata"] if isinstance(r["metadata"], dict) else (json.loads(r["metadata"]) if r["metadata"] else {})
                results.append({
                    "id": r["id"],
                    "content": r["document"],
                    "metadata": metadata,
                    "similarity_score": 0.0
                })
            conn.close()

        return results

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "I don't have relevant information in the FAQs to answer that."

        # build context: include filename & page & a reasonably large window
        context_parts = []
        for i, d in enumerate(retrieved_docs, 1):
            filename = d["metadata"].get("filename", "Unknown")
            page = d["metadata"].get("page", "Unknown")
            chunk_id = d["metadata"].get("chunk_id", "Unknown")
            content = d["content"]
            # keep up to 2000 characters per chunk to avoid truncation of important lines
            if len(content) > 2000:
                content = content[:1200] + "\n...\n" + content[-800:]
            context_parts.append(f"[{i}] {filename} (page {page}, chunk {chunk_id})\n{content}")

        context = "\n\n".join(context_parts)
        prompt = self.prompt_template.format(context=context, question=query)

        try:
            response = self.llm.invoke(prompt)
            # depending on SDK, response may have .content or be string-like
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            # surface helpful guidance for model issues
            err = str(e)
            if "not found" in err.lower() or "404" in err:
                raise RuntimeError(f"LLM error: {err}. Verify the model name (use ListModels) and that your key has access.")
            raise

    def query(self, question: str, k: int = 4, similarity_threshold: float = 0.3, include_sources: bool = True) -> Dict[str, Any]:
        docs = self.similarity_search(question, k=k, similarity_threshold=similarity_threshold)
        answer = self.generate_answer(question, docs)
        result = {
            "question": question,
            "answer": answer,
            "num_sources": len(docs),
            "collection": self.collection_name,
            "provider": "google"
        }
        if include_sources and docs:
            sources = []
            for d in docs:
                sources.append({
                    "filename": d["metadata"].get("filename", "Unknown"),
                    "page": d["metadata"].get("page", "Unknown"),
                    "chunk_id": d["metadata"].get("chunk_id", "Unknown"),
                    "similarity_score": round(d["similarity_score"], 3),
                    "content": d["content"][:400].replace("\n", " ")
                })
            result["sources"] = sources
        return result

    def get_collection_stats(self) -> Dict[str, Any]:
        collection_id = self._get_collection_id()
        conn = psycopg2.connect(self.connection_string)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM document_embeddings WHERE collection_id = %s", (collection_id,))
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT metadata->>'filename') FROM document_embeddings WHERE collection_id = %s", (collection_id,))
        files = cur.fetchone()[0]
        conn.close()
        return {
            "collection": self.collection_name,
            "total_documents": total,
            "unique_files": files,
            "embedding_dimension": self.embedding_dimension
        }

# if __name__ == "__main__":
#     # Example usage
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     CONNECTION_STRING = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"
    
#     retriever = GoogleRAGRetriever(
#     connection_string=CONNECTION_STRING,
#     google_api_key=GOOGLE_API_KEY,
#     collection_name="faq_embeddings",
#     llm_model=os.getenv("GOOGLE_LLM_MODEL") or "models/gemini-2.0-flash"
#     )

#     print("Collection stats:", retriever.get_collection_stats())
#     while (q := input("Enter your question (or 'exit' to quit): ")) != "exit":
#         if q.strip() == "":
#             print("Exiting...")
#             break
#         ans = retriever.query(q, k=4, similarity_threshold=0.2, include_sources=True)
#         print(ans["answer"])
