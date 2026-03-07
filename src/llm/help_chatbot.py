"""Help Chatbot — a page-aware, self-hosted RAG assistant.

Orchestrates the full RAG pipeline:
1. Expand query with a fast local LLM.
2. Retrieve context from local pgvector (docs+code) and optionally web search.
3. Re-rank retrieved chunks with a local cross-encoder.
4. Synthesize the final answer using a fast or deep local LLM.
"""
import logging
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.data.db_router import get_router
from src.llm.web_search import search_bing

logger = logging.getLogger(__name__)

FAST_MODEL = "deepseek-r1:14b"
DEEP_MODEL = "deepseek-r1:70b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BASE_URL = "http://localhost:11434"


class HelpChatbot:
    """Self-hosted RAG chatbot for platform assistance."""

    def __init__(self, config: dict = None):
        if config is None:
            try:
                import yaml
                with open("config.yaml") as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                config = {}
        llm_config = config.get("llm", {})
        self.base_url = llm_config.get("base_url", DEFAULT_BASE_URL)
        self.router = get_router(config)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)

    def ask(self, question: str, page_context: str, use_deep_model: bool = False) -> dict:
        """Main entry point for asking a question."""
        try:
            expanded_queries = self._expand_query(question, page_context)
            queries = [question] + expanded_queries

            vector_chunks = self._retrieve_from_vectors(queries)
            web_chunks = self._retrieve_from_web(queries)
            all_chunks = pd.concat([vector_chunks, web_chunks]).drop_duplicates(subset=["chunk_text"])

            if all_chunks.empty:
                return {"answer": "I could not find any relevant information.", "sources": []}

            reranked_chunks = self._rerank_chunks(question, all_chunks)
            top_5_chunks = reranked_chunks.head(5)

            answer = self._synthesize(question, page_context, top_5_chunks, use_deep_model)
            sources = top_5_chunks["source_path"].tolist()

            return {"answer": answer, "sources": sources}
        except Exception as e:
            logger.error("HelpChatbot.ask failed: %s", e, exc_info=True)
            return {"answer": f"An error occurred: {e}", "sources": []}

    def _expand_query(self, question: str, page_context: str) -> list[str]:
        """Use the fast LLM to generate alternative phrasings."""
        prompt = (
            "You are a query expansion assistant. Given a user's question and the "
            "current page they are on, generate 3 alternative ways to phrase the "
            "question to improve search results. Focus on synonyms and related "
            "technical terms. Return only a numbered list.\n\n"
            f'User Question: "{question}"\n'
            f'Page Context: "{page_context}"\n\n'
            "Alternative Questions:\n1. "
        )
        try:
            response = self._call_llm(prompt, model=FAST_MODEL)
            return [
                line.strip().split(". ", 1)[1]
                for line in response.strip().split("\n")
                if ". " in line
            ][:3]
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            return []

    def _retrieve_from_vectors(self, queries: list[str]) -> pd.DataFrame:
        """Retrieve relevant chunks from the local pgvector knowledge base."""
        all_results = []
        for q in queries:
            embedding = self.embedder.encode(q, normalize_embeddings=True).tolist()
            df = self.router.vector_search_knowledge(embedding, limit=10)
            if not df.empty:
                all_results.append(df)
        return pd.concat(all_results) if all_results else pd.DataFrame(
            columns=["source_path", "chunk_text", "similarity"]
        )

    def _retrieve_from_web(self, queries: list[str]) -> pd.DataFrame:
        """Retrieve relevant snippets from the web using Bing Search."""
        all_results = []
        web_results = search_bing(queries[0])
        for r in web_results:
            name = r.get("name", "")
            snippet = r.get("snippet", "")
            all_results.append({
                "source_path": r.get("url", ""),
                "chunk_text": f"{name}: {snippet}",
                "similarity": 0.8,
            })
        if not all_results:
            return pd.DataFrame(columns=["source_path", "chunk_text", "similarity"])
        return pd.DataFrame(all_results)

    def _rerank_chunks(self, question: str, chunks: pd.DataFrame) -> pd.DataFrame:
        """Use a cross-encoder to re-rank retrieved chunks for relevance."""
        pairs = [[question, row["chunk_text"]] for _, row in chunks.iterrows()]
        scores = self.reranker.predict(pairs)
        chunks = chunks.copy()
        chunks["rerank_score"] = scores
        return chunks.sort_values("rerank_score", ascending=False)

    def _synthesize(self, question: str, page_context: str,
                    chunks: pd.DataFrame, use_deep_model: bool) -> str:
        """Use an LLM to generate a final answer from the top-ranked chunks."""
        context_parts = []
        for _, row in chunks.iterrows():
            src = row["source_path"]
            txt = row["chunk_text"]
            context_parts.append(f"Source: {src}\nContent: {txt}")
        context_str = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are a helpful AI assistant for a stock analysis platform. "
            "Answer the user's question based *only* on the provided context. "
            "Be concise and clear. Cite sources by mentioning the file path "
            "(e.g., `src/data/features.py`).\n\n"
            f'Page Context: "{page_context}"\n\n'
            f'User Question: "{question}"\n\n'
            f"Context:\n{context_str}\n\nAnswer:"
        )
        model = DEEP_MODEL if use_deep_model else FAST_MODEL
        return self._call_llm(prompt, model=model)

    def _call_llm(self, prompt: str, model: str) -> str:
        """Helper to make a raw API call to a local Ollama model."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
