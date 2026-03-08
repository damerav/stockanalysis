"""Help Chatbot — a page-aware, self-hosted RAG assistant.

Orchestrates a streamlined RAG pipeline:
1. Retrieve context from local pgvector knowledge base.
2. Re-rank retrieved chunks with a local cross-encoder.
3. Synthesize the final answer using a fast or deep local LLM.
"""
import logging
import re
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.data.db_router import get_router
from src.llm.web_search import search_web

logger = logging.getLogger(__name__)

FAST_MODEL = "qwen3:8b"
DEEP_MODEL = "qwen3:8b"
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
        # Warm up the fast model so first user query isn't slow
        self._warmup()

    def _warmup(self):
        """Send a tiny prompt to Ollama to pre-load the fast model into GPU memory."""
        try:
            requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": FAST_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.3, "num_predict": 1},
                },
                timeout=120,
            )
            logger.info("HelpChatbot: warmed up %s", FAST_MODEL)
        except Exception as e:
            logger.warning("HelpChatbot warmup failed: %s", e)

    def ask(self, question: str, page_context: str, use_deep_model: bool = False) -> dict:
        """Main entry point — vector lookup + web search + rerank + synthesize."""
        try:
            # 1. Always retrieve from both sources in parallel
            vector_chunks = self._retrieve_from_vectors(question)
            web_chunks = self._retrieve_from_web(question)

            parts = [df for df in [vector_chunks, web_chunks] if not df.empty]
            if not parts:
                # No retrieval results at all — fall back to pure LLM
                answer = self._call_llm(
                    f"Answer this question concisely: {question}", 
                    model=DEEP_MODEL if use_deep_model else FAST_MODEL
                )
                return {"answer": answer, "sources": []}

            all_chunks = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["chunk_text"])

            # 2. Re-rank and take top 5
            reranked = self._rerank_chunks(question, all_chunks)
            top_chunks = reranked.head(5)

            # 3. Synthesize answer
            answer = self._synthesize(question, page_context, top_chunks, use_deep_model)
            sources = top_chunks["source_path"].tolist()

            return {"answer": answer, "sources": sources}
        except Exception as e:
            logger.error("HelpChatbot.ask failed: %s", e, exc_info=True)
            return {"answer": f"An error occurred: {e}", "sources": []}

    def _retrieve_from_vectors(self, query: str) -> pd.DataFrame:
        """Retrieve relevant chunks from the local pgvector knowledge base."""
        try:
            embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
            df = self.router.vector_search_knowledge(embedding, limit=10)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
        return pd.DataFrame(columns=["source_path", "chunk_text", "similarity"])

    def _retrieve_from_web(self, query: str) -> pd.DataFrame:
        """Retrieve relevant snippets from the web using Bing Search."""
        all_results = []
        web_results = search_web(query)
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
        """Use a cross-encoder to re-rank retrieved chunks for relevance.

        Vector DB (local knowledge base) results get a boost so platform-specific
        content is preferred over generic web results.
        """
        pairs = [[question, row["chunk_text"]] for _, row in chunks.iterrows()]
        scores = self.reranker.predict(pairs)
        chunks = chunks.copy()
        chunks["rerank_score"] = scores
        # Boost local knowledge base results — web URLs start with http
        is_web = chunks["source_path"].str.startswith("http")
        chunks.loc[~is_web, "rerank_score"] += 1.5  # strong boost for local docs
        return chunks.sort_values("rerank_score", ascending=False)

    def _synthesize(self, question: str, page_context: str,
                    chunks: pd.DataFrame, use_deep_model: bool) -> str:
        """Use an LLM to generate a final answer from the top-ranked chunks."""
        context_parts = []
        for _, row in chunks.iterrows():
            src = row["source_path"]
            txt = row["chunk_text"][:500]  # Truncate long chunks
            context_parts.append(f"[{src}]: {txt}")
        context_str = "\n---\n".join(context_parts)

        prompt = (
            "You are a concise AI assistant for the SPY/SPX Predictor & ES Futures "
            "Strategy platform. This platform uses ML models (XGBoost, LightGBM, BiLSTM, "
            "and Transformer in a stacking ensemble with logistic meta-learner) for daily "
            "market predictions, with a real-time ES futures trading engine and Streamlit "
            "dashboard.\n"
            "Answer the question directly. Use the context below when relevant, "
            "but you may also use your general knowledge for generic finance/trading "
            "questions. For platform-specific questions, prefer the context and cite "
            "file paths. Do NOT explain your reasoning process. Do NOT start with "
            "'Let me analyze' or similar preambles. Keep answers under 200 words.\n\n"
            f"Page: {page_context}\n"
            f"Question: {question}\n\n"
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
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 300},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Strip any residual <think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip leaked thinking preambles (qwen3:4b sometimes ignores think=False)
        # Patterns: "Hmm, the user...", "Looking at the context...", "Let me..."
        content = re.sub(
            r"^(?:Hmm,?\s.*?\n\n|Looking at the context.*?\n\n|Let me .*?\n\n|"
            r"I need to .*?\n\n|The user .*?\n\n|Okay,?\s.*?\n\n)+",
            "", content, flags=re.DOTALL
        ).strip()
        return content
