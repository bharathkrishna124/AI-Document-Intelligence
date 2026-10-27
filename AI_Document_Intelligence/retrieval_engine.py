from pathlib import Path
import hashlib
import json
import pickle
import re
import time

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\Desktop\AI_Document_Intelligence")
PROCESSED_DIR = PROJECT_DIR / "processed_data"

CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
INDEX_FILE = PROCESSED_DIR / "faiss.index"
METADATA_FILE = PROCESSED_DIR / "retrieval_metadata.json"
BM25_FILE = PROCESSED_DIR / "bm25_index.pkl"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_CANDIDATES = 30
FINAL_RESULTS = 5
RRF_K = 60

ENABLE_RERANKING = True


# ============================================================
# TEXT UTILITIES
# ============================================================

def tokenize(text):
    """Tokenize text for BM25 keyword retrieval."""

    if not isinstance(text, str):
        return []

    return re.findall(
        r"[a-zA-Z0-9]+(?:[._+-][a-zA-Z0-9]+)*",
        text.lower()
    )


def normalize_chunk(chunk, index):
    """
    Normalize chunk metadata so that both uploaded documents
    and previously processed chunks use the same structure.
    """

    text = str(chunk.get("text", "")).strip()

    if not text:
        return None

    source = (
        chunk.get("file_name")
        or chunk.get("source")
        or "Unknown document"
    )

    page = chunk.get(
        "page_number",
        chunk.get("page", 1)
    )

    document_id = chunk.get("document_id")

    if not document_id:
        document_id = hashlib.sha256(
            str(source).encode("utf-8")
        ).hexdigest()[:16]

    chunk_id = chunk.get("chunk_id")

    if not chunk_id:
        chunk_id = hashlib.sha256(
            f"{document_id}:{page}:{index}:{text}".encode("utf-8")
        ).hexdigest()[:20]

    return {
        **chunk,
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "file_name": str(source),
        "page_number": int(page),
        "text": text
    }


# ============================================================
# RETRIEVAL ENGINE
# ============================================================

class RetrievalEngine:

    def __init__(
        self,
        enable_reranking=ENABLE_RERANKING
    ):
        self.model = None
        self.reranker = None

        self.index = None
        self.chunks = []
        self.bm25 = None

        self.embeddings = None
        self.chunk_fingerprint = None

        self.enable_reranking = enable_reranking

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    def load_models(self):
        """Load embedding and reranking models only when needed."""

        if self.model is None:
            print("Loading embedding model...")

            self.model = SentenceTransformer(
                EMBEDDING_MODEL_NAME
            )

            print("Embedding model loaded.")

        if self.enable_reranking and self.reranker is None:
            print("Loading cross-encoder reranker...")

            self.reranker = CrossEncoder(
                RERANKER_MODEL_NAME
            )

            print("Reranker loaded.")

    # --------------------------------------------------------
    # LOAD PREPROCESSED CHUNKS
    # --------------------------------------------------------

    def load_chunks(self, chunks_file=CHUNKS_FILE):
        """Load previously generated chunks.json."""

        chunks_file = Path(chunks_file)

        if not chunks_file.is_file():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_file}"
            )

        with open(
            chunks_file,
            "r",
            encoding="utf-8"
        ) as file:
            raw_chunks = json.load(file)

        self.set_chunks(raw_chunks)

    # --------------------------------------------------------
    # SET CHUNKS DIRECTLY
    # --------------------------------------------------------

    def set_chunks(self, chunks):
        """
        Initialize the engine using chunks supplied by Streamlit
        or another Python component.
        """

        if not isinstance(chunks, list):
            raise ValueError("Chunks must be supplied as a list.")

        normalized = []

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue

            item = normalize_chunk(chunk, i)

            if item:
                normalized.append(item)

        if not normalized:
            raise ValueError("No valid text chunks were supplied.")

        self.chunks = normalized
        self.chunk_fingerprint = self.calculate_fingerprint()

    # --------------------------------------------------------
    # FINGERPRINT
    # --------------------------------------------------------

    def calculate_fingerprint(self):
        """Calculate a fingerprint of the current chunks."""

        digest = hashlib.sha256()

        for chunk in self.chunks:
            digest.update(
                chunk["chunk_id"].encode("utf-8")
            )

            digest.update(
                chunk["text"].encode("utf-8")
            )

        return digest.hexdigest()

    # --------------------------------------------------------
    # BUILD INDEXES
    # --------------------------------------------------------

    def build_indexes(self):
        """Build the FAISS and BM25 indexes."""

        if not self.chunks:
            raise RuntimeError(
                "Load or set document chunks before building indexes."
            )

        self.load_models()

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        print(f"Embedding {len(texts)} chunks...")

        start_time = time.perf_counter()

        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.embeddings = np.asarray(
            self.embeddings,
            dtype=np.float32
        )

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(self.embeddings)

        tokenized_documents = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        elapsed = time.perf_counter() - start_time

        print(f"FAISS vectors: {self.index.ntotal}")
        print(f"Index creation time: {elapsed:.2f} seconds")

    # --------------------------------------------------------
    # SAVE INDEXES
    # --------------------------------------------------------

    def save_indexes(self):
        """
        Persist the current indexes.

        Only load BM25 pickle files generated by a trusted source.
        """

        if self.index is None or self.bm25 is None:
            raise RuntimeError("Build the indexes before saving.")

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(INDEX_FILE)
        )

        metadata = {
            "chunk_fingerprint": self.chunk_fingerprint,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "chunk_count": len(self.chunks),
            "chunks": self.chunks
        }

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

        with open(
            BM25_FILE,
            "wb"
        ) as file:
            pickle.dump(self.bm25, file)

        print("FAISS and BM25 indexes saved.")

    # --------------------------------------------------------
    # LOAD SAVED INDEXES
    # --------------------------------------------------------

    def load_saved_indexes(self):
        """Load indexes when they match the current chunks."""

        required_files = [
            INDEX_FILE,
            METADATA_FILE,
            BM25_FILE
        ]

        if not all(path.is_file() for path in required_files):
            return False

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        if metadata.get("embedding_model") != EMBEDDING_MODEL_NAME:
            return False

        if metadata.get("chunk_fingerprint") != self.chunk_fingerprint:
            return False

        if metadata.get("chunk_count") != len(self.chunks):
            return False

        if not isinstance(metadata.get("chunks"), list):
            return False

        self.load_models()

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        if self.index.ntotal != len(self.chunks):
            self.index = None
            return False

        # Do not load a pickle from an untrusted source.
        with open(
            BM25_FILE,
            "rb"
        ) as file:
            self.bm25 = pickle.load(file)

        print("Loaded saved FAISS and BM25 indexes.")

        return True

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    def semantic_search(
        self,
        query,
        top_k=INITIAL_CANDIDATES,
        document_id=None
    ):
        """Search for semantically similar chunks."""

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        search_k = min(
            self.index.ntotal,
            max(top_k * 3, top_k)
            if document_id is not None
            else top_k
        )

        scores, indices = self.index.search(
            query_embedding,
            search_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]

            if (
                document_id is not None
                and chunk["document_id"] != document_id
            ):
                continue

            results.append({
                "chunk": chunk,
                "semantic_score": float(score)
            })

            if len(results) >= top_k:
                break

        return results

    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    def keyword_search(
        self,
        query,
        top_k=INITIAL_CANDIDATES,
        document_id=None
    ):
        """Search for chunks matching important keywords."""

        tokens = tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[int(index)]

            if (
                document_id is not None
                and chunk["document_id"] != document_id
            ):
                continue

            results.append({
                "chunk": chunk,
                "keyword_score": score
            })

            if len(results) >= top_k:
                break

        return results

    # --------------------------------------------------------
    # HYBRID SEARCH USING RRF
    # --------------------------------------------------------

    def hybrid_search(
        self,
        query,
        top_k=INITIAL_CANDIDATES,
        document_id=None
    ):
        """Combine semantic and keyword search rankings."""

        semantic_results = self.semantic_search(
            query,
            top_k=top_k,
            document_id=document_id
        )

        keyword_results = self.keyword_search(
            query,
            top_k=top_k,
            document_id=document_id
        )

        fused = {}

        def add_results(results):
            for rank, result in enumerate(results, start=1):
                chunk = result["chunk"]
                chunk_id = chunk["chunk_id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "chunk": chunk,
                        "rrf_score": 0.0,
                        "semantic_score": None,
                        "keyword_score": None
                    }

                fused[chunk_id]["rrf_score"] += (
                    1.0 / (RRF_K + rank)
                )

                if "semantic_score" in result:
                    fused[chunk_id]["semantic_score"] = (
                        result["semantic_score"]
                    )

                if "keyword_score" in result:
                    fused[chunk_id]["keyword_score"] = (
                        result["keyword_score"]
                    )

        add_results(semantic_results)
        add_results(keyword_results)

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        return ranked[:top_k]

    # --------------------------------------------------------
    # RERANKING
    # --------------------------------------------------------

    def rerank_results(self, query, results):
        """Rerank retrieved chunks with a cross-encoder."""

        if not results or not self.enable_reranking:
            return results

        self.load_models()

        pairs = [
            (query, item["chunk"]["text"])
            for item in results
        ]

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=False
        )

        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)

        return sorted(
            results,
            key=lambda item: item["rerank_score"],
            reverse=True
        )

    # --------------------------------------------------------
    # FINAL RETRIEVAL
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=FINAL_RESULTS,
        document_id=None
    ):
        """Return ranked chunks with standardized metadata."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Please enter a question.")

        if self.index is None or self.bm25 is None:
            raise RuntimeError(
                "Build the indexes before searching."
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        start_time = time.perf_counter()

        candidates = self.hybrid_search(
            query,
            top_k=max(INITIAL_CANDIDATES, top_k),
            document_id=document_id
        )

        ranked = self.rerank_results(
            query,
            candidates
        )

        final_results = ranked[:top_k]

        results = []

        for rank, item in enumerate(final_results, start=1):
            chunk = item["chunk"]

            if item.get("rerank_score") is not None:
                final_score = item["rerank_score"]
            else:
                final_score = item["rrf_score"]

            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "text": chunk["text"],
                "source": chunk["file_name"],
                "page": chunk["page_number"],
                "score": float(final_score),
                "semantic_score": item.get("semantic_score"),
                "keyword_score": item.get("keyword_score"),
                "rrf_score": item.get("rrf_score"),
                "rerank_score": item.get("rerank_score")
            })

        return {
            "query": query,
            "results": results,
            "retrieval_time_seconds": round(
                time.perf_counter() - start_time,
                4
            )
        }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():
    engine = RetrievalEngine()

    try:
        engine.load_chunks()
        engine.load_models()

        engine.chunk_fingerprint = engine.calculate_fingerprint()

        if not engine.load_saved_indexes():
            engine.build_indexes()
            engine.save_indexes()

        print("\nRetrieval engine is ready.")

        while True:
            query = input(
                "\nAsk a question (or type exit): "
            ).strip()

            if query.lower() in {"exit", "quit"}:
                break

            if not query:
                continue

            result = engine.retrieve(query)

            print(
                f"\nRetrieval time: "
                f"{result['retrieval_time_seconds']} seconds"
            )

            for item in result["results"]:
                print("\n" + "=" * 60)
                print("Source:", item["source"])
                print("Page:", item["page"])
                print("Score:", item["score"])
                print(item["text"][:1000])

    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()