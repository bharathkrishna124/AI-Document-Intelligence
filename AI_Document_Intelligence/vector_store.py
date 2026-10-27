import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentVectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.index = None
        self.chunks = []

    def split_documents(
        self,
        documents,
        chunk_size=800,
        overlap=150
    ):
        """
        Split extracted page text into overlapping chunks.
        Preserve source filename and page number.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size"
            )

        chunks = []
        step = chunk_size - overlap

        for document in documents:
            text = document["text"]

            for start in range(0, len(text), step):
                chunk_text = text[start:start + chunk_size]

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "source": document["source"],
                        "page": document["page"]
                    })

        return chunks

    def build_index(self, chunks):
        """
        Generate embeddings and create a FAISS search index.
        """

        if not chunks:
            raise ValueError("No text chunks were provided")

        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, question, top_k=4):
        """
        Retrieve the most relevant document chunks.
        """

        if self.index is None:
            raise RuntimeError("Build the index before searching")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32
        )

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            question_embedding,
            k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            result = self.chunks[int(index)].copy()
            result["score"] = float(score)

            results.append(result)

        return results