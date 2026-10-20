from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Locate the knowledge folder
PROJECT_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"


def load_documents():
    """
    Load all Markdown documents from the knowledge folder.
    """

    documents = []

    for file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = file.read_text(encoding="utf-8")

        if content.strip():
            documents.append({
                "source": file.name,
                "content": content
            })

    if not documents:
        raise FileNotFoundError(
            "No Markdown knowledge documents found."
        )

    return documents


def split_into_chunks(text, chunk_size=150, overlap=30):
    """
    Split a document into overlapping word-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            continue

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


class AutomotiveRAGRetriever:
    """
    TF-IDF retrieval system for automotive knowledge documents.
    """

    def __init__(self):
        self.documents = load_documents()
        self.chunks = []

        for document in self.documents:
            passages = split_into_chunks(document["content"])

            for index, passage in enumerate(passages):
                self.chunks.append({
                    "source": document["source"],
                    "chunk_id": index,
                    "text": passage
                })

        if not self.chunks:
            raise ValueError("No searchable text found.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def search(self, question, top_k=3):
        """
        Retrieve relevant passages and their sources.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_vector = self.vectorizer.transform([question])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices:
            score = float(similarities[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append({
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "relevance_score": round(score, 4),
                "passage": chunk["text"]
            })

            if len(results) >= top_k:
                break

        return results


def main():
    retriever = AutomotiveRAGRetriever()

    questions = [
        "Why was Ridge regression selected?",
        "How was forecast accuracy measured?",
        "What are the limitations of the forecasting model?"
    ]

    for question in questions:
        print(f"\nQUESTION: {question}")

        results = retriever.search(question)

        if not results:
            print("No relevant passages found.")
            continue

        for result in results:
            print(f"\nSource: {result['source']}")
            print(f"Relevance score: {result['relevance_score']}")
            print(f"Passage: {result['passage']}")
            print("-" * 60)


if __name__ == "__main__":
    main()
