"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import re
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Split documents into heading-based chunks
        self.chunks = [
            chunk
            for filename, text in self.documents
            for chunk in self.chunk_document(text, filename)
        ]

        # Build a retrieval index over chunks (implemented in Phase 1)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------

    def chunk_document(self, text, filename):
        """Split text on double newlines (paragraphs) into sections for more precise retrieval."""
        sections = re.split(r'\n\n+', text)
        return [(filename, s.strip()) for s in sections if s.strip()]

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, chunks):
        """
        Build an inverted index mapping lowercase words to chunk indices.
        This allows precise retrieval of chunks containing specific words.
        """
        index = {}
        for i, (filename, text) in enumerate(chunks):
            for token in text.lower().split():
                token = token.strip(".,!?;:\"'()[]{}")
                if token:
                    if token not in index:
                        index[token] = []
                    index[token].append(i)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a relevance score as the ratio of matching query words to total query words.
        This normalizes for query length and provides a score between 0 and 1.
        """
        text_words = set(text.lower().split())
        query_words = set(query.lower().split())  # Use set to avoid double counting
        matching = sum(1 for word in query_words if word in text_words)
        return matching / len(query_words) if query_words else 0
        """
        Use the index to find candidate chunks containing query words,
        score them, and return top_k relevant snippets.
        """
        candidates = set()
        for word in query.lower().split():
            word = word.strip(".,!?;:\"'()[]{}")
            for i in self.index.get(word, []):
                candidates.add(i)

        results = []
        for i in candidates:
            filename, text = self.chunks[i]
            score = self.score_document(query, text)
            if score >= min_score:
                results.append((filename, text, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return [(filename, text) for filename, text, _ in results[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
