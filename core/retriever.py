from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, documents):

        self.documents = documents

        # ---- Embeddings ----
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding_model
        )

        # ---- BM25 ----
        self.tokenized_docs = [
            doc.page_content.split() for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    # -----------------------------------
    # Hybrid Search
    # -----------------------------------
    def retrieve(self, query, top_k=5):

        # Semantic search
        semantic_docs = self.vectorstore.similarity_search(query, k=top_k)

        # Keyword search
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_docs = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        keyword_docs = [doc for doc, _ in ranked_docs[:top_k]]

        # Merge + remove duplicates
        combined = {}
        for doc in semantic_docs + keyword_docs:
            combined[doc.page_content] = doc

        return list(combined.values())[:top_k]
