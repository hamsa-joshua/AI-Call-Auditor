import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class RagEngine:
    def __init__(self, policy_path, index_path="database/vector_store"):
        self.policy_path = policy_path
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def build_vector_store(self):
        """
        Reads policy file, splits into chunks, creates embeddings, and saves FAISS index.
        """
        print(f"Loading policy from {self.policy_path}...")
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")

        with open(self.policy_path, "r") as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_text(text)
        print(f"Split policy into {len(texts)} chunks.")

        print("Creating embeddings (this may take a moment on CPU)...")
        self.vector_store = FAISS.from_texts(texts, self.embeddings)
        
        print(f"Saving FAISS index to {self.index_path}...")
        self.vector_store.save_local(self.index_path)
        print("Vector store built and saved.")

    def load_vector_store(self):
        """
        Loads the existing FAISS index.
        """
        if os.path.exists(self.index_path) and os.path.exists(os.path.join(self.index_path, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True # Trusted local source
            )
            print("Vector store loaded.")
        else:
            print("Vector store not found. Building new one...")
            self.build_vector_store()

    def retrieve_context(self, query, k=3):
        """
        Retrieves top k relevant policy snippets for a given query.
        """
        if not self.vector_store:
            self.load_vector_store()
            
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
