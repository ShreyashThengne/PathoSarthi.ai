"""
Classes:
    VectorDBGeneration: Handles the end-to-end process of creating a FAISS vector database from PDF documents.
                        This class loads PDF files from a specified directory, splits them into manageable text chunks,
                        generates embeddings using a HuggingFace model, and saves the resulting FAISS vector database locally.
    Args:
        db_faiss_path (str): Path where the FAISS vector database will be saved.
        extension (str): File extension to filter documents (e.g., '.pdf').
        books_path (str): Directory path containing the source PDF documents.
    Methods:
        load_pdfs(path): Loads PDF documents from the specified directory.
        generate_chunks(documents): Splits documents into text chunks for embedding.
        generate(): Orchestrates the loading, splitting, embedding, and saving of the vector database.
        Initializes the VectorDBGeneration class with the specified database path, file extension, and source directory.
"""

from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


# self.file_type = "*.pdf"
# self.books_path = 'medical_texts/'
# self.db_faiss_path = "vectorstore/haematology_db"


class VectorDBGeneration:
    """Handles the end-to-end process of creating a FAISS vector database from PDF documents.
    This class loads PDF files from a specified directory, splits them into manageable text chunks,
    generates embeddings using a HuggingFace model, and saves the resulting FAISS vector database locally."""

    def __init__(self, db_faiss_path, extension, books_path):
        self.db_faiss_path = db_faiss_path
        self.file_type = f"*{extension}"
        self.books_path = books_path
        self.embedding_model = HuggingFaceEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")

    def load_pdfs(self, path):
        """Loads PDF documents from the specified directory."""

        loader = DirectoryLoader(
            path=path,
            glob=self.file_type,
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return documents

    @staticmethod
    def generate_chunks(documents):
        """Splits documents into text chunks for embedding."""

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
        text_chunks = text_splitter.split_documents(documents)

        return text_chunks
    
    def generate(self):
        """Orchestrates the loading, splitting, embedding, and saving of the vector database."""
        
        docs = self.load_pdfs(self.books_path)
        doc_chunks = self.generate_chunks(docs)

        db = FAISS.from_documents(doc_chunks, embedding=self.embedding_model)
        db.save_local(self.db_faiss_path)

if __name__ == "__main__":
    db_faiss_path = "vectorstore/context_db"
    extension = ".pdf"
    books_path = 'medical_texts/'
    vector_db_gen = VectorDBGeneration(db_faiss_path, extension, books_path)
    vector_db_gen.generate()