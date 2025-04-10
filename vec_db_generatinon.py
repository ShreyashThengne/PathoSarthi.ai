from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


# self.file_type = "*.pdf"
# self.books_path = 'medical_texts/'
# self.db_faiss_path = "vectorstore/haematology_db"


class VectorDBGeneration:
    def __init__(self, db_faiss_path, extension, books_path):
        self.db_faiss_path = db_faiss_path
        self.file_type = f"*{extension}"
        self.books_path = books_path
        self.embedding_model = HuggingFaceEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")

    def load_pdfs(self, path):
        loader = DirectoryLoader(
            path=path,
            glob=self.file_type,
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return documents

    @staticmethod
    def generate_chunks(documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
        text_chunks = text_splitter.split_documents(documents)

        return text_chunks
    
    def generate(self):
        docs = self.load_pdfs(self.books_path)
        doc_chunks = self.generate_chunks(docs)

        db = FAISS.from_documents(doc_chunks, embedding=self.embedding_model)
        db.save_local(self.db_faiss_path)

if __name__ == "__main__":
    db_faiss_path = "vectorstore/haematology_db"
    extension = ".pdf"
    books_path = 'medical_texts/'
    vector_db_gen = VectorDBGeneration(db_faiss_path, extension, books_path)
    vector_db_gen.generate()