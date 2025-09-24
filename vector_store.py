import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from uu import Error
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.document_converter import DocumentConverter

from embedding import Embedding


class VectorStore:
    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "faiss_index")
        self.vectorstore = None
        self.embedding_model = Embedding()

        os.makedirs(storage_dir, exist_ok=True)
        if self.load_index():
            print(f"▶ Initialized: existing index found at {self.index_path}")
        else:
            print("▶ Initialized: no existing index, ready to create new one")


    def load_from_gdrive(self, temp_path: str, folder_id: str) -> List[Document]:
        os.makedirs(temp_path, exist_ok=True)
        try:
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()
            drive = GoogleDrive(gauth)
            file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
            all_chunks = []

            for file in file_list:
                local_path = os.path.join(temp_path, file['title'])
                file.GetContentFile(local_path)
                print("Now exists:", Path(local_path).exists(), "at", local_path)
                input("Press Enter to continue...")  # pause so you can check manually

                if file['mimeType'] == "application/pdf" or local_path.lower().endswith(".pdf"):
                    print(f"▶ Processing PDF with OCR: {file['title']}")
                    text = self.ocr_pdf(local_path)
                else:
                    print(f"▶ Processing text file: {file['title']}")
                    text = Path(local_path).read_text(encoding='utf-8', errors='ignore')
                chunks = self.chunking(text, local_path)
                all_chunks.extend(chunks)
            return all_chunks
        except Exception as e:
            raise Error(f"Error loading from Google Drive: {e}")
        finally:
            shutil.rmtree(temp_path)


    def ocr_pdf(self, path: str) -> str:
        try:
            ocr_options = EasyOcrOptions(lang=["vie"], force_full_page_ocr=True)
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                ocr_options=ocr_options,
                table_structure_options=TableFormerMode.ACCURATE
            )
            converter = DocumentConverter(pipeline_options)
            text = converter.convert(path).document
            return text.export_to_markdown()
        except Exception as e:
            raise Error(f"Error during OCR processing: {e}")


    def chunking(self, text: str, source_path: str) -> List[Document]:
        try:
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
            )
            sections = header_splitter.split_text(text)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            chunks = []
            for section in sections:
                meta = section.metadata or {}
                header_path = " > ".join([v for k, v in meta.items() if k.startswith("h")])
                base_meta = {
                    "source": str(source_path),
                    "section": header_path,
                }
                split_docs = text_splitter.split_documents([Document(page_content=section.page_content, metadata=base_meta)])
                chunks.extend(split_docs)
            return chunks
        except Exception as e:
            raise Error(f"Error during text chunking: {e}")

    
    def create_vector_store(self, documents: List[Document]):
        if not documents:
            print("No documents provided.")
            return self
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
            self.save_index()
            print(f"▶ Created vector store with {len(documents)} documents")
            return self
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return self
    

    def load_index(self) -> bool:
        return os.path.exists(self.index_path)
    

    def load_index_with_embeddings(self) -> bool:
        if not self.load_index():
            return False
        try:
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            doc_count = len(self.vectorstore.docstore._dict)
            print(f"▶ Loaded vector store with {doc_count} documents")
            return True
        except Exception as e:
            print(f"Failed to load index with embeddings: {e}")
            return False
        
    
    def save_index(self) -> bool:
        if self.vectorstore is None:
            print("Nothing to save")
            return False
        try:
            self.vectorstore.save_local(self.index_path)
            print(f"▶ Saved vector store to {self.index_path}")
            return True
        except Exception as e:
            raise Error(f"Error saving index: {e}")


    def delete_index(self) -> bool:
        if os.path.exists(self.index_path):
            try:
                shutil.rmtree(self.index_path)
                self.vectorstore = None
                print(f"▶ Deleted index at {self.index_path}")
                return True
            except Exception as e:
                raise Error(f"Error deleting index: {e}")
        else:
            print("No index to delete")
            return False


    def get_vector_results(self, query: str, top_k: int = 5, use_mmr: bool = True, lambda_mult: float = 0.5) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            self.load_index_with_embeddings()

        try:
            docs = self.vectorstore.max_marginal_relevance_search(query, k=top_k, fetch_k=max(top_k * 4, 10), lambda_mult=lambda_mult)
            rescored = self.vectorstore.similarity_search_with_score(query, k=max(top_k * 4, 10))
            score_map = {d.page_content: float(s) for d, s in rescored}
            return [
                {
                    "text": d.page_content,
                    "score": score_map.get(d.page_content, 0.0),
                    "metadata": d.metadata,
                    "rank": i + 1,
                }
                for i, d in enumerate(docs[:top_k])
            ]
        except Exception as e:
            raise Error(f"Error during vector search: {e}")