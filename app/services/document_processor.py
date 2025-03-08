# app/services/document_processor.py
from pathlib import Path
from typing import Dict, List, Optional
import logging
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

class DocumentProcessor:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.batch_size = 20  # Process 20 chunks at a time

    def process_document(self, file_path: Path, metadata: Dict) -> List[Dict]:
        try:
            # Check for existing document
            existing_docs = self.vectorstore.similarity_search(
                "Check for duplicates",
                filter={"file_id": metadata.get("file_id")},
                k=1
            )
            
            if existing_docs:
                logger.info(f"Document {metadata.get('title')} already exists in vectorstore. Skipping processing.")
                return []

            file_extension = file_path.suffix.lower()
            logger.info(f"Processing document with extension: {file_extension}")
            
            if file_extension not in ['.pdf', '.docx']:
                raise DocumentProcessingError(f"Unsupported file type: {file_extension}")

            # Load document
            loader = PyMuPDFLoader(str(file_path)) if file_extension == '.pdf' else Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            if not documents:
                raise DocumentProcessingError("Document loaded but contains no text")

            # Add metadata
            for doc in documents:
                doc.metadata.update(metadata)

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            if not chunks:
                raise DocumentProcessingError("Document split resulted in no chunks")

            # Process chunks in batches
            all_chunks = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                
                texts = []
                metadatas = []
                
                for chunk in batch:
                    if chunk.page_content.strip():
                        texts.append(chunk.page_content)
                        chunk_metadata = chunk.metadata.copy()
                        chunk_metadata['text'] = chunk.page_content
                        metadatas.append(chunk_metadata)

                if texts:
                    # Add batch to vectorstore
                    self.vectorstore.add_texts(
                        texts=texts,
                        metadatas=metadatas
                    )
                    all_chunks.extend([{"chunk_id": len(all_chunks) + i, "text": text[:100]} 
                                     for i, text in enumerate(texts)])

            if not all_chunks:
                raise DocumentProcessingError("No valid text content found in document")
                
            return all_chunks
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise DocumentProcessingError(str(e))