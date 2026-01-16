import os
import traceback
import logging
from utils.logger import setup_logging
from utils.rag.document_processing_pipeline import DocumentProcessingPipeline
from utils.rag.parser import DoclingParser
from utils.rag.chunker import MarkdownChunker
from utils.rag.vector_db import QdrantDB
from utils.rag.dropbox_downloader import DropboxDownloader
from utils.chat_client import ChatClient
# Set up logging
setup_logging()

# Dropbox shared folder link from .env
DROPBOX_SHARED_LINK = os.getenv("DROPBOX_SHARED_LINK", "")

# Initialize Dropbox downloader
_downloader = None

def get_downloader():
    """Get or create Dropbox downloader instance."""
    global _downloader
    if _downloader is None:
        if not DROPBOX_SHARED_LINK:
            raise ValueError("DROPBOX_SHARED_LINK not set in .env file")
        _downloader = DropboxDownloader(shared_link=DROPBOX_SHARED_LINK)
    return _downloader

def main():
    logging.info("Initializing pipeline...")
    # Initialize pipeline
    document_processing_pipeline = DocumentProcessingPipeline(
        parser=DoclingParser(),
        chunker=MarkdownChunker(
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1000")), 
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        ),
        qdrant_db=QdrantDB(
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            collection_name=os.getenv("COLLECTION_NAME", ""), 
            qdrant_host=os.getenv("QDRANT_HOST", ""), 
            qdrant_api_key=os.getenv("QDRANT_API_KEY") ,
        )
    )

    # Get Dropbox downloader
    logging.info(f"Connecting to Dropbox folder: {DROPBOX_SHARED_LINK}")
    downloader = get_downloader()
    
    # List all files from the folder
    logging.info("Listing files from folder...")
    files = downloader.list_files()
    logging.info(f"Found {len(files)} files")
    
    if not files:
        logging.error("No files found in the folder. Exiting.")
        return
    
    # Process each file
    for file_name, file_path in files:
        logging.info(f"Processing {file_name}...")
        try:
            logging.info(f"Downloading {file_name}...")
            file_data = downloader.download_file(file_path)

            logging.info(f"Processing {file_name}...")
            document_processing_pipeline.process(file_data=file_data, file_name=file_name)
            logging.info(f"Completed processing {file_name}")
        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")
            logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()