import logging
from tempfile import NamedTemporaryFile
import os
import sys
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from langchain_core.documents import Document



class DoclingParser:
    """Parser for PDF documents using Docling library.
    
    Converts PDF files to markdown format with support for table structure
    extraction and page-level document splitting.
    """
    
    def __init__(self) -> None:
        """Initialize the Docling parser with optimized pipeline options."""
        sys.setrecursionlimit(10000)
            
        accelerator_options = AcceleratorOptions(
            num_threads=2, device=AcceleratorDevice.CPU
        )

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            do_formula_enrichment=False,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                mode=TableFormerMode.FAST,
            ),
            accelerator_options=accelerator_options,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseV4DocumentBackend,
                )
            }
)

    def parse(self, file_data: bytes, file_name: str) -> list[Document]:
        """Parse PDF file from bytes into page-level documents.
        
        Args:
            file_data: PDF file content as bytes.
            file_name: Name of the file for metadata.
            
        Returns:
            List of Document objects, one per page, with page number and file name
            in metadata.
        """
        pages: list[Document] = []
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name
        
        try:
            logging.info(f"Parsing {file_name}...")
            result = self.converter.convert(tmp_file_path)
            logging.info(f"Parsed {file_name}, extracting pages...")
            document = result.document
            for page_idx in range(len(document.pages)):
                pages.append(Document(page_content=document.export_to_markdown(page_no=page_idx), metadata={"page_number": page_idx+1, "file_name": file_name}))
            logging.info(f"Extracted {len(pages)} pages from {file_name}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
        return pages


