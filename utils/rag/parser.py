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
    def __init__(self) -> None:
        sys.setrecursionlimit(10000)
            
        accelerator_options = AcceleratorOptions(
            num_threads=2, device=AcceleratorDevice.CPU
        )

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,  # Can be disabled for speed
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
        """
        Parse a file from bytes.
        
        Args:
            file_data: File content as bytes
            file_name: Name of the file
            
        Returns:
            List of Document objects
        """
        pages: list[Document] = []
        # Write bytes to temporary file and parse
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
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
        return pages


