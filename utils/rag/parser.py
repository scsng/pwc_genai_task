from xml.dom.minidom import Document
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
            print(f"Parsing {file_name}...")
            result = self.converter.convert(tmp_file_path)
            print(f"Parsed {file_name}, extracting pages...")
            for page in result.document.pages:
                pages.append(Document(page_content=page.export_to_markdown(), metadata=page.metadata, file_name=file_name))
            print(f"Extracted {len(pages)} pages from {file_name}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
        return pages






# Initialize the DocumentConverter and convert the document at the given path
