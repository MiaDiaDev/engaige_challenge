import re
from typing import List

from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document


class LegalDocumentSplitter(DocumentSplitter):
    """Custom splitter for German legal code that detects sections and structure"""

    def _create_doc(self, content: str, main_section: str, subsection: str,
                    para_num: str, original_doc: Document) -> Document:
        """Creates a new document for a section with metadata"""
        meta = {
            "main_section": main_section,
            "subsection": subsection,
            "paragraph_number": para_num
        }

        # Preserve original metadata
        if original_doc.meta:
            meta.update(original_doc.meta)

        doc = Document(content=content, meta=meta)
        return doc

    def split(self, documents: List[Document]) -> List[Document]:
        split_docs = []

        for doc in documents:
            # Initialize tracking
            main_section = ""
            subsection = ""
            para_num = ""
            current_text = []

            # Split into lines and process
            lines = doc.content.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Main section detection (e.g. "A. Private Altersvorsorge")
                if re.match(r'^[A-Z]\.\s+\w+', line):
                    # Save previous section if exists
                    if current_text:
                        split_docs.append(self._create_doc(
                            content="\n".join(current_text),
                            main_section=main_section,
                            subsection=subsection,
                            para_num=para_num,
                            original_doc=doc
                        ))
                        current_text = []
                    main_section = line
                    subsection = ""

                # Subsection detection (e.g. "I. Förderung")
                elif re.match(r'^(?:I|V|X)+\.\s+\w+', line):
                    if current_text:
                        split_docs.append(self._create_doc(
                            content="\n".join(current_text),
                            main_section=main_section,
                            subsection=subsection,
                            para_num=para_num,
                            original_doc=doc
                        ))
                        current_text = []
                    subsection = line

                # Paragraph number detection (e.g. "123" or "Rz. 123")
                elif re.match(r'^\d+$|\bRz\.\s*\d+', line):
                    if current_text:
                        split_docs.append(self._create_doc(
                            content="\n".join(current_text),
                            main_section=main_section,
                            subsection=subsection,
                            para_num=para_num,
                            original_doc=doc
                        ))
                        current_text = []
                    para_num = line

                current_text.append(line)

            # Add final section
            if current_text:
                split_docs.append(self._create_doc(
                    content="\n".join(current_text),
                    main_section=main_section,
                    subsection=subsection,
                    para_num=para_num,
                    original_doc=doc
                ))

        return split_docs
