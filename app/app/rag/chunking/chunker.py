import hashlib
import re

from app.rag.models import Document, DocumentChunk


_MARKER_RE = re.compile(
    r"\[(Page|Slide|Sheet):?\s*([^\]]+)\]",
    re.IGNORECASE,
)


class DocumentChunker:
    """
    Paragraph-aware document chunker.

    Responsibilities:
    - Normalize document text
    - Preserve Page / Slide / Sheet locations
    - Split documents into manageable chunks
    - Apply chunk overlap
    - Generate deterministic chunk IDs
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be > 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap must be >= 0"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be < chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def chunk(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        text = self._normalize(
            document.text
        )

        if not text:
            return []

        sections = self._split_sections(
            text
        )

        chunks: list[DocumentChunk] = []

        current = ""

        current_location: dict[str, str] = {}

        for section_text, location in sections:

            # Update current source location.
            if location:
                current_location = location.copy()

            paragraphs = re.split(
                r"\n\s*\n",
                section_text,
            )

            for paragraph in paragraphs:

                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                candidate = (
                    f"{current}\n\n{paragraph}"
                    if current
                    else paragraph
                ).strip()

                # --------------------------------------------------
                # Paragraph fits into current chunk
                # --------------------------------------------------

                if len(candidate) <= self.chunk_size:

                    current = candidate

                    continue

                # --------------------------------------------------
                # Current chunk is full
                # --------------------------------------------------

                if current:

                    chunks.append(
                        self._create_chunk(
                            document=document,
                            text=current,
                            index=len(chunks),
                            location=current_location,
                        )
                    )

                # --------------------------------------------------
                # Start next chunk with overlap
                # --------------------------------------------------

                overlap = ""

                if self.chunk_overlap:

                    overlap = current[
                        -self.chunk_overlap:
                    ]

                current = (
                    f"{overlap}\n\n{paragraph}"
                    if overlap
                    else paragraph
                ).strip()

                # --------------------------------------------------
                # Handle extremely large paragraphs
                # --------------------------------------------------

                while len(current) > self.chunk_size:

                    piece = current[
                        :self.chunk_size
                    ].strip()

                    chunks.append(
                        self._create_chunk(
                            document=document,
                            text=piece,
                            index=len(chunks),
                            location=current_location,
                        )
                    )

                    start = (
                        self.chunk_size
                        - self.chunk_overlap
                    )

                    current = current[
                        start:
                    ].strip()

        # ----------------------------------------------------------
        # Final chunk
        # ----------------------------------------------------------

        if current:

            chunks.append(
                self._create_chunk(
                    document=document,
                    text=current,
                    index=len(chunks),
                    location=current_location,
                )
            )

        return chunks

    # ==========================================================
    # SECTION DETECTION
    # ==========================================================

    def _split_sections(
        self,
        text: str,
    ) -> list[tuple[str, dict[str, str]]]:

        matches = list(
            _MARKER_RE.finditer(text)
        )

        if not matches:

            return [
                (
                    text,
                    {},
                )
            ]

        sections = []

        # Text before first marker
        if matches[0].start() > 0:

            prefix = text[
                :matches[0].start()
            ].strip()

            if prefix:

                sections.append(
                    (
                        prefix,
                        {},
                    )
                )

        for i, match in enumerate(matches):

            start = match.end()

            if i + 1 < len(matches):

                end = matches[
                    i + 1
                ].start()

            else:

                end = len(text)

            section_text = text[
                start:end
            ].strip()

            marker_type = (
                match.group(1)
                .strip()
                .lower()
            )

            marker_value = (
                match.group(2)
                .strip()
            )

            if marker_type == "page":

                location = {
                    "page": marker_value
                }

            elif marker_type == "slide":

                location = {
                    "slide": marker_value
                }

            else:

                location = {
                    "sheet": marker_value
                }

            if section_text:

                sections.append(
                    (
                        section_text,
                        location,
                    )
                )

        return sections

    # ==========================================================
    # TEXT NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        # Normalize line endings
        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Replace tabs / repeated spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Normalize excessive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ==========================================================
    # DETERMINISTIC CHUNK ID
    # ==========================================================

    @staticmethod
    def create_chunk_id(
        document_id: str,
        chunk_index: int,
    ) -> str:

        """
        Generate a deterministic chunk ID.

        The same document + chunk index will always
        produce the same ID.
        """

        raw_id = (
            f"{document_id}:{chunk_index}"
        )

        return hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()[:24]

    # ==========================================================
    # CREATE CHUNK
    # ==========================================================

    @classmethod
    def _create_chunk(
        cls,
        document: Document,
        text: str,
        index: int,
        location: dict[str, str],
    ) -> DocumentChunk:

        chunk_id = cls.create_chunk_id(
            document_id=document.document_id,
            chunk_index=index,
        )

        metadata = {
            **document.metadata,

            "source_path": (
                document.source_path
            ),

            "file_name": (
                document.file_name
            ),

            "file_type": (
                document.file_type
            ),

            "document_id": (
                document.document_id
            ),

            "chunk_id": chunk_id,

            "chunk_index": index,

            **location,
        }

        return DocumentChunk(
            chunk_id=chunk_id,

            document_id=(
                document.document_id
            ),

            text=text,

            chunk_index=index,

            metadata=metadata,
        )