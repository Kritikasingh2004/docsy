from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Text extracted from a single PDF page."""

    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)


class ChunkMetadata(BaseModel):
    """Metadata attached to an indexed text chunk"""

    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)


class DocumentChunk(BaseModel):
    """A chunk of document text ready for embedding."""

    text: str = Field(min_length=1)
    metadata: ChunkMetadata
