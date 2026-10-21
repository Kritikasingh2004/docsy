import pymupdf
import pytest

from app.rag.pdf import (
    chunk_pages,
    extract_text_by_page,
    process_pdf,
)
from app.rag.schema import DocumentPage


def make_pdf(*page_texts: str) -> bytes:
    """Create an in-memory PDF containing the supplied page texts."""
    doc = pymupdf.open()

    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_encrypted_pdf() -> bytes:
    """Create a password-protected PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret content")

    pdf_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    doc.close()
    return pdf_bytes


# -------------------------
# extract_text_by_page
# -------------------------


def test_extract_text_by_page_preserves_page_numbers():
    pdf_bytes = make_pdf("First page content", "Second page content")

    pages = extract_text_by_page(pdf_bytes)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "First page content" in pages[0].text
    assert "Second page content" in pages[1].text


def test_extract_text_by_page_skips_empty_pages():
    pdf_bytes = make_pdf("First page", "", "Third page")

    pages = extract_text_by_page(pdf_bytes)

    assert [page.page_number for page in pages] == [1, 3]


def test_extract_text_by_page_rejects_password_protected_pdf():
    pdf_bytes = make_encrypted_pdf()

    with pytest.raises(
        ValueError,
        match="Password-protected PDFs are not supported",
    ):
        extract_text_by_page(pdf_bytes)


# -------------------------
# chunk_pages
# -------------------------


def test_chunk_pages_splits_text_with_overlap():
    page = DocumentPage(
        page_number=3,
        text="one two three four five six seven eight nine ten",
    )

    chunks = chunk_pages([page], chunk_size=4, overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
        "ten",
    ]

    assert all(chunk.metadata.page_number == 3 for chunk in chunks)
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1, 2, 3]


def test_chunk_pages_handles_text_shorter_than_chunk_size():
    page = DocumentPage(page_number=1, text="hello world")

    chunks = chunk_pages([page], chunk_size=5, overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].metadata.page_number == 1
    assert chunks[0].metadata.chunk_index == 0


def test_chunk_pages_handles_zero_overlap():
    page = DocumentPage(
        page_number=1,
        text="one two three four five six",
    )

    chunks = chunk_pages([page], chunk_size=3, overlap=0)

    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "four five six",
    ]


def test_chunk_pages_returns_empty_list_for_empty_pages():
    page = DocumentPage(page_number=1, text=" ")

    assert chunk_pages([page]) == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_pages_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_pages([], chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_chunk_pages_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap must be"):
        chunk_pages([], chunk_size=4, overlap=overlap)


# -------------------------
# process_pdf
# -------------------------


def test_process_pdf_extracts_and_chunks_pdf():
    pdf_bytes = make_pdf("one two three four five six")

    chunks = process_pdf(pdf_bytes)

    assert chunks
    assert all(chunk.metadata.page_number == 1 for chunk in chunks)
    assert "one" in chunks[0].text


def test_process_pdf_rejects_pdf_without_extractable_text():
    pdf_bytes = make_pdf("")

    with pytest.raises(
        ValueError,
        match="No extractable text found",
    ):
        process_pdf(pdf_bytes)
