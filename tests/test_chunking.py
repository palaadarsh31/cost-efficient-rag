import pytest

from docqa.chunking import TOKEN_RE, chunk_document, count_tokens
from docqa.loaders import Block, Document, load_document

from conftest import CORPUS


def _doc(paragraphs, headings_at=()):
    blocks = []
    for i, text in enumerate(paragraphs):
        kind = "heading" if i in headings_at else "text"
        blocks.append(Block(text, kind=kind, section=None))
    return Document(doc_id="t.md", path=None, doc_type="md", collection="default", title="t",
                    content_hash="x", blocks=blocks).finalize()


def test_chunks_are_exact_spans_and_within_budget():
    doc = load_document(CORPUS / "prometheus/practices-histograms.md", CORPUS)
    chunks = chunk_document(doc, chunk_size=200, overlap=30)
    assert len(chunks) > 5
    for c in chunks:
        assert doc.text[c.start:c.end] == c.text
        assert c.n_tokens <= 200
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap_by_whole_sentences():
    sentences = [f"Sentence number {i} talks about topic {i} in some detail." for i in range(60)]
    doc = _doc([" ".join(sentences)])
    chunks = chunk_document(doc, chunk_size=60, overlap=15)
    assert len(chunks) > 3
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end, "expected overlap between neighbouring chunks"
        assert doc.text[b.start - 1] == " " or b.start == 0  # starts at a sentence boundary


def test_no_overlap_when_disabled():
    doc = _doc([f"Paragraph {i} " + "word " * 30 for i in range(10)])
    chunks = chunk_document(doc, chunk_size=80, overlap=0)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start >= a.end


def test_oversized_sentence_is_windowed():
    doc = _doc(["token " * 500])
    chunks = chunk_document(doc, chunk_size=100, overlap=10)
    assert all(c.n_tokens <= 100 for c in chunks)
    assert sum(count_tokens(c.text) for c in chunks) >= 500


def test_chunk_never_ends_on_a_heading():
    paras = ["Intro " + "text " * 40, "Heading Two", "Body " + "more " * 40, "Heading Three", "End " + "x " * 40]
    doc = _doc(paras, headings_at=(1, 3))
    for c in chunk_document(doc, chunk_size=50, overlap=0):
        assert not c.text.rstrip().endswith(("Heading Two", "Heading Three"))


def test_overlap_must_be_smaller_than_size():
    doc = _doc(["a b c"])
    try:
        chunk_document(doc, chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


@pytest.mark.parametrize("size,overlap", [(8, 0), (8, 2), (16, 3), (128, 16)])
def test_carried_headings_and_full_body_never_exceed_budget_or_drop_tokens(size, overlap):
    doc = _doc(["Heading One", "Heading Two", " ".join(f"word{i}" for i in range(size)),
                "Final Heading"], headings_at=(0, 1, 3))
    chunks = chunk_document(doc, chunk_size=size, overlap=overlap)
    assert chunks and all(0 < c.n_tokens <= size for c in chunks)
    assert all(doc.text[c.start:c.end] == c.text for c in chunks)
    assert all(any(c.start <= token.start() and c.end >= token.end() for c in chunks)
               for token in TOKEN_RE.finditer(doc.text))


@pytest.mark.parametrize("size,overlap", [(128, 16), (256, 32), (350, 50), (512, 64), (800, 100)])
def test_corpus_chunk_budgets_and_coverage(size, overlap):
    for path in sorted(CORPUS.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".md", ".html", ".pdf"):
            doc = load_document(path, CORPUS)
            chunks = chunk_document(doc, chunk_size=size, overlap=overlap)
            assert all(c.n_tokens <= size for c in chunks), path
            assert all(doc.text[c.start:c.end] == c.text for c in chunks), path
            assert all(any(c.start <= token.start() and c.end >= token.end() for c in chunks)
                       for token in TOKEN_RE.finditer(doc.text)), path


def test_overlapping_unit_packing_uses_actual_source_span():
    doc = _doc(["token " * 13])
    chunks = chunk_document(doc, chunk_size=10, overlap=8)
    assert all(c.n_tokens <= 10 for c in chunks)
    assert all(any(c.start <= token.start() and c.end >= token.end() for c in chunks)
               for token in TOKEN_RE.finditer(doc.text))


def test_sentence_split_keeps_closing_punctuation():
    doc = _doc(['A sentence that ends here.) Another sentence that ends here." A final sentence.'])
    chunks = chunk_document(doc, chunk_size=9, overlap=0)
    assert all(any(c.start <= token.start() and c.end >= token.end() for c in chunks)
               for token in TOKEN_RE.finditer(doc.text))


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10)])
def test_invalid_chunk_budgets_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(_doc(["some text"]), chunk_size=size, overlap=overlap)
