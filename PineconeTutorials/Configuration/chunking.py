

import io
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# CONSTANTS
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# ═══════════════════════════════════════════════════════════════
# STAGE 1: LOAD — File object → plain text
# ═══════════════════════════════════════════════════════════════
def load_document(file_obj, filename: str) -> str:
    
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Reset stream to start (in case it was partially read)
    try:
        file_obj.seek(0)
    except (AttributeError, io.UnsupportedOperation):
        pass

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_obj)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        import docx2txt
        return docx2txt.process(file_obj)

    elif ext in (".txt", ".md"):
        data = file_obj.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return data


# ═══════════════════════════════════════════════════════════════
# STAGE 2: NORMALIZE — Clean whitespace
# ═══════════════════════════════════════════════════════════════
def normalize(text: str) -> str:
    """Clean whitespace without destroying paragraph structure."""
    text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ newlines → 2
    text = re.sub(r"[ \t]+", " ", text)      # multiple spaces → 1
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════════════
# STAGE 3: MERGE HEADINGS — Attach short lines to next paragraph
# ═══════════════════════════════════════════════════════════════
def merge_headings(text: str, max_heading_words: int = 8) -> str:
    """Merge short heading lines with the following paragraph."""
    lines = text.split("\n")
    merged = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            i += 1
            continue

        # Short line followed by content → merge
        if 0 < len(line.split()) <= max_heading_words and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line:
                merged.append(f"{line}\n{next_line}")
                i += 2
                continue

        merged.append(line)
        i += 1

    return "\n\n".join(m for m in merged if m)


# ═══════════════════════════════════════════════════════════════
# STAGE 4: SPLIT — Recursive, sentence-aware chunking
# ═══════════════════════════════════════════════════════════════
def split_into_chunks(
    text: str,
    words_per_chunk: int = 300,
    overlap_words: int = 45,
) -> list[str]:
    """Split text into ~300-word chunks at natural boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=words_per_chunk,
        chunk_overlap=overlap_words,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    return splitter.split_text(text)


# ═══════════════════════════════════════════════════════════════
# STAGE 5: CLEAN — Strip punctuation, drop tiny chunks
# ═══════════════════════════════════════════════════════════════
def clean_chunks(chunks: list[str], min_words: int = 40) -> list[str]:
    """Strip orphan punctuation, drop chunks smaller than min_words."""
    cleaned = []
    for c in chunks:
        c = c.strip()
        c = re.sub(r"^[.,;:!?\-\s]+", "", c)
        c = re.sub(r"[ \t]+", " ", c)
        if len(c.split()) >= min_words:
            cleaned.append(c)
    return cleaned


# ═══════════════════════════════════════════════════════════════
# STAGE 6: STRUCTURE — Package chunks with metadata
# ═══════════════════════════════════════════════════════════════
def structure_chunks(chunks: list[str], source: str) -> list[dict]:
    """Wrap chunks with metadata for downstream use."""
    total = len(chunks)
    return [
        {
            "chunk_index": i,
            "text": chunk,
            "word_count": len(chunk.split()),
            "char_count": len(chunk),
            "source": source,
            "total_chunks": total,
        }
        for i, chunk in enumerate(chunks)
    ]


# ═══════════════════════════════════════════════════════════════
# MASTER FUNCTION — Full pipeline
# ═══════════════════════════════════════════════════════════════
def process_document(
    file_obj,
    filename: str,
    words_per_chunk: int = 300,
    overlap_words: int = 45,
    min_words: int = 40,
) -> list[dict]:
    
    print(f"📄 Loading: {filename}")
    raw = load_document(file_obj, filename)
    print(f"   ✅ Loaded {len(raw):,} characters")

    print("🧹 Normalizing...")
    clean = normalize(raw)

    print("🔗 Merging headings...")
    clean = merge_headings(clean)
    print(f"   ✅ Total words: {len(clean.split()):,}")

    print(f"✂️  Splitting (~{words_per_chunk} words/chunk)...")
    chunks = split_into_chunks(clean, words_per_chunk, overlap_words)
    print(f"   ✅ Raw chunks: {len(chunks)}")

    print(f"🧽 Cleaning (min {min_words} words)...")
    chunks = clean_chunks(chunks, min_words)
    print(f"   ✅ Final chunks: {len(chunks)}")

    return structure_chunks(chunks, source=filename)

