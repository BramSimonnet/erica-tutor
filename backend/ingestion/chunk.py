def split_into_chunks(text, max_chars=1200):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += p + "\n"
        else:
            chunks.append(current.strip())
            current = p + "\n"

    if current:
        chunks.append(current.strip())

    return chunks
