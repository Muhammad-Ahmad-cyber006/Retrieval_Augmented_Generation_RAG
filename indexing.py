import hashlib
import re
from pathlib import Path

import numpy as np

# Every vector we create has this many numbers. Real models use 768-3072.
DIM = 256


# STEP 1: COUNT TOKENS
# LLMs and embedding models read "tokens" (roughly 3/4 of a word), not
# characters. Chunk sizes are limited in tokens, so we need a way to count.
# Here a token is just a word. Real code would use tiktoken instead.
def count_tokens(text):
    return len(re.findall(r"\w+", text))


# STEP 2: EMBEDDINGS AND COSINE SIMILARITY
# An embedding turns text into a list of numbers (a vector) so that text with
# similar meaning gets similar vectors. This is what makes search by meaning
# possible, instead of search by exact words.
#
# Real embedding models (OpenAI, etc.) learn meaning from huge amounts of text.
# Our toy version only knows which words appear, so "cat" and "kitten" look
# unrelated to it. That is fine for learning the pipeline.
def embed(text):
    vec = np.zeros(DIM)
    for word in re.findall(r"\w+", text.lower()):
        # md5 gives the same big number for the same word every time,
        # so a word always lands in the same slot of the vector.
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        slot = h % DIM
        sign = 1 if (h >> 8) & 1 else -1  # random +1/-1 keeps collisions from piling up
        vec[slot] += sign
    return vec


# Cosine similarity measures the angle between two vectors:
#   1.0 = pointing the same way (very similar)
#   0.0 = unrelated
# It ignores vector length, so a long text and a short text on the same
# topic can still score high.
def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


# STEP 3: LOAD DOCUMENTS
# Read raw text from somewhere. Here: every .txt file in a folder.
# We keep the file path as "source" so that later we can say where an answer
# came from. Real projects also load web pages, PDFs, databases, and so on.
def load_docs(folder):
    return [{"text": p.read_text(), "source": str(p)} for p in sorted(Path(folder).glob("*.txt"))]


# STEP 4: SPLIT DOCUMENTS INTO CHUNKS
# Why split? One vector for a whole book would blur every topic together.
# Small chunks give each vector one focused idea, so search is sharper.
#
# Why "recursive"? Cutting every N tokens would slice sentences in half.
# Instead we try the biggest natural break first:
#     1) blank line (paragraph)  2) new line  3) space (word)
# and only fall back to a smaller break when a piece is still too big.
# That keeps paragraphs and sentences together as long as possible.
#
# chunk_size:    max tokens per chunk.
# chunk_overlap: tokens repeated between neighbouring chunks. Without it, an
#                idea that sits on the boundary gets cut in two and neither
#                half makes full sense.
class RecursiveSplitter:
    def __init__(self, chunk_size=300, chunk_overlap=50, separators=("\n\n", "\n", " ")):
        assert chunk_overlap < chunk_size  # otherwise chunks would never move forward
        self.size, self.overlap, self.seps = chunk_size, chunk_overlap, separators

    def split(self, text, seps=None):
        seps = self.seps if seps is None else seps

        # Small enough already: this is one chunk, nothing to do.
        if count_tokens(text) <= self.size:
            return [text]

        # Cut on the current (biggest) separator.
        sep = seps[0]
        parts = [p for p in text.split(sep) if p.strip()]

        # This separator is not in the text at all, so try the next smaller one.
        if len(parts) == 1 and len(seps) > 1:
            return self.split(text, seps[1:])

        # Any part that is still too big gets split again with smaller separators.
        pieces = []
        for part in parts:
            if count_tokens(part) > self.size and len(seps) > 1:
                pieces += self.split(part, seps[1:])
            else:
                pieces.append(part)

        # Now glue small pieces back together into chunks close to chunk_size.
        return self._merge(pieces, sep)

    def _merge(self, pieces, sep):
        chunks, current, total = [], [], 0
        for piece in pieces:
            n = count_tokens(piece)

            # Adding this piece would overflow, so close the current chunk.
            if current and total + n > self.size:
                chunks.append(sep.join(current))

                # Drop pieces from the front until at most `overlap` tokens remain
                # (and the new piece fits). What remains is the overlap: it becomes
                # the beginning of the next chunk.
                while current and (total > self.overlap or total + n > self.size):
                    total -= count_tokens(current.pop(0))

            current.append(piece)
            total += n

        if current:
            chunks.append(sep.join(current))  # last chunk
        return chunks

    # Split every document and remember which document each chunk came from.
    def split_docs(self, docs):
        return [{"text": c, "source": d["source"]} for d in docs for c in self.split(d["text"])]


# STEP 5: EMBED EVERY CHUNK AND STORE THE VECTORS
# The "vector store" is the finished index: one row of numbers per chunk, plus
# the original chunk text so we can hand it to the LLM later.
# Real stores (Chroma, FAISS, Pinecone) do the same thing but save to disk and
# search millions of vectors fast. Searching is Part 3 (Retrieval), not here.
class VectorStore:
    def __init__(self):
        self.chunks = []                  # chunk text + source, in order
        self.matrix = np.zeros((0, DIM))  # row i is the vector for chunks[i]

    def add(self, chunks):
        vectors = np.array([embed(c["text"]) for c in chunks])
        self.matrix = np.vstack([self.matrix, vectors])
        self.chunks += chunks


# The whole indexing pipeline in one call: split the docs, embed, store.
def build_index(docs, chunk_size=300, chunk_overlap=50):
    store = VectorStore()
    store.add(RecursiveSplitter(chunk_size, chunk_overlap).split_docs(docs))
    return store



# DEMO + SELF-CHECK  (run: python3 indexing.py)
# Walks through each step on a tiny example and prints what happens,
# then asserts the results so you know the code really works.
if __name__ == "__main__":
    # ---- Step 1 demo ----
    print("STEP 1 - tokens")
    print("  'My favorite pet is a cat.' ->", count_tokens("My favorite pet is a cat."), "tokens\n")

    # ---- Step 2 demo ----
    print("STEP 2 - embeddings + cosine similarity")
    question = "What kinds of pets do I like?"
    related = "My favorite pet is a cat. I like pets"
    unrelated = "Stock markets fell sharply today"
    print("  vector length:", len(embed(question)), "numbers")
    s_rel = cosine_similarity(embed(question), embed(related))
    s_unrel = cosine_similarity(embed(question), embed(unrelated))
    print(f"  question vs related text   : {s_rel:.2f}")
    print(f"  question vs unrelated text : {s_unrel:.2f}\n")
    assert s_rel > s_unrel  # related text must score higher

    # ---- Steps 3-5 demo: a fake document with 3 topics, 4 paragraphs each ----
    topics = {
        "agents": "Agents use memory and planning to decompose tasks into steps.",
        "cooking": "Boil the pasta in salted water then add the tomato sauce.",
        "space": "The rocket reached orbit and deployed the satellite above Earth.",
    }
    text = "\n\n".join(f"{t} " * 6 for t in topics.values() for _ in range(4))
    docs = [{"text": text, "source": "demo"}]
    print("STEP 3 - loaded 1 doc,", count_tokens(text), "tokens total\n")

    store = build_index(docs, chunk_size=40, chunk_overlap=10)
    sizes = [count_tokens(c["text"]) for c in store.chunks]
    print("STEP 4 - split into", len(store.chunks), "chunks, token sizes:", sizes)
    print("STEP 5 - matrix shape:", store.matrix.shape, "(chunks x vector length)\n")

    assert len(store.chunks) > 1                            # the doc really got split
    assert max(sizes) <= 40                                 # no chunk exceeds chunk_size
    assert store.matrix.shape == (len(store.chunks), DIM)   # one vector per chunk

    # Overlap check: the end of chunk 0 should reappear at the start of chunk 1.
    a, b = store.chunks[0]["text"].split(), store.chunks[1]["text"].split()
    assert any(a[-k:] == b[:k] for k in range(1, 11))
    print("overlap: chunk 0 ends with the same words chunk 1 starts with")

    # Index sanity check: the chunk closest to this query should be about rockets.
    scores = store.matrix @ embed("rocket satellite orbit")
    best = store.chunks[int(np.argmax(scores))]
    assert "rocket" in best["text"]
    print("closest chunk to 'rocket satellite orbit':", best["text"][:50], "...")
    print("\nok: indexing pass")