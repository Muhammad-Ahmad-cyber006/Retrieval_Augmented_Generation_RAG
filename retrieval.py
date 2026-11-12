# =============================================================================
# RAG PART 2: RETRIEVAL
# =============================================================================
# Goal of retrieval: given a question, find the few chunks in the index that
# are most likely to contain the answer.
#
# The idea is simple. At indexing time every chunk became a vector. Now we:
#
#   question -> [embed it the same way] -> [score every chunk] -> [keep top k]
#
# Chunks whose vectors point in nearly the same direction as the question's
# vector are "close" in meaning, so they are the best candidates to give the
# LLM in the next step (Generation).
#
# This file reuses the indexing code from indexing.py (same folder).
# shortcut: we compare the question against EVERY chunk (brute force). That is
# fine for thousands of chunks. Real vector databases (FAISS, Chroma) use
# approximate search so it stays fast with millions.
# =============================================================================
import numpy as np

from indexing import build_index, cosine_similarity, embed


# =============================================================================
# STEP 1: SCORE EVERY CHUNK AGAINST THE QUESTION
# =============================================================================
# Critical rule: the question must be embedded with the SAME function used for
# the chunks. Vectors from different models live in different "spaces" and
# cannot be compared.
#
# One similarity number per chunk comes out. Higher = more related.
def score_chunks(store, question):
    q = embed(question)
    return np.array([cosine_similarity(q, row) for row in store.matrix])


# =============================================================================
# STEP 2: KEEP THE TOP k
# =============================================================================
# k is how many chunks to return. It is a trade-off:
#   small k -> short, focused context, but may miss part of the answer
#   large k -> more coverage, but more noise, and the LLM gets more to read
# We sort scores from best to worst and keep the first k.
def retrieve(store, question, k=3):
    scores = score_chunks(store, question)
    best = np.argsort(scores)[::-1][:k]  # argsort is ascending, so reverse it
    return [{**store.chunks[i], "score": float(scores[i])} for i in best]


# =============================================================================
# STEP 3: WRAP IT AS A RETRIEVER OBJECT
# =============================================================================
# Same idea as LangChain's `vectorstore.as_retriever(search_kwargs={"k": 1})`
# in the notebook: remember the store and k, so callers only pass a question.
class Retriever:
    def __init__(self, store, k=3):
        self.store, self.k = store, k

    def get_relevant_documents(self, question):
        return retrieve(self.store, question, self.k)


# =============================================================================
# DEMO + SELF-CHECK  (run: python3 retrieval.py)
# =============================================================================
if __name__ == "__main__":
    # Build a small index first (this is Part 1, Indexing, from indexing.py).
    # Each paragraph is its own chunk because it is under the size limit.
    paragraphs = [
        "Task decomposition breaks a hard task into smaller steps so an agent can plan.",
        "Agents keep memory of past actions and use planning to decide the next step.",
        "Boil the pasta in salted water, then add the tomato sauce and basil.",
        "The rocket reached orbit and deployed the satellite above the Earth.",
    ]
    docs = [{"text": "\n\n".join(paragraphs), "source": "demo"}]
    store = build_index(docs, chunk_size=15, chunk_overlap=0)
    print("index has", len(store.chunks), "chunks\n")

    question = "What is task decomposition?"

    # ---- k = 1: only the single best chunk ----
    top1 = Retriever(store, k=1).get_relevant_documents(question)
    print("k=1 ->", top1[0]["text"][:55], f"(score {top1[0]['score']:.2f})")
    assert "decomposition" in top1[0]["text"]  # best chunk answers the question

    # ---- k = 3: best three, ranked ----
    top3 = Retriever(store, k=3).get_relevant_documents(question)
    print("\nk=3, ranked best to worst:")
    for r in top3:
        print(f"  {r['score']:.2f}  {r['text'][:55]}")
    scores = [r["score"] for r in top3]
    assert scores == sorted(scores, reverse=True)  # ranking is best first
    assert top3[0]["text"] == top1[0]["text"]      # larger k keeps the same winner

    # ---- an off-topic question lands on a different chunk ----
    other = retrieve(store, "how do I cook pasta with tomato sauce", k=1)[0]
    assert "pasta" in other["text"]
    print("\npasta question ->", other["text"][:55])

    # ---- k bigger than the number of chunks just returns everything ----
    assert len(retrieve(store, question, k=99)) == len(store.chunks)
    print("\nok: retrieval pass")