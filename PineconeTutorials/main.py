"""
Async RAG pipeline:
  1. Chunk (sync — CPU)
  2. Embed (async — Gemini)
  3. Get index (async — Pinecone)
  4. Upsert (async — Pinecone)
  5. Retrieve top-3 (async — Pinecone)
  6. Generate answer (async — Gemini LLM)
"""

import asyncio

from Configuration.chunking import process_document
from Configuration.embeddings import embedBatch
from Configuration.creatingIndex import getIndex
from Configuration.config import gemini_client , GEMINI_MODEL


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TEST_FILE = "PakHistory.txt"
NAMESPACE = "pakhistory"
WORDS_PER_CHUNK = 300
OVERLAP_WORDS = 45
MIN_WORDS = 40


# ═══════════════════════════════════════════════════════════════
# GENERATE — ask Gemini LLM with the retrieved context
# ═══════════════════════════════════════════════════════════════
async def generate_answer(question: str, contexts: list[str]) -> str:
    """Send question + retrieved context to Gemini; return the answer."""
    context_block = "\n\n---\n\n".join(contexts)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context_block}

Question: {question}

Answer:"""

    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
async def main():
    # # ─── Step 1: Chunk ───
    # print("=" * 60)
    # print("STEP 1: CHUNKING")
    # print("=" * 60)

    # with open(TEST_FILE, "rb") as f:
    #     chunks = process_document(
    #         file_obj=f,
    #         filename=TEST_FILE,
    #         words_per_chunk=WORDS_PER_CHUNK,
    #         overlap_words=OVERLAP_WORDS,
    #         min_words=MIN_WORDS,
    #     )

    # if not chunks:
    #     print("❌ No chunks produced. Exiting.")
    #     return

    # print(f"\n✅ Total chunks: {len(chunks)}")

    # # ─── Step 2: Embed ───
    # print("\n" + "=" * 60)
    # print("STEP 2: EMBEDDING")
    # print("=" * 60)

    # texts = [c["text"] for c in chunks]
    # print(f"🧠 Embedding {len(texts)} chunks with Gemini...")
    # vectors = await embedBatch(texts, task_type="RETRIEVAL_DOCUMENT")
    # print(f"✅ Generated {len(vectors)} embeddings")

    # dimension = len(vectors[0])
    # print(f"📐 Dimensions per vector: {dimension}")

    # # ─── Step 3: Get index ───
    # print("\n" + "=" * 60)
    # print("STEP 3: PINECONE UPSERT")
    # print("=" * 60)
    index = await getIndex(3072)

    # # ─── Step 4: Build records ───
    # records = [
    #     {
    #         "id": f"{chunks[i]['source']}::{chunks[i]['chunk_index']}",
    #         "values": vectors[i],
    #         "metadata": {
    #             "text": chunks[i]["text"],
    #             "source": chunks[i]["source"],
    #             "chunk_index": chunks[i]["chunk_index"],
    #             "word_count": chunks[i]["word_count"],
    #             "total_chunks": chunks[i]["total_chunks"],
    #         },
    #     }
    #     for i in range(len(chunks))
    # ]

    # # ─── Step 5: Upsert ───
    # print(f"⬆️  Upserting {len(records)} records...")
    # await index.upsert(vectors=records, namespace=NAMESPACE)

    # stats = await index.describe_index_stats()
    # print(f"✅ Upsert complete")
    # print(f"   Total vectors in index: {stats['total_vector_count']}")

    # ─── Step 6: Retrieve top-3 ───
    print("\n" + "=" * 60)
    print("STEP 4: RETRIEVE TOP-3")
    print("=" * 60)

    question = "define and explain fast api"
    print(f"❓ {question}")

    q_vec = (await embedBatch([question], task_type="RETRIEVAL_QUERY"))[0]
    results = await index.query(
        vector=q_vec,
        top_k=3,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    matches = results["matches"]
    if not matches:
        print("⚠️  No matches found.")
        return

    contexts = [m["metadata"]["text"] for m in matches]

    print(f"🔍 Retrieved {len(matches)} chunks:")
    for i, m in enumerate(matches, 1):
        print(f"\n  {i}. Score: {m['score']:.4f}")
        print(f"     Source: {m['metadata']['source']} (chunk {m['metadata']['chunk_index']})")
        print(f"     Text:   {m['metadata']['text'][:120]}...")

    # ─── Step 7: Generate answer via Gemini LLM ───
    print("\n" + "=" * 60)
    print("STEP 5: GENERATE ANSWER (Gemini LLM)")
    print("=" * 60)

    answer = await generate_answer(question, contexts)

    print(f"\n💡 Answer:\n{answer}")

  

# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(main())