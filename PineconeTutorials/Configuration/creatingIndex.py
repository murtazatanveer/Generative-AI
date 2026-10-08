
import asyncio
from pinecone import ServerlessSpec

from Configuration.config import pc, INDEX_NAME, CLOUD, REGION


# ═══════════════════════════════════════════════════════════════
# GLOBAL INDEX SINGLETON
# ═══════════════════════════════════════════════════════════════
_index = None
_index_lock: asyncio.Lock | None = None


async def getIndex(dimension: int = 3072):
    
   
    global _index, _index_lock

    if _index is not None:
        return _index

    if _index_lock is None:
        _index_lock = asyncio.Lock()

    async with _index_lock:
        if _index is not None:
            return _index

        index_list = await pc.list_indexes()
        existing = [i.name for i in index_list]

        if INDEX_NAME not in existing:
            print(f"🔨 Creating index '{INDEX_NAME}' ({dimension} dims)...")
            await pc.create_index(
                name=INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            )

            while not (await pc.describe_index(INDEX_NAME)).status["ready"]:
                print("⏳ Waiting for index to be ready...")
                await asyncio.sleep(2)

            print("✅ Index created")
        else:
            print(f"✅ Using existing index: {INDEX_NAME}")

        _index = await pc.index(INDEX_NAME)
        return _index