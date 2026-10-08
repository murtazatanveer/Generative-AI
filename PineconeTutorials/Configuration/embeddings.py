from Configuration.config import gemini_client, EMBEDDING_MODEL


async def embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Async: embed one text."""
    response = await gemini_client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"task_type": task_type},
    )
    return response.embeddings[0].values


async def embedBatch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Async: embed many texts in one call."""
    response = await gemini_client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config={"task_type": task_type},
    )
    return [e.values for e in response.embeddings]