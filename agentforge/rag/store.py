
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient, models
from agentforge.memory.embedding import BaseEmbedding
from agentforge.rag.models import Chunk, SearchHit

class RAGVectorStore:

    def __init__(
        self,
        client: QdrantClient,
        embedder: BaseEmbedding,
        collection: str = "agentforge_rag"
    ):
        self.client= client
        self.embedder = embedder
        self.collection = collection

        if not client.collection_exists(collection_name=collection):

            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size= embedder.dimension,
                    distance=models.Distance.COSINE
                )
            )

        else :
            config =  client.get_collection(
                 collection_name=collection
            ).config.params.vectors

            if not isinstance(config, models.VectorParams) or config.size != embedder.dimension:
                raise ValueError(
                    "Qdrant collection dimension mismatch"
                )


    @staticmethod
    def point_id(chunk_id: str)-> str:
        return str(uuid5(NAMESPACE_URL, chunk_id))


    def add(
        self,
        user_id:str,
        chunks: list[Chunk],
        namespace: str
    ) -> int:

        if not chunks:
            return 0

        points = []

        for chunk in chunks:
            points.append(
                models.PointStruct(
                    id= self.point_id(f"{user_id}:{namespace}:{chunk.id}"),
                    vector = self.embedder.embed(
                        chunk.text
                    ),
                    payload={
                        "user_id": user_id,
                        "namespace": namespace,
                        "chunk": chunk.model_dump()
                    }
                )
            )


        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True
        )

        return len(points)

    def  search(
        self,
        query: str,
        user_id: str,
        namespace: str,
        limit: int = 10,
    ) -> list[SearchHit]:
        if limit < 0:
            raise ValueError("Limit should be positive")

        if not query.strip():
            return []


        result = self.client.query_points(
            collection_name= self.collection,
            query=self.embedder.embed(query),
            query_filter= models.Filter(
                must = [
                    models.FieldCondition(
                        key = "user_id",
                        match= models.MatchValue(
                            value=user_id
                        )
                    ),
                    models.FieldCondition(
                        key="namespace",
                        match=models.MatchValue(
                            value=namespace
                        )
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )

        hits = []

        for point in result.points:
            payload = point.payload or {}
            chunk = payload.get("chunk")

            if not isinstance(chunk, dict):
                continue

            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=float(point.score),
                    method="dense"
                )
            )

        return hits

    def list_chunks(
        self,
        user_id: str,
        namespace: str,
    ) -> list[Chunk]:
        chunks = []
        offset = None

        conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="namespace",
                    match=models.MatchValue(
                        value=namespace
                    ),
                ),
            ]
        )

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=conditions,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                raw = (point.payload or {}).get("chunk")

                if isinstance(raw, dict):
                    chunks.append(
                        Chunk.model_validate(raw)
                    )

            if offset is None:
                break

        return chunks


