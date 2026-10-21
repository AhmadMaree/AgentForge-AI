
from pathlib import Path

from qdrant_client import QdrantClient, models

class QDrantVectorStore:

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        path: str | None = None,
        client: QdrantClient | None = None
    ):
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")

        self.collection_name = collection_name
        self.vector_size = vector_size

        if client is not None:
            self.client = client
        else:
            if path is None:
                raise ValueError("Provide path or client")

            Path(path).mkdir(
                parents=True,
                exist_ok=True
            )

            self.client = QdrantClient(
                path=path
            )


        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size= vector_size,
                    distance=models.Distance.COSINE
                )
            )
        else:
            collection = self.client.get_collection(
                collection_name=collection_name
            )

            params = (
                collection.config.params.vectors
            )

            if(
                not isinstance(
                    params,
                    models.VectorParams
                )
                or params.size != vector_size
            ):
                raise ValueError("Existing collection has an incomplete vector configuration")


    def add(
        self,
        meomery_id: str,
        vector: list[float],
        user_id: str, 
    ) -> None:

        if len(vector) != self.vector_size:
            raise ValueError("Embedding dimention mismatch")

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id= meomery_id,
                    vector=vector,
                    payload= {
                        "user_id": user_id
                    }
                )
            ],
            wait=True
        )

    def serach(
        self,
        vector: list[float],
        user_id: str,
        limit: int = 5
    ) -> list[tuple[str,float]]:

        if len(vector) != self.vector_size:
            raise ValueError("Embedding dimention mismatch")

        if limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(
                            value=user_id
                        )
                    )
                ]
            ),
            limit=limit,
            with_payload=False
        )

        return [
            (str(point.id), float(point.score))
            for point in response.points
        ]


    def close(self) -> None:
        self.client.close()

    def remove(
        self,
        memory_id: str,
    ) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[memory_id]
            ),
            wait=True,
        )

        
