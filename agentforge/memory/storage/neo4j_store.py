from neo4j import GraphDatabase


from agentforge.memory.graph_models import (
    Entity,
    GraphFact,
    Relation,
    entity_key,
)


class Neo4jGraphStore:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
    ):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )
        ## check connction
        self.driver.verify_connectivity()

        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT entity_key_unique
                IF NOT EXISTS
                FOR (e:Entity)
                REQUIRE e.key IS UNIQUE
            """)


    def add_knowledge(
        self,
        user_id: str,
        memory_id: str,
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        entity_rows = [
            {
                "key": entity_key(user_id, e.name),
                "name": e.name,
                "type": e.entity_type,
            }
            for e in entities
        ]

        relation_rows = [
            {
                "source": entity_key(user_id, r.source),
                "target": entity_key(user_id, r.target),
                "relation": r.relation,
            }
            for r in relations
        ]

        def write(tx):
            tx.run("""
                UNWIND $rows AS row
                MERGE (e:Entity {key: row.key})
                SET e.name = row.name,
                    e.entity_type = row.type,
                    e.user_id = $user_id
            """, rows=entity_rows, user_id=user_id).consume()

            tx.run("""
                UNWIND $rows AS row
                MATCH (a:Entity {
                    key: row.source,
                    user_id: $user_id
                })
                MATCH (b:Entity {
                    key: row.target,
                    user_id: $user_id
                })
                MERGE (a)-[r:FACT {
                    user_id: $user_id,
                    memory_id: $memory_id,
                    relation: row.relation
                }]->(b)
            """,
                rows=relation_rows,
                user_id=user_id,
                memory_id=memory_id,
            ).consume()

        with self.driver.session() as session:
            session.execute_write(write)

    def search(
        self,
        user_id: str,
        names: list[str],
        limit: int = 20,
    ) -> list[GraphFact]:
        if not names or limit <= 0:
            return []

        keys = [
            entity_key(user_id, name)
            for name in names
        ]

        with self.driver.session() as session:
            records = session.run("""
                MATCH (a:Entity)-[r:FACT]->(b:Entity)
                WHERE r.user_id = $user_id
                  AND (
                    a.key IN $keys
                    OR b.key IN $keys
                  )
                RETURN
                    r.memory_id AS memory_id,
                    a.name AS source,
                    r.relation AS relation,
                    b.name AS target
                ORDER BY r.memory_id
                LIMIT $limit
            """,
                user_id=user_id,
                keys=keys,
                limit=limit,
            )

            return [
                GraphFact(**record.data())
                for record in records
            ]

    def remove_memory(
        self,
        user_id: str,
        memory_id: str,
    ) -> None:
        with self.driver.session() as session:
            session.run("""
                MATCH ()-[r:FACT {
                    user_id: $user_id,
                    memory_id: $memory_id
                }]->()
                DELETE r
            """,
                user_id=user_id,
                memory_id=memory_id,
            ).consume()

    def close(self) -> None:
        self.driver.close()