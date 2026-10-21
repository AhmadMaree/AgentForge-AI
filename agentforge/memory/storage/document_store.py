
import sqlite3
import json

from pathlib import Path
from agentforge.memory.base import ItemMemory

class SQLiteDecumentStore:

    def __init__(self, dbPath: str):
        self.dbPath = dbPath

        Path(dbPath).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._intialize()

    def _connect(self) -> sqlite3.Connection:
        connction = sqlite3.connect(
            self.dbPath,
            timeout=30
        )
        connction.row_factory = sqlite3.Row
        connction.execute(
            "PRAGMA busy_timeout = 30000" #sleep 30s while other connction is writing.
        )

        return connction
    def _intialize(self) -> None:

        with self._connect() as connection:
            connection.execute("""
                    CREATE TABLE IF NOT EXISTS memories(
                        id TEXT PRIMARY kEY,
                        user_id Text NOT NULL,
                        session_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        importance REAL NOT NULL,
                        metadata text NOT NULL
                    )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS 
                idx_memories_user_type_time ON memories(
                    user_id,
                    memory_type,
                    timestamp
                )
            """)

    def save(self, item: ItemMemory) -> str:
        session_id = str(item.metadata.get("session_id", "default"))

        with self._connect() as connection:
            connection.execute("""
                INSERT INTO memories(
                    id,
                    user_id,
                    session_id,
                    content,
                    memory_type,
                    timestamp,
                    importance,
                    metadata
                ) 
                VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                item.id,
                item.user_id,
                session_id,
                item.content,
                item.memory_type,
                item.timestamp.isoformat(),
                item.importance,
                json.dumps(item.metadata)
            ))

        return item.id

    def get(
        self,
        memory_id: str,
        user_id: str,
    )-> ItemMemory | None:

        with self._connect() as connction:
            row = connction.execute("""
                SELECT * FROM memories WHERE id=? AND user_id=?
            """,(memory_id,user_id)).fetchone()

        return self._to_item(row) if row else None   

    @staticmethod
    def _to_item(
        row: sqlite3.Row
    ) -> ItemMemory:

        return ItemMemory(
            id = row["id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            timestamp=row["timestamp"],
            metadata=json.loads(
                row["metadata"]
            )
        )

    def delete(self, memory_id: str, user_id: str) -> bool:

        with self._connect() as connctions:
            cursor = connctions.execute("""
                DELETE FROM memories WHERE id=? AND user_id=?
            """,(memory_id,user_id))

            return cursor.rowcount > 0


    def list_memories(
        self,
        user_id: str,
        memory_type: str | None = None,
        session_id: str | None = None,
    ) -> list[ItemMemory]:

        query = """
            SELECT * FROM memories WHERE user_id=?
            """
        paramters = [user_id]

        if memory_type is not None:
            query += " AND memory_type=?"
            paramters.append(memory_type)

        if session_id is not None:
            query += " AND session_id=?"
            paramters.append(session_id)

        query += " ORDER BY timestamp DESC"
        with self._connect() as conn:
            rows = conn.execute(
                query,
                paramters
            ).fetchall()

        return [
            self._to_item(row)
            for row in rows
        ]

        
        

