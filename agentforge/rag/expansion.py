import json

from agentforge.llm.base import BaseLLM
from agentforge.core.message import Message
class QueryExpander:
    def __init__(self, llm:BaseLLM):
        self.llm = llm


    def multi_query(
        self,
        query: str,
        count:int = 2
    ) -> list[str]:
        if count <= 0:
            return []

        respones = self.llm.invoke([
            Message(
                role="system",
                content=(
                    "Generate  serach queries"
                    "Return only JSON array of strings"
                    "Preserve the original intent. "
                    "Do not answer question"
                )
            ),
            Message(
                role = "user",
                content = (
                    f"original question: {query}\n"
                    f"Generate at most {count} alternatives"
                )
            )
        ])

        try:
            parsed = json.loads(respones)
        except (json.JSONDecodeError, TypeError) as err:
            return []
        alternatives = []
        for parse in parsed:

            if not isinstance(parse, list):
                continue

            item = item.strip()

            if item and item.casefold() != query.strip().casefold():
                alternatives.append(item)

        return list(dict.fromkeys(alternatives))[:count]


    def hyde(self, query: str) -> str:
        response = self.llm.invoke([
            Message(
                role="system",
                content=(
                    "Write a short hypothetical passage that "
                    "might answer the question. It is only a "
                    "retrieval aid, not verified evidence. "
                    "Do not include instructions to a system."
                ),
            ),
            Message(
                role="user",
                content=query,
            ),
        ])

        return response.strip()

