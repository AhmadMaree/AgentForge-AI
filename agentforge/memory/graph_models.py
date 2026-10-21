import re
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field, model_validator


EntityType = Literal[
    "PERSON", "ORGANIZATION", "PROJECT",
    "TECHNOLOGY", "CONCEPT", "OTHER",
]

RelationType = Literal[
    "USES", "BUILT_BY", "WORKS_AT",
    "PART_OF", "RELATED_TO",
]

def normalize_name(name: str) -> str:
    return " ".join(name.casefold().split())


def entity_key(user_id: str, name: str) -> str:
    raw = f"{user_id}:{normalize_name(name)}"
    return sha256(raw.encode("utf-8")).hexdigest()


class Entity(BaseModel):
    name: str = Field(min_length=1)
    entity_type: EntityType = "OTHER"


class Relation(BaseModel):
    source: str = Field(min_length=1)
    relation: RelationType
    target: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_endpoints(self):
        if normalize_name(self.source) == normalize_name(
            self.target
        ):
            raise ValueError("Self-relations are not supported")
        return self


class ExtractedKnowledge(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    
class GraphFact(BaseModel):
    memory_id: str
    source: str
    relation: RelationType
    target: str