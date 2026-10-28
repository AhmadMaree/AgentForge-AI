from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator
from enum import Enum
from typing import Optional
from datetime import datetime, timezone

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ResearchStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RESEARCHING = "researching"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"

class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=1000)
    max_tasks: int = Field(default=3, ge=3, le=5)
    max_results_per_task: int = Field(
        default=5, ge=1, le=10
    )

class TodoItem(BaseModel):
    id: int = Field(ge=1)
    title: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    query: str = Field(min_length=1)
    status: TaskStatus = TaskStatus.PENDING

class SearchResult(BaseModel):
    title: str
    url: HttpUrl = ""
    snippet: str = ""
    backend: str = ""
    retrieved_at: datetime = Field(default_factory=utc_now)


class TaskResult(BaseModel):
    task: TodoItem
    sources: list[SearchResult] = Field(default_factory=list)
    summary: str  = ""
    error: Optional[str] = None

class ResearchPlan(BaseModel):
    tasks: list[TodoItem] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def check_ids(self):
        ids = [
            item.id
            for item in self.tasks
        ]

        #check if there a duplicate ids!!
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate task IDs")

        return self


class ResearchState(BaseModel):

    research_id: str = Field(
        default_factory= lambda : uuid4().hex
    )
    research_topic: str = Field(min_length=3)
    status: ResearchStatus = ResearchStatus.PENDING
    todo_list: list[TodoItem] = Field(default_factory=list)
    results: list[TaskResult] = Field(default_factory=list)
    final_report: str = ""
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def progress(self) -> float:

        if not self.todo_list:
            return 0.0

        completedTask = sum(
            task.status == TaskStatus.COMPLETED
            for task in self.todo_list
        )

        return round(
            100 * completedTask / len(self.todo_list), 2
        )



class ResearchEvent(BaseModel):
    type: str
    message: str = ""
    task_id: int | None = None
    progress: float | None = None
    data: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)