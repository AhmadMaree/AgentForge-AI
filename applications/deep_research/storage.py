import json
from pathlib import Path

from .models import ResearchState, TaskResult


class ResearchStorage:

    def __init__(self, workspace: str | Path = "workspace/deep_research"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(
            parents=True,
            exist_ok=True
        )

    def session_dir(self, research_id: str)->  Path:
        if (
            len(research_id) != 32
            or not all(
                char in "0123456789abcdef"
                for char in research_id
            )
        ):
            raise ValueError("Invalid research ID")

        folder = self.workspace / research_id
        folder.mkdir(
            parents=True,
            exist_ok=True
        )
        return folder


    def save_state(self, state: ResearchState):
        folder = self.session_dir(state.research_id)
        stateFile = folder / "state.json"
        stateFile.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8"
        )


    def save_tasks(
        self,
        research_id: str,
        result: TaskResult
    ):

        folder = (
            self.session_dir(research_id) / "notes"
        )

        folder.mkdir(
            parents=True, exist_ok=True
        )

        lines = [
            f"# Task {result.task.id}: {result.task.title}",
            "",
            f"**Intent:** {result.task.intent}",
            "",
            f"**Search query:** {result.task.query}",
            "",
            "## Summary",
            "",
            result.summary,
            "",
            "## Sources",
            "",
        ]

        for index, source in enumerate(
            result.sources, start=1
        ):

            lines.extend(
                [
                    f"### [{index}] {source.title}",
                    f"URL: {source.url}",
                    f"Snippet: {source.snippet}",
                    "",
                ]
            )

        if result.error:
            lines.extend(
                ["## Error", "", result.error]
            )

        path =  folder / f"{result.task.id}.md"
        path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    def save_report(self, state: ResearchState):

        folder = (
            self.session_dir(state.research_id)
            / "reports"
        )
        folder.mkdir(
            parents=True, exist_ok=True
        )

        path = folder / "final_report.md"

        path.write_text(
            state.final_report,
            encoding="utf-8",
        )


    def load_state(
        self,
        research_id: str,
    ) -> ResearchState:
        path = (
            self.session_dir(research_id)
            / "state.json"
        )

        return ResearchState.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        

        