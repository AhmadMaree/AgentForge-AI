
from collections.abc import Iterator

from .agents import (
    PlanningService,
    ReportingService,
    SummarizationService
)

from .models import (
    ResearchEvent,
    ResearchRequest,
    ResearchState,
    ResearchStatus,
    TaskResult,
    TaskStatus,
)

from .search import SearchService
from .storage import ResearchStorage

class DeepResearchAgent:

    def __init__(
        self,
        planner: PlanningService,
        summarizer: SummarizationService,
        reporter: ReportingService,
        search: SearchService,
        storage: ResearchStorage
    ):
        self.planner = planner
        self.summarizer = summarizer
        self.reporter = reporter
        self.search = search
        self.storage = storage

    def run(
        self,
        requests: ResearchRequest
    ) -> ResearchState:

        state = None

        for event, curr_state in self.run_stream(
            requests
        ):
            state = curr_state


        if state is None:
            raise RuntimeError(
                "Research produced no state"
            )

        return state



    def run_stream(
        self,
        request: ResearchRequest,
    ) -> Iterator[tuple[ResearchEvent, ResearchState]]:

        state = ResearchState(
            research_topic= request.topic
        )

        try:
            #PLAANING
            state.status = ResearchStatus.PLANNING
            self.storage.save_state(state)

            yield ResearchEvent(
                type="status",
                message="Planning resarch task..."
            ), state

            tasks = self.planner.plan(
                topic=request.topic,
                max_tasks=request.max_tasks
            )

            state.todo_list = tasks
            self.storage.save_state(state)

            yield ResearchEvent(
                type="task",
                message="Research plan created",
                data= {
                    "tasks": [
                        task.model_dump(mode="json")
                        for task in tasks
                    ]
                }
            ), state

            # PHASE 2: RESEARCH
            state.status = ResearchStatus.RESEARCHING

            for task in tasks:
                task.status = TaskStatus.RUNNING
                self.storage.save_state(state)

                yield ResearchEvent(
                    type="status",
                    task_id=task.id,
                    message=f"Searching: {task.title}",
                    progress=state.progress,
                ), state

                try:
                    sources = self.search.search(
                        task.query,
                        request.max_results_per_task
                    )

                    yield ResearchEvent(
                        type="sources",
                        task_id=task.id,
                        message=(
                            f"Found {len(sources)} sources"
                        ),
                        data={
                            "sources": [
                                source.model_dump(mode="json")
                                for source in sources
                            ]
                        },
                    ), state


                    yield ResearchEvent(
                        type="status",
                        task_id=task.id,
                        message=f"Summarizing: {task.title}",
                    ), state

                    summary = self.summarizer.summarize(
                        task,
                        sources
                    )

                    task.status = TaskStatus.COMPLETED

                    result =  TaskResult(
                        task=task,
                        sources= sources,
                        summary=summary,
                    )
                except Exception as exc:
                    task.status = TaskStatus.FAILED

                    result = TaskResult(
                        task=task.model_copy(deep=True),
                        error=str(exc),
                        summary=(
                            "This task could not be completed."
                        ),
                    )

                state.results.append(result)
                self.storage.save_tasks(
                    state.research_id,
                    result
                )
                self.storage.save_state(state)

                yield ResearchEvent(
                    type="task",
                    task_id=task.id,
                    message=(
                        f"Task {task.id}: {task.status.value}"
                    ),
                    progress=state.progress,
                    data={
                        "task": task.model_dump(mode="json"),
                        "summary": result.summary,
                        "error": result.error,
                    },
                ), state

            # PHASE 3: REPORT
            state.status = ResearchStatus.REPORTING
            self.storage.save_state(state)

            yield ResearchEvent(
                type="status",
                message="Generating final report...",
                progress=state.progress,
            ), state

            state.final_report = self.reporter.generate(
                request.topic,
                state.results
            )

            state.status = ResearchStatus.COMPLETED
            self.storage.save_report(state)
            self.storage.save_state(state)

            yield ResearchEvent(
                type="report",
                message="Research completed",
                progress=100,
                data={
                    "research_id": state.research_id,
                    "report": state.final_report,
                },
            ), state
        except Exception as exc:
            state.status = ResearchStatus.FAILED
            state.error = str(exc)
            self.storage.save_state(state)


            yield ResearchEvent(
                type="error",
                message=str(exc),
                data={
                    "research_id": state.research_id
                },
            ), state






