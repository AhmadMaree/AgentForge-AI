"""FastAPI backend for AgentForge Deep Research."""
import os
import threading
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    ResearchRequest
)

from .orchestrator import (
    DeepResearchAgent
)

from .agents import (
    PlanningService,
    ReportingService,
    SummarizationService
)

from typing import Iterator
from agentforge.core.config import LLMConfig
from agentforge.llm.gemini import GeminiLLM

from .search import SearchService

from .storage import ResearchStorage

load_dotenv()


app = FastAPI(
    title="Deep Research Agent",
    version="0.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"]
)


def buildAgent() -> DeepResearchAgent:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini api key is required")

    llm = GeminiLLM(
        config=LLMConfig(
            provider="gemini",
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            ),
            temperature=0.2,
            max_output_token=4096,
        ),
        api_key=api_key,
    )

    return DeepResearchAgent(
        planner=PlanningService(llm),
        summarizer=SummarizationService(llm),
        reporter=ReportingService(llm),
        search=SearchService(),
        storage=ResearchStorage()
    )   



@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "deep-research",
    }


@app.post("/research/stream")

def research_stream(request: ResearchRequest):

    def event_generator() -> Iterator[str]:
        try:
            agent = buildAgent()

            for event, state in agent.run_stream(request):
                payload = event.model_dump(
                    mode="json"
                )
                yield (
                    "data: "
                    + json.dumps(payload, ensure_ascii=False)
                    + "\n\n"
                )
        except Exception as exc:
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": str(exc),
                    }
                )
                + "\n\n"
            )


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/research/{research_id}")
def get_research(research_id: str):
    storage = ResearchStorage()

    try:
        state = storage.load_state(research_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=404,
            detail="Research session not found",
        )

    return state.model_dump(mode="json")


@app.get("/research/{research_id}/report")
def get_report(research_id: str):

    storage = ResearchStorage()

    try :
        state = storage.load_state(research_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=404,
            detail="Research session not found",
        )

    if not state.final_report:
        raise HTTPException(
            status_code=404,
            detail="Report not available",
        )

    return {
        "research_id": research_id,
        "report": state.final_report,
    }


