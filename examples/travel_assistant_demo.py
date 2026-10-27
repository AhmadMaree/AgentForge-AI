
import json
import os
from datetime import date
from dotenv import load_dotenv

from agentforge.core.config import LLMConfig
from agentforge.llm.gemini import GeminiLLM

from applications.travel_assistant.agents import (
    AttractiveSeacrhAgent,
    WeatherQueryAgent,
    HotelAgent,
    PlannerAgent,
)
from applications.travel_assistant.services import (
    MockTravelProvider,
    MockHotelProvider,
)
from applications.travel_assistant.orchestrator import (
    TripPlannerAgent,
)
from applications.travel_assistant.models import (
    TripPlanRequest,
)

from agentforge.llm.factory import FactoryLLM


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    llm = GeminiLLM(
        config=LLMConfig(
            provider="gemini",
            model=gemini_model,
            temperature=0.2,
        ),
        api_key=api_key,
    )

    provider = MockTravelProvider()

    planner = TripPlannerAgent(
        attraction_agent=AttractiveSeacrhAgent(provider),
        weather_agent=WeatherQueryAgent(provider),
        hotel_agent=HotelAgent(MockHotelProvider()),
        planner_agent=PlannerAgent(llm),
    )

    request = TripPlanRequest(
        city="Tokyo",
        start_date=date(2026, 11, 10),
        end_date=date(2026, 11, 12),
        preferences="History and culture",
        budget="medium",
        currency="USD",
    )

    result = planner.plan_trip(request)

    print(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
