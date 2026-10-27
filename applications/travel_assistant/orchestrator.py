
from concurrent.futures import ThreadPoolExecutor

from .agents import (
    AttractiveSeacrhAgent,
    WeatherQueryAgent,
    HotelAgent,
    PlannerAgent,
)

from .models import (
    TripPlanRequest,
    TripPlan
)

class TripPlannerAgent:
    def __init__(
        self,
        attraction_agent: AttractiveSeacrhAgent,
        weather_agent: WeatherQueryAgent,
        hotel_agent: HotelAgent,
        planner_agent: PlannerAgent
    ):
        self.attraction_agent = attraction_agent
        self.weather_agent = weather_agent
        self.hotel_agent = hotel_agent
        self.planner_agent = planner_agent

    def plan_trip(
        self,
        request: TripPlanRequest 
    ) -> TripPlan :
        with ThreadPoolExecutor(max_workers=3) as pool:
            attraction_future = pool.submit(
                self.attraction_agent.run,
                request
            )

            weather_future = pool.submit(
                self.weather_agent.run,
                request
            )

            hotel_future = pool.submit(
                self.hotel_agent.run,
                request
            )

            weathers = weather_future.result()
            attration = attraction_future.result()
            hotel = hotel_future.result()

        return self.planner_agent.run(
            request=request,
            attractions=attration,
            hotels=hotel,
            weather=weathers
        )
        