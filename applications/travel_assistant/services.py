
from typing import Protocol

from .models import (
    TripPlanRequest,
    Attraction,
    WeatherInfo,
    Hotel
)

class AttractionProvider(Protocol):
    def search(
        self,
        request: TripPlanRequest
    ) -> list[Attraction]:
        ...

class WeatherProvider(Protocol):
    def forcast(
        self,
        request: TripPlanRequest,
    ) -> list[WeatherInfo]:
        ...

class HotelProvider(Protocol):
    def search(
        self,
        request: TripPlanRequest,
    ) -> list[Hotel]:
        ...



##Mock samples data
class MockTravelProvider:
    def search(
        self,
        requests: TripPlanRequest
    ) -> list[Attraction]:

        return [
            Attraction(
                name="Demo Museum",
                address="Example address",
                description="Example cultural attraction",
                visit_duration=90,
                ticket_price=20,
            ),
            Attraction(
                name="Demo Park",
                address="Example address",
                description="Example outdoor attraction",
                visit_duration=120,
                ticket_price=0,
            )
        ]

    def forcast(
        self,
        requests: TripPlanRequest
    ) -> list[WeatherInfo]:

        from datetime import timedelta

        return [
            WeatherInfo(
                date= requests.start_date + timedelta(days=i),
                day_weather="Unknown",
                night_weather="Unknown"
            )
            for i in range(requests.days) 
        ]


class MockHotelProvider:
     def search(
        self,
        request: TripPlanRequest,
    ) -> list[Hotel]:
        return [
            Hotel(
                name="Demo Hotel",
                address="Example address",
                estimated_cost=80,
            )
        ]
