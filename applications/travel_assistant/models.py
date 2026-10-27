
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal

from datetime import date

class Location(BaseModel):
    latitude: float = Field(ge=-180, le=180, description="latitude")
    longitude: float = Field(ge=-90, le=90, description="longitude")


class Attraction(BaseModel):
    name: str = Field(...,description="Attraction Name")
    address: str = Field(..., description="Address")
    location: Location | None = None
    visit_duration: int = Field(default=60, gt=0)
    description: str = ""
    category: Optional[str] = "Attraction"
    rating: Optional[float] = Field(default=None, ge=0, le=5, description="Rating")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    ticket_price: float = Field(default=0, ge=0)


class Meal(BaseModel):
    type: Literal["dinner", "breakfast", "lunch", "snack"]
    name: str
    address: Optional[str] = None
    location: Optional[Location] = None
    estimated_cost: float = Field(default=0, ge=0)

class Hotel(BaseModel):
    name: str
    address: Optional[str] = ""
    location: Optional[Location] = None
    rating: float | None = Field(default=None, ge=0, le=5)
    estimated_cost: float = Field(default=0, ge=0)
    price_range: str = Field(default="", description="Price range")
    distance: str = Field(default="", description="Distance to attraction")

class WeatherInfo(BaseModel):
    date: date
    day_weather: str
    night_weather: str = ""
    day_temp: float | None = None
    night_temp: float | None = None

class Budget(BaseModel):
    currency: str = "USD"
    total_attractions: float = Field(default=0, ge=0)
    total_hotels: float = Field(default=0, ge=0)
    total_meals: float = Field(default=0, ge=0)
    total_transportation: float = Field(default=0, ge=0)

    @property
    def total(self) -> float:
        return (
            self.total_attractions
            + self.total_hotels
            + self.total_meals
            + self.total_transportation
        )



class DayPlan(BaseModel):
    date: date
    day_index: int = Field(ge=0)
    description: str = ""
    transportation: str = ""
    hotel: Optional[Hotel] = None
    meal: list[Meal] = Field(default_factory=list)
    attractions: list[Attraction] = Field(default_factory=list)
    accommodation: str = Field(..., description="Accommodation arrangement")


class TripPlanRequest(BaseModel):
    city: str = Field(min_length=1)
    start_date: date
    end_date: date
    preferences: str = ""
    budget: str = "medium"
    transportation: str = "public_transport"
    accommodation: str = "hotel"
    currency: str = "USD"

    @model_validator(mode="after")
    def validate_date(self):
        if self.end_date < self.start_date:
            raise ValueError(
                "End date must be on after start date"
            )
        return self

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1


class TripPlan(BaseModel):
    city: str
    start_date: date
    end_date: date
    days: list[DayPlan] = Field(default_factory=list)
    weather_info: list[WeatherInfo] = Field(default_factory=list)
    overall_suggestions: str = ""
    budget: Budget | None = None



