from typing import Optional
from pydantic import BaseModel


class Property(BaseModel):
    id: str
    title: str
    location: str
    price: float
    bedrooms: int
    area_sqm: int
    property_type: str
    finishing: str = "Not specified"
    match_score: Optional[int] = None
    description: str = ""
    image_url: Optional[str] = None
