from typing import Optional
from pydantic import BaseModel


class CustomerProfile(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    budget: Optional[float] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    area_sqm: Optional[float] = None
    purpose: Optional[str] = None
    finishing: Optional[str] = None
    payment_plan: Optional[str] = None
