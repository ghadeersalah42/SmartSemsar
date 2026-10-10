"""Demo property data access layer.

Replace these demo functions with a database/API implementation later.
Views should call this module instead of embedding data access logic.
"""

DEMO_PROPERTIES = [
    {
        "id": "PROP-001", "title": "Modern Apartment in New Cairo",
        "location": "New Cairo", "price": 4_800_000, "bedrooms": 3,
        "area_sqm": 155, "property_type": "Apartment",
        "finishing": "Fully Finished", "match_score": 96,
        "description": "A bright family apartment close to main services.",
    },
    {
        "id": "PROP-002", "title": "Contemporary Apartment in Mostakbal City",
        "location": "New Cairo", "price": 4_250_000, "bedrooms": 3,
        "area_sqm": 145, "property_type": "Apartment",
        "finishing": "Semi Finished", "match_score": 91,
        "description": "A practical layout with flexible payment options.",
    },
    {
        "id": "PROP-003", "title": "Elegant Duplex in Sheikh Zayed",
        "location": "Sheikh Zayed", "price": 6_200_000, "bedrooms": 4,
        "area_sqm": 210, "property_type": "Duplex",
        "finishing": "Fully Finished", "match_score": 78,
        "description": "Spacious duplex in a quiet residential community.",
    },
    {
        "id": "PROP-004", "title": "Family Apartment in 6th of October",
        "location": "6th of October", "price": 3_600_000, "bedrooms": 3,
        "area_sqm": 160, "property_type": "Apartment",
        "finishing": "Fully Finished", "match_score": 86,
        "description": "Comfortable apartment with nearby schools and shops.",
    },
    {
        "id": "PROP-005", "title": "Premium Villa in New Administrative Capital",
        "location": "New Administrative Capital", "price": 9_500_000,
        "bedrooms": 5, "area_sqm": 320, "property_type": "Villa",
        "finishing": "Semi Finished", "match_score": 65,
        "description": "A premium villa with generous indoor and outdoor space.",
    },
]


def get_properties() -> list[dict]:
    return [item.copy() for item in DEMO_PROPERTIES]


def get_property_by_id(property_id: str) -> dict | None:
    return next(
        (item.copy() for item in DEMO_PROPERTIES if item["id"] == property_id),
        None,
    )


def search_properties(
    location: str = "All locations",
    property_type: str = "All types",
    max_price: float | None = None,
    min_bedrooms: int = 0,
) -> list[dict]:
    results = get_properties()
    if location != "All locations":
        results = [p for p in results if p["location"] == location]
    if property_type != "All types":
        results = [p for p in results if p["property_type"] == property_type]
    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]
    if min_bedrooms:
        results = [p for p in results if p["bedrooms"] >= min_bedrooms]
    return results
