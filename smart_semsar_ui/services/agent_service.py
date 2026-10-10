"""Future integration boundary for recommendations and alternative search."""


def get_demo_recommendations(properties: list[dict]) -> list[dict]:
    """Demo-only ranking; replace with a matching/ranking service later."""
    return sorted(
        properties,
        key=lambda item: item.get("match_score") or 0,
        reverse=True,
    )
