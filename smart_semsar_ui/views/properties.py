import streamlit as st
from components.cards import page_header
from services.property_service import search_properties


def render_properties() -> None:
    page_header("Property catalogue", "Explore Properties", "Search the demo catalogue by location, type, budget, and bedroom count.")

    locations = ["All locations", "New Cairo", "New Administrative Capital", "Sheikh Zayed", "6th of October"]
    types = ["All types", "Apartment", "Villa", "Duplex"]
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        location = c1.selectbox("Location", locations)
        property_type = c2.selectbox("Property type", types)
        max_price = c3.number_input("Maximum price (EGP)", min_value=0, value=10_000_000, step=250_000)
        bedrooms = c4.number_input("Min. bedrooms", min_value=0, max_value=10, value=0)
    results = search_properties(location, property_type, max_price or None, bedrooms)
    st.caption(f"{len(results)} demo properties found")
    if not results:
        st.warning("No properties match these filters. Try adjusting your search.")
        return
    columns = st.columns(2)
    for index, item in enumerate(results):
        with columns[index % 2]:
            with st.container(border=True):
                st.caption(item["id"])
                st.subheader(item["title"])
                st.write(f"📍 {item['location']}")
                st.metric("Price", f"EGP {item['price']:,.0f}")
                st.write(f"🛏️ {item['bedrooms']} bedrooms  ·  📐 {item['area_sqm']} m²")
                st.write(f"Finishing: {item['finishing']}")
                st.write(item["description"])
                c1, c2 = st.columns(2)
                if c1.button("View details", key=f"details_{item['id']}", use_container_width=True):
                    st.session_state["selected_property_id"] = item["id"]
                    st.session_state["current_page"] = "Recommendations"
                    st.rerun()
                is_favorite = item["id"] in st.session_state["favorite_property_ids"]
                if c2.button("♥ Saved" if is_favorite else "♡ Save", key=f"fav_{item['id']}", use_container_width=True):
                    favorites = st.session_state["favorite_property_ids"]
                    if is_favorite:
                        favorites.remove(item["id"])
                    else:
                        favorites.append(item["id"])
                    st.rerun()
