import streamlit as st
import requests
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Favorites - ARCHINZA",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .favorite-card {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .favorite-card:hover {
        border-color: #ff4b4b;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=5)
def get_favorites():
    """Fetch favorites from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/favorites", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def remove_favorite(image_id):
    """Remove an image from favorites."""
    try:
        response = requests.delete(f"{API_BASE_URL}/favorites/{image_id}", timeout=5)
        return response.status_code == 200
    except:
        return False


def clear_all_favorites():
    """Clear all favorites."""
    try:
        response = requests.delete(f"{API_BASE_URL}/favorites", timeout=5)
        return response.status_code == 200
    except:
        return False


def update_note(image_id, note):
    """Update note for a favorite."""
    try:
        response = requests.put(
            f"{API_BASE_URL}/favorites/{image_id}/note",
            json={"note": note},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


# Header
st.markdown('<div class="main-header">❤️ My Favorites</div>', unsafe_allow_html=True)
st.markdown("Your saved architectural images")

st.divider()

# Load favorites
favorites_data = get_favorites()

if favorites_data and favorites_data.get("favorites"):
    favorites_list = favorites_data["favorites"]
    total_count = favorites_data.get("total_count", len(favorites_list))

    # Stats and Actions Bar
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.metric("💾 Total Favorites", total_count)

    with col2:
        view_mode = st.selectbox("View", ["Grid", "List"], index=0)

    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_clear'):
                if clear_all_favorites():
                    st.success("✅ All favorites cleared!")
                    st.session_state.confirm_clear = False
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm")

    st.divider()

    # Filter
    search_filter = st.text_input("🔍 Filter favorites", placeholder="Search by image ID or labels...")

    # Apply filter
    if search_filter:
        favorites_list = [
            fav for fav in favorites_list
            if search_filter.lower() in fav.get("image_id", "").lower()
            or any(search_filter.lower() in label.lower() for label in fav.get("labels", []))
        ]

    st.caption(f"Showing {len(favorites_list)} favorite(s)")

    if view_mode == "Grid":
        # Grid View
        cols_per_row = 3
        for i in range(0, len(favorites_list), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, col in enumerate(cols):
                if i + j < len(favorites_list):
                    fav = favorites_list[i + j]

                    with col:
                        with st.container():
                            # Image
                            try:
                                st.image(fav["url"], use_container_width=True)
                            except:
                                st.error("Failed to load image")

                            # Image ID
                            st.caption(f"📷 **{fav.get('image_id', 'Unknown')}**")

                            # Labels
                            if fav.get("labels"):
                                labels_str = ", ".join(fav["labels"][:3])
                                if len(fav["labels"]) > 3:
                                    labels_str += f" +{len(fav['labels']) - 3} more"
                                st.caption(f"🏷️ {labels_str}")

                            # Objects
                            if fav.get("objects"):
                                objects_str = ", ".join(fav["objects"][:3])
                                if len(fav["objects"]) > 3:
                                    objects_str += f" +{len(fav['objects']) - 3} more"
                                st.caption(f"📦 {objects_str}")

                            # Timestamp
                            try:
                                dt = datetime.fromisoformat(fav["timestamp"].replace('Z', '+00:00'))
                                formatted_time = dt.strftime("%b %d, %Y")
                                st.caption(f"⏰ Saved: {formatted_time}")
                            except:
                                pass

                            # Actions
                            if st.button("❌ Remove", key=f"remove_grid_{fav['id']}", use_container_width=True):
                                if remove_favorite(fav["image_id"]):
                                    st.success("💔 Removed from favorites")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error("Failed to remove")

    else:
        # List View
        for fav in favorites_list:
            with st.container():
                list_col1, list_col2 = st.columns([1, 3])

                with list_col1:
                    # Thumbnail
                    try:
                        st.image(fav["url"], use_container_width=True)
                    except:
                        st.error("Failed to load")

                with list_col2:
                    # Details
                    st.markdown(f"### 📷 {fav.get('image_id', 'Unknown')}")

                    # Metadata
                    detail_cols = st.columns(3)

                    with detail_cols[0]:
                        if fav.get("labels"):
                            st.caption(f"**🏷️ Labels:**")
                            for label in fav["labels"][:5]:
                                st.caption(f"  • {label}")

                    with detail_cols[1]:
                        if fav.get("objects"):
                            st.caption(f"**📦 Objects:**")
                            for obj in fav["objects"][:5]:
                                st.caption(f"  • {obj}")

                    with detail_cols[2]:
                        try:
                            dt = datetime.fromisoformat(fav["timestamp"].replace('Z', '+00:00'))
                            formatted_time = dt.strftime("%b %d, %Y at %I:%M %p")
                            st.caption(f"**⏰ Saved:**")
                            st.caption(formatted_time)
                        except:
                            pass

                    # Note
                    if fav.get("note"):
                        st.markdown(f"**📝 Note:** {fav['note']}")

                    # Actions
                    action_col1, action_col2 = st.columns([4, 1])

                    with action_col2:
                        if st.button("❌ Remove", key=f"remove_list_{fav['id']}", use_container_width=True):
                            if remove_favorite(fav["image_id"]):
                                st.success("💔 Removed from favorites")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error("Failed to remove")

                st.divider()

else:
    st.info("💔 No favorites yet. Start saving images from your search results!")

    # Show some tips
    with st.expander("💡 How to add favorites"):
        st.markdown("""
        1. Search for images on the main search page
        2. Click the **🤍 Save** button below any image
        3. The button will change to **❤️ Saved**
        4. Your favorites will appear here
        """)

# Navigation hint
st.divider()
st.info("👈 Use the sidebar menu to navigate back to the main search page")
