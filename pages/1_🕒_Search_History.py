import streamlit as st
import requests
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Search History - ARCHINZA",
    page_icon="🕒",
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
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=5)
def get_search_history(limit=50):
    """Fetch search history from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/history", params={"limit": limit}, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


@st.cache_data(ttl=10)
def get_history_stats():
    """Fetch history statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/history/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def delete_session(session_id):
    """Delete a search session."""
    try:
        response = requests.delete(f"{API_BASE_URL}/history/{session_id}", timeout=5)
        return response.status_code == 200
    except:
        return False


def clear_all_history():
    """Clear all search history."""
    try:
        response = requests.delete(f"{API_BASE_URL}/history", timeout=5)
        return response.status_code == 200
    except:
        return False


# Header
st.markdown('<div class="main-header">🕒 Search History</div>', unsafe_allow_html=True)
st.markdown("View and manage your past search sessions")

st.divider()

# Stats Section
stats = get_history_stats()
if stats:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Searches", stats.get("total_searches", 0))

    with col2:
        st.metric("🖼️ Image Searches", stats.get("image_only_searches", 0))

    with col3:
        st.metric("📝 Text Searches", stats.get("text_only_searches", 0))

    with col4:
        st.metric("🔀 Combined Searches", stats.get("combined_searches", 0))

st.divider()

# Filter and Actions Bar
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    search_filter = st.text_input("🔍 Filter sessions", placeholder="Search by title or query...")

with col2:
    limit = st.selectbox("Show", [10, 20, 50, 100], index=1)

with col3:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
        if clear_all_history():
            st.success("✅ All history cleared!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("❌ Failed to clear history")

st.divider()

# History Display
history_data = get_search_history(limit=limit)

if history_data and history_data.get("history"):
    sessions = history_data["history"]

    # Apply filter
    if search_filter:
        sessions = [
            s for s in sessions
            if search_filter.lower() in s.get("title", "").lower()
            or search_filter.lower() in s.get("initial_text_query", "").lower()
        ]

    st.caption(f"Showing {len(sessions)} session(s)")

    for session in sessions:
        session_id = session.get("id")
        title = session.get("title", "Untitled Session")
        timestamp = session.get("timestamp", "")
        last_updated = session.get("last_updated", timestamp)
        conversation_count = len(session.get("conversation", []))
        total_results = session.get("total_results_returned", 0)
        query_type = session.get("initial_query_type", "unknown")

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime("%b %d, %Y at %I:%M %p")
        except:
            formatted_time = timestamp[:19] if len(timestamp) > 19 else timestamp

        # Session Card
        with st.expander(f"💬 **{title}**", expanded=False):
            # Session Metadata
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

            with meta_col1:
                st.metric("⏰ Started", formatted_time)

            with meta_col2:
                st.metric("💬 Messages", conversation_count)

            with meta_col3:
                st.metric("📊 Total Results", total_results)

            with meta_col4:
                query_type_display = query_type.replace("_", " ").title()
                st.metric("🔍 Query Type", query_type_display)

            st.divider()

            # Initial Query Info
            if session.get("initial_text_query"):
                st.markdown(f"**📝 Initial Query:** {session['initial_text_query']}")

            if session.get("initial_image_filename"):
                st.markdown(f"**🖼️ Image:** {session['initial_image_filename']}")

            st.divider()

            # Conversation Thread
            st.markdown("### 💬 Conversation")

            for msg in session.get("conversation", []):
                role = msg.get("role")
                content = msg.get("content")
                msg_time = msg.get("timestamp", "")

                if role == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(content)

                        # Show detected labels/objects if available
                        labels = msg.get("detected_labels", [])
                        objects = msg.get("detected_objects", [])
                        results_count = msg.get("results_count")

                        if labels:
                            st.caption(f"🏷️ Labels: {', '.join(labels[:5])}")
                        if objects:
                            st.caption(f"📦 Objects: {', '.join(objects[:5])}")
                        if results_count is not None:
                            st.caption(f"📊 Results: {results_count}")
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(content)

            st.divider()

            # Actions
            action_col1, action_col2 = st.columns([3, 1])

            with action_col2:
                if st.button("🗑️ Delete Session", key=f"delete_{session_id}", use_container_width=True):
                    if delete_session(session_id):
                        st.success("✅ Session deleted!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete session")

else:
    st.info("📭 No search history yet. Start searching to build your history!")

# Navigation hint
st.divider()
st.info("👈 Use the sidebar menu to navigate back to the main search page")
