import streamlit as st
import requests
from PIL import Image
import io
import time
from datetime import datetime
import threading

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="ARCHINZA Search",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for dual-panel layout
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-panel {
        background: #f8f9fa;
        border-left: 3px solid #1E3A5F;
        padding: 1.5rem;
        border-radius: 10px;
        height: 600px;
        overflow-y: auto;
    }
    .chat-message {
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-radius: 8px;
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    .ai-message {
        background: #e3f2fd;
        color: #1565c0;
        margin-right: 2rem;
    }
    .label-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.75rem;
        margin-right: 0.3rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }
    .object-tag {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.75rem;
        margin-right: 0.3rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }
    .result-card {
        border-radius: 10px;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=10)  # Cache for 10 seconds - reduces API calls on page reloads
def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


@st.cache_data(ttl=5)  # Cache for 5 seconds - shows recent updates quickly
def get_search_history(limit=10):
    """Fetch search history from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/history", params={"limit": limit}, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def create_search_session(query_type, text_query, image_filename, user_type, detected_labels, detected_objects, results_count, ai_summary):
    """Create a new search session."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/history/session",
            json={
                "query_type": query_type,
                "text_query": text_query,
                "image_filename": image_filename,
                "user_type": user_type,
                "detected_labels": detected_labels,
                "detected_objects": detected_objects,
                "results_count": results_count,
                "ai_summary": ai_summary
            },
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("session_id")
        return None
    except:
        return None


def create_search_session_async(query_type, text_query, image_filename, user_type, detected_labels, detected_objects, results_count, ai_summary):
    """Create a new search session in background thread (non-blocking)."""
    def _create():
        try:
            session_id = create_search_session(
                query_type, text_query, image_filename, user_type,
                detected_labels, detected_objects, results_count, ai_summary
            )
            if session_id and 'current_session_id' in st.session_state:
                st.session_state.current_session_id = session_id
        except:
            pass

    thread = threading.Thread(target=_create, daemon=True)
    thread.start()


def add_to_search_session(session_id, user_message, ai_response, query_type, detected_labels, detected_objects, results_count):
    """Add a refinement to an existing session."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/history/session/{session_id}/add",
            json={
                "user_message": user_message,
                "ai_response": ai_response,
                "query_type": query_type,
                "detected_labels": detected_labels,
                "detected_objects": detected_objects,
                "results_count": results_count
            },
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def add_to_search_session_async(session_id, user_message, ai_response, query_type, detected_labels, detected_objects, results_count):
    """Add a refinement to an existing session in background thread (non-blocking)."""
    def _add():
        try:
            add_to_search_session(
                session_id, user_message, ai_response, query_type,
                detected_labels, detected_objects, results_count
            )
        except:
            pass

    thread = threading.Thread(target=_add, daemon=True)
    thread.start()


@st.cache_data(ttl=5)  # Cache for 5 seconds - shows recent updates quickly
def get_favorites():
    """Fetch favorites from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/favorites", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def add_favorite(image_id, s3_key, url, labels=None, objects=None):
    """Add an image to favorites."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/favorites",
            json={
                "image_id": image_id,
                "s3_key": s3_key,
                "url": url,
                "labels": labels or [],
                "objects": objects or []
            },
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def remove_favorite(image_id):
    """Remove an image from favorites."""
    try:
        response = requests.delete(f"{API_BASE_URL}/favorites/{image_id}", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_favorite(image_id):
    """Check if an image is favorited."""
    try:
        response = requests.get(f"{API_BASE_URL}/favorites/check/{image_id}", timeout=5)
        if response.status_code == 200:
            return response.json().get("is_favorited", False)
        return False
    except:
        return False


def batch_check_favorites(image_ids):
    """
    Check multiple images for favorite status in one API call.
    This is significantly faster than checking each image individually.

    Args:
        image_ids: List of image IDs to check

    Returns:
        Dictionary mapping image_id to favorite status (True/False)
    """
    if not image_ids:
        return {}

    try:
        response = requests.post(
            f"{API_BASE_URL}/favorites/check-batch",
            json={"image_ids": image_ids},
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("favorites", {})
        return {}
    except:
        return {}


def search_images(image_file=None, text_query=None, user_type="general"):
    """Call the search API."""
    files = {}
    data = {"user_type": user_type}

    if image_file is not None:
        files["image"] = ("image.jpg", image_file, "image/jpeg")

    if text_query:
        data["text_query"] = text_query

    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            files=files if files else None,
            data=data,
            timeout=30
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except Exception as e:
        return None, f"Error: {str(e)}"


def find_similar_images(image_id, top_k=5):
    """
    Find similar images using the fast image_id-based search API.
    This is 20x faster than downloading and re-processing the image.

    Args:
        image_id: The image ID to find similar images for
        top_k: Number of similar images to return

    Returns:
        Tuple of (results, error)
    """
    try:
        # Use the new fast endpoint that searches by image_id directly
        # No download, no CLIP processing - just embedding lookup and search
        response = requests.get(
            f"{API_BASE_URL}/search/similar/{image_id}",
            params={"top_k": top_k},
            timeout=5  # Much faster now, reduced from 20s to 5s
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Search failed: {response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def generate_contextual_suggestions(results):
    """Generate contextual suggestions based on search results."""
    suggestions = []

    detected_labels = results.get("detected_labels", [])
    detected_objects = results.get("detected_objects", [])

    # If objects detected, suggest variations
    if detected_objects:
        primary_object = detected_objects[0] if detected_objects else None
        if primary_object:
            # Suggest color variants
            suggestions.append(f"Show {primary_object} in different colors")
            # Suggest style variants
            suggestions.append(f"Show more modern {primary_object} designs")
            # Suggest material variants if furniture
            if any(word in primary_object.lower() for word in ['sofa', 'chair', 'table', 'bed']):
                suggestions.append(f"Show {primary_object} with leather upholstery")

    # If architectural styles detected, suggest variations
    if detected_labels:
        primary_style = detected_labels[0] if detected_labels else None
        if primary_style:
            # Suggest combining with other elements
            suggestions.append(f"Add natural lighting to {primary_style}")
            suggestions.append(f"Show {primary_style} with open floor plan")

    # Add contextual refinements based on objects
    if detected_objects:
        # If multiple objects, suggest filtering
        if len(detected_objects) > 2:
            suggestions.append(f"Focus only on {detected_objects[0]} and {detected_objects[1]}")

        # Suggest room type filters
        if any(obj in str(detected_objects).lower() for obj in ['sofa', 'coffee table', 'tv']):
            suggestions.append("Show only living room designs")
        elif any(obj in str(detected_objects).lower() for obj in ['bed', 'nightstand']):
            suggestions.append("Show only bedroom designs")
        elif any(obj in str(detected_objects).lower() for obj in ['dining table', 'dining chair']):
            suggestions.append("Show only dining room designs")

    # Fallback to general suggestions if nothing specific
    if not suggestions:
        suggestions = [
            "Show more with natural materials",
            "Add vintage elements",
            "Show minimalist variations"
        ]

    # Return top 3-4 suggestions
    return suggestions[:4]


@st.dialog("✨ More Like This", width="large")
def show_similar_images_modal(image_url, image_id):
    """Modal dialog showing similar images to the selected one. Optimized for speed."""
    st.markdown("### Finding similar images...")

    with st.spinner("🔍 Searching for similar images..."):
        # Fast search using image_id - no download or re-processing needed
        similar_results, error = find_similar_images(image_id, top_k=5)

    if error:
        st.error(f"Failed to find similar images: {error}")
        return

    if similar_results and similar_results.get("results"):
        st.markdown(f"**Found {len(similar_results['results'])} similar images:**")
        st.divider()

        # Display original image on the left
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Selected Image:**")
            st.image(image_url, use_container_width=True)

        with col2:
            st.markdown("**Similar Images:**")
            # Display similar images in a grid
            results = similar_results["results"]
            for i in range(0, len(results), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        result = results[i + j]
                        with col:
                            st.image(result["url"], use_container_width=True)
                            if result.get("labels"):
                                labels_str = ", ".join(result["labels"][:2])
                                st.caption(f"🏷️ {labels_str}")
    else:
        st.info("No similar images found.")


def render_image_grid(results_list, cols_per_row=3):
    """Render image results in a grid layout with favorite and 'More Like This' buttons."""
    if not results_list:
        st.info("No results to display")
        return

    # Initialize favorites cache in session state
    if 'favorites_cache' not in st.session_state:
        st.session_state.favorites_cache = set()

    # REMOVED: Automatic batch favorites check to reduce latency
    # Users can manually favorite/unfavorite images
    # all_image_ids = [result["image_id"] for result in results_list]
    # favorites_status = batch_check_favorites(all_image_ids)

    for i in range(0, len(results_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(results_list):
                result = results_list[i + j]
                image_id = result["image_id"]

                with col:
                    with st.container():
                        try:
                            st.image(result["url"], use_container_width=True)
                        except:
                            st.error("Failed to load image")

                        # Action buttons in two columns
                        btn_col1, btn_col2 = st.columns(2)

                        with btn_col1:
                            # Favorite button with session state only (no API check for better performance)
                            is_fav = image_id in st.session_state.favorites_cache
                            if is_fav:
                                st.session_state.favorites_cache.add(image_id)

                            fav_button_label = "❤️" if is_fav else "🤍"

                            if st.button(fav_button_label, key=f"fav_{image_id}_{i}_{j}", use_container_width=True, help="Save to favorites"):
                                if is_fav:
                                    # Remove from favorites
                                    if remove_favorite(image_id):
                                        if image_id in st.session_state.favorites_cache:
                                            st.session_state.favorites_cache.remove(image_id)
                                        # Update sidebar count
                                        if 'sidebar_favorites_count' in st.session_state and st.session_state.sidebar_favorites_count > 0:
                                            st.session_state.sidebar_favorites_count -= 1
                                        st.toast("💔 Removed from favorites", icon="✅")
                                        st.rerun()
                                else:
                                    # Add to favorites
                                    if add_favorite(
                                        image_id=image_id,
                                        s3_key=result["s3_key"],
                                        url=result["url"],
                                        labels=result.get("labels"),
                                        objects=None
                                    ):
                                        st.session_state.favorites_cache.add(image_id)
                                        # Update sidebar count
                                        if 'sidebar_favorites_count' in st.session_state:
                                            st.session_state.sidebar_favorites_count += 1
                                        st.toast("❤️ Added to favorites!", icon="✅")
                                        st.rerun()

                        with btn_col2:
                            # More Like This button - Fast search by image_id
                            if st.button("🔍", key=f"similar_{image_id}_{i}_{j}", use_container_width=True, help="Find similar images"):
                                show_similar_images_modal(result["url"], image_id)

                        # Labels (Score hidden per user request)
                        if result.get("labels"):
                            labels_str = ", ".join(result["labels"][:3])
                            st.caption(f"🏷️ {labels_str}")


def main():
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'chat_active' not in st.session_state:
        st.session_state.chat_active = False
    if 'current_query' not in st.session_state:
        st.session_state.current_query = None
    if 'last_search_time' not in st.session_state:
        st.session_state.last_search_time = 0
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None

    # Header
    st.markdown('<h1 class="main-header">🏛️ ARCHINZA Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Architectural Image Search with Conversational Refinement</p>', unsafe_allow_html=True)

    # Check API status (cached - only check once per session)
    if 'api_status_checked' not in st.session_state:
        st.session_state.api_status = check_api_health()
        st.session_state.api_status_checked = True

    api_status = st.session_state.api_status

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
            return

        st.divider()

        user_type = st.selectbox(
            "👤 User Type",
            options=["general", "professional", "student", "enthusiast"],
            help="Select your profile for tailored suggestions"
        )

        # Store user type in session state for 'More Like This' feature
        st.session_state.user_type = user_type

        st.divider()

        # Stats (cached - only fetch once per session or when explicitly refreshed)
        if 'index_stats' not in st.session_state:
            try:
                stats = requests.get(f"{API_BASE_URL}/index/stats").json()
                st.session_state.index_stats = stats
            except:
                st.session_state.index_stats = {"total_vectors": 0}

        st.metric("📚 Indexed Images", st.session_state.index_stats.get("total_vectors", 0))

        st.divider()

        # Navigation Links - Performance Optimized
        st.markdown("### 📊 Navigation")
        st.markdown("Access your data on dedicated pages for better performance:")

        # Quick counts (cached - only fetch once per session or after history/favorite operations)
        if 'sidebar_history_count' not in st.session_state:
            try:
                history_data = get_search_history(limit=1)
                st.session_state.sidebar_history_count = history_data.get("total_count", 0) if history_data else 0
            except:
                st.session_state.sidebar_history_count = 0

        history_count = st.session_state.sidebar_history_count

        if 'sidebar_favorites_count' not in st.session_state:
            try:
                favorites_data = get_favorites()
                st.session_state.sidebar_favorites_count = favorites_data.get("total_count", 0) if favorites_data else 0
            except:
                st.session_state.sidebar_favorites_count = 0

        favorites_count = st.session_state.sidebar_favorites_count

        # Navigation info with counts
        st.info(f"""
        📊 **Quick Access:**
        - 🕒 Search History: **{history_count}** sessions
        - ❤️ Favorites: **{favorites_count}** saved images

        👉 Use the sidebar menu above to navigate to these pages
        """)

        st.caption("💡 Moved to separate pages for faster search performance")

        st.divider()

        # About Section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **ARCHINZA Search Pipeline**

            AI-powered architectural image search using:
            - 🤖 CLIP for visual understanding
            - 🔍 FAISS for fast similarity search
            - 💬 GPT-4o-mini for smart summaries

            Optimized for 5K+ images and 500+ searches/day
            """)

    # DUAL PANEL LAYOUT
    if not st.session_state.chat_active:
        # INITIAL STATE: Full-width search
        st.subheader("🔍 Search for Architectural Images")

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload an architectural image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Upload an image to find similar architectural designs"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            text_query = st.text_area(
                "Describe what you're looking for",
                placeholder="e.g., modern minimalist kitchen with marble countertops",
                height=100,
                key="initial_search"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_clicked:
            if not uploaded_file and not text_query:
                st.warning("⚠️ Please upload an image or enter a text query")
            else:
                image_bytes = None
                if uploaded_file:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()

                with st.spinner("🔍 Searching..."):
                    start_time = time.time()
                    results, error = search_images(
                        image_file=image_bytes,
                        text_query=text_query if text_query else None,
                        user_type=user_type
                    )
                    search_time = time.time() - start_time
                    st.session_state.last_search_time = search_time

                if error:
                    st.error(error)
                elif results:
                    st.session_state.search_results = results
                    st.session_state.chat_active = True
                    st.session_state.current_query = text_query or "Image search"

                    # Store original search context for refinements
                    st.session_state.original_image_bytes = image_bytes
                    st.session_state.original_text_query = text_query
                    st.session_state.detected_labels = results.get("detected_labels", [])
                    st.session_state.detected_objects = results.get("detected_objects", [])

                    # Generate contextual suggestions based on detected elements
                    contextual_suggestions = generate_contextual_suggestions(results)

                    # Create a new search session (non-blocking - runs in background)
                    query_type = results.get("query_type", "text_only")
                    create_search_session_async(
                        query_type=query_type,
                        text_query=text_query,
                        image_filename=uploaded_file.name if uploaded_file else None,
                        user_type=user_type,
                        detected_labels=results.get("detected_labels", []),
                        detected_objects=results.get("detected_objects", []),
                        results_count=results.get("total_results", 0),
                        ai_summary=results["summary"]
                    )

                    # Add AI summary as first chat message
                    st.session_state.chat_history = [{
                        "role": "assistant",
                        "content": results["summary"],
                        "suggestions": contextual_suggestions,
                        "timestamp": datetime.now().isoformat()
                    }]
                    # Rerun to immediately show results in dual-panel layout
                    st.rerun()

    else:
        # DUAL PANEL STATE: Left = Results, Right = Chat
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("📸 Search Results")

            # Display results
            if st.session_state.search_results:
                results = st.session_state.search_results

                # Metrics with Search Time
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Results", results["total_results"])
                with col2:
                    st.metric("⚡ Search Time", f"{st.session_state.last_search_time:.2f}s")
                with col3:
                    st.metric("🔄 Query Type", results["query_type"].replace("_", " ").title())
                with col4:
                    if st.button("🔄 New Search"):
                        st.session_state.chat_active = False
                        st.session_state.chat_history = []
                        st.session_state.search_results = None
                        st.session_state.current_session_id = None
                        # Clear search context for fresh start
                        st.session_state.original_image_bytes = None
                        st.session_state.original_text_query = None
                        st.session_state.detected_labels = []
                        st.session_state.detected_objects = []
                        st.rerun()

                # Detected Elements
                if results.get("detected_labels") or results.get("detected_objects"):
                    with st.expander("🏷️ Detected Elements", expanded=False):
                        if results.get("detected_labels"):
                            st.markdown("**Architectural Styles:**")
                            labels_html = " ".join([
                                f'<span class="label-tag">{label}</span>'
                                for label in results["detected_labels"]
                            ])
                            st.markdown(labels_html, unsafe_allow_html=True)

                        if results.get("detected_objects"):
                            st.markdown("<br>**Furniture & Objects:**", unsafe_allow_html=True)
                            objects_html = " ".join([
                                f'<span class="object-tag">{obj}</span>'
                                for obj in results["detected_objects"]
                            ])
                            st.markdown(objects_html, unsafe_allow_html=True)

                st.divider()

                # Image Grid
                results_list = results.get("results", [])
                render_image_grid(results_list, cols_per_row=3)

        with right_col:
            st.subheader("💬 Refine Your Search")
            st.caption("Ask me to adjust your results conversationally")

            # Chat history container
            chat_container = st.container()

            with chat_container:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "assistant":
                        st.markdown(f"""
                        <div class="chat-message ai-message">
                            <strong>🤖 AI Assistant:</strong><br>
                            {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)

                        if msg.get("suggestions"):
                            st.markdown("**💡 Try asking:**")
                            for suggestion in msg["suggestions"]:
                                if st.button(f"→ {suggestion}", key=f"sug_{hash(suggestion)}_{msg['timestamp']}"):
                                    # Trigger search with suggestion only (no re-upload)
                                    with st.spinner("🔍 Refining search..."):
                                        start_time = time.time()

                                        # Search with just the refinement text
                                        # Context is maintained through chat history
                                        results, error = search_images(
                                            text_query=suggestion,
                                            user_type=user_type
                                        )
                                        search_time = time.time() - start_time
                                        st.session_state.last_search_time = search_time

                                    if results:
                                        st.session_state.search_results = results

                                        # Generate contextual suggestions based on new results
                                        contextual_suggestions = generate_contextual_suggestions(results)

                                        # Add refinement to session (non-blocking - runs in background)
                                        if st.session_state.current_session_id:
                                            add_to_search_session_async(
                                                session_id=st.session_state.current_session_id,
                                                user_message=suggestion,
                                                ai_response=results["summary"],
                                                query_type="text_only",
                                                detected_labels=results.get("detected_labels", []),
                                                detected_objects=results.get("detected_objects", []),
                                                results_count=results.get("total_results", 0)
                                            )

                                        st.session_state.chat_history.append({
                                            "role": "user",
                                            "content": suggestion,
                                            "timestamp": datetime.now().isoformat()
                                        })
                                        st.session_state.chat_history.append({
                                            "role": "assistant",
                                            "content": results["summary"],
                                            "suggestions": contextual_suggestions,
                                            "timestamp": datetime.now().isoformat()
                                        })
                                        st.rerun()

                    else:  # user message
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong><br>
                            {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)

            # Chat input at bottom
            st.markdown("---")
            chat_input = st.text_input(
                "💬 Type your refinement",
                placeholder="e.g., 'show sectional sofa in blue' or 'add natural lighting'",
                key="chat_input",
                label_visibility="collapsed"
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                send_clicked = st.button("Send", type="primary", use_container_width=True)
            with col2:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

            if send_clicked and chat_input:
                with st.spinner("🔍 Refining search..."):
                    start_time = time.time()

                    # Search with just the refinement text
                    # Context is maintained through chat history, no re-upload needed
                    results, error = search_images(
                        text_query=chat_input,
                        user_type=user_type
                    )
                    search_time = time.time() - start_time
                    st.session_state.last_search_time = search_time

                if results:
                    st.session_state.search_results = results

                    # Generate contextual suggestions based on new results
                    contextual_suggestions = generate_contextual_suggestions(results)

                    # Add refinement to session (non-blocking - runs in background)
                    if st.session_state.current_session_id:
                        add_to_search_session_async(
                            session_id=st.session_state.current_session_id,
                            user_message=chat_input,
                            ai_response=results["summary"],
                            query_type="text_only",
                            detected_labels=results.get("detected_labels", []),
                            detected_objects=results.get("detected_objects", []),
                            results_count=results.get("total_results", 0)
                        )

                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": chat_input,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": results["summary"],
                        "suggestions": contextual_suggestions,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.rerun()


if __name__ == "__main__":
    main()
