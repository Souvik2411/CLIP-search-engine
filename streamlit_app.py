import streamlit as st
import requests
from PIL import Image
import io
import time

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="ARCHINZA Search",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .result-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .score-badge {
        background: #1E3A5F;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
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
    .summary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .suggestion-item {
        background: #f0f7ff;
        padding: 0.75rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #1E3A5F;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_search_history(limit=10):
    """Fetch search history from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/history", params={"limit": limit}, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


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


def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ ARCHINZA Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Architectural Image Search</p>', unsafe_allow_html=True)

    # Check API status
    api_status = check_api_health()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Status
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
            st.info("Start the API with:\n```\nuvicorn app.main:app --port 8000\n```")

        st.divider()

        # User Type Selection
        user_type = st.selectbox(
            "👤 User Type",
            options=["general", "professional", "student", "enthusiast"],
            help="Select your profile for tailored suggestions"
        )

        st.divider()

        # Stats
        if api_status:
            try:
                stats = requests.get(f"{API_BASE_URL}/index/stats").json()
                st.metric("📚 Indexed Images", stats.get("total_vectors", 0))
            except:
                pass

        st.divider()

        # Search History
        st.markdown("### 🕒 Recent Searches")
        if api_status:
            history_data = get_search_history(limit=5)
            if history_data and history_data.get("history"):
                for item in history_data["history"]:
                    with st.expander(f"🔍 {item['query_type'].replace('_', ' ').title()}", expanded=False):
                        st.caption(f"⏰ {item['timestamp'][:19]}")

                        if item.get("text_query"):
                            st.markdown(f"**Query:** {item['text_query']}")

                        if item.get("image_filename"):
                            st.markdown(f"**Image:** {item['image_filename']}")

                        if item.get("detected_labels"):
                            st.markdown(f"**Styles:** {', '.join(item['detected_labels'][:3])}")

                        if item.get("detected_objects"):
                            st.markdown(f"**Objects:** {', '.join(item['detected_objects'][:3])}")

                        st.markdown(f"**Results:** {item['results_count']}")
            else:
                st.info("No search history yet")

        st.divider()

        # Info
        st.markdown("### 📊 Search Modes")
        st.markdown("""
        - **Image Only**: Upload an image to find similar
        - **Text Only**: Describe what you're looking for
        - **Image + Text**: Combine both for refined results
        """, help="Choose your search method")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Image Input")
        uploaded_file = st.file_uploader(
            "Upload an architectural image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image to find similar architectural designs"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("✏️ Text Query")
        text_query = st.text_area(
            "Describe what you're looking for",
            placeholder="e.g., modern minimalist kitchen with marble countertops",
            height=100
        )

        # Search button
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True,
            disabled=not api_status
        )

    # Validation
    if search_clicked:
        if not uploaded_file and not text_query:
            st.warning("⚠️ Please upload an image or enter a text query")
            return

        # Prepare image bytes
        image_bytes = None
        if uploaded_file:
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()

        # Search
        with st.spinner("🔍 Searching..."):
            start_time = time.time()
            results, error = search_images(
                image_file=image_bytes,
                text_query=text_query if text_query else None,
                user_type=user_type
            )
            search_time = time.time() - start_time

        if error:
            st.error(error)
            return

        if not results:
            st.warning("No results found")
            return

        # Display results - PROGRESSIVE LOADING
        st.divider()

        # Metrics (Show immediately)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Results Found", results["total_results"])
        with col2:
            st.metric("⚡ Search Time", f"{search_time:.2f}s")
        with col3:
            st.metric("🔄 Query Type", results["query_type"].replace("_", " ").title())

        # Detected Labels and Objects (Show immediately for image queries)
        if results.get("detected_labels") or results.get("detected_objects"):
            st.subheader("🏷️ Detected Elements")

            # Architectural Styles
            if results.get("detected_labels"):
                st.markdown("**Architectural Styles:**")
                labels_html = " ".join([
                    f'<span class="label-tag">{label}</span>'
                    for label in results["detected_labels"]
                ])
                st.markdown(labels_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # Furniture & Objects
            if results.get("detected_objects"):
                st.markdown("**Furniture & Objects:**")
                objects_html = " ".join([
                    f'<span class="label-tag" style="background: #e8f5e9; color: #2e7d32;">{obj}</span>'
                    for obj in results["detected_objects"]
                ])
                st.markdown(objects_html, unsafe_allow_html=True)

        # AI Summary FIRST - Connect with users before showing results
        st.divider()
        st.subheader("🤖 AI Summary")
        st.markdown(f"""
        <div class="summary-box">
            {results["summary"]}
        </div>
        """, unsafe_allow_html=True)

        # Follow-up Suggestions
        if results.get("follow_up_suggestions"):
            st.subheader("💡 What to Explore Next")
            for i, suggestion in enumerate(results["follow_up_suggestions"], 1):
                st.info(f"**{i}.** {suggestion}")

        # PROGRESSIVE RESULTS: Show top 3 first (AFTER AI Summary)
        st.divider()
        st.subheader("📸 Search Results")
        results_list = results.get("results", [])
        cols_per_row = 3

        if len(results_list) > 0:
            # Create placeholder for progressive loading
            results_container = st.container()

            with results_container:
                # Phase 1: Show top 3 results first (fast)
                st.markdown("**🚀 Top Results**")
                top_results = results_list[:3]

                cols = st.columns(min(3, len(top_results)))
                for j, result in enumerate(top_results):
                    with cols[j]:
                        try:
                            st.image(result["url"], use_container_width=True)
                        except:
                            st.error("Failed to load image")

                        if result.get("labels"):
                            labels_str = ", ".join(result["labels"][:3])
                            st.caption(f"🏷️ {labels_str}")

                        st.markdown(f"**Score:** {result['score']:.3f}")

                time.sleep(0.3)  # Brief pause for visual effect

                # Phase 2: Show remaining results progressively
                if len(results_list) > 3:
                    st.markdown("**📋 More Results**")

                    remaining_results = results_list[3:]
                    for i in range(0, len(remaining_results), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(remaining_results):
                                result = remaining_results[i + j]
                                with col:
                                    try:
                                        st.image(result["url"], use_container_width=True)
                                    except:
                                        st.error("Failed to load image")

                                    if result.get("labels"):
                                        labels_str = ", ".join(result["labels"][:3])
                                        st.caption(f"🏷️ {labels_str}")

                                    st.markdown(f"**Score:** {result['score']:.3f}")

                        # Small delay between rows for progressive feel
                        if i + cols_per_row < len(remaining_results):
                            time.sleep(0.2)


if __name__ == "__main__":
    main()