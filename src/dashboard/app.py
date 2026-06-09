import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
import re
import sys

# Page configuration
st.set_page_config(
    page_title="YouTube Comment Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 YouTube Comment Intelligence Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_topics' not in st.session_state:
    st.session_state.df_topics = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df_topic_info' not in st.session_state:
    st.session_state.df_topic_info = None

# ---------------------------
# Sidebar - File Uploader & Settings
# ---------------------------
with st.sidebar:
    st.header("📁 Data Input")
    
    # Option to choose between file path or upload
    input_method = st.radio("Input Method", ["YouTube URL", "File Path", "File Upload"])
    
    if input_method == "YouTube URL":
        url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        
        if st.button("🚀 Run Analysis", type="primary"):
            if not url:
                st.error("⚠️ Please enter a URL")
            else:
                try:
                    with st.spinner("Running pipeline... This may take a while."):
                        # Ensure the project root is in path
                        project_root = str(Path(__file__).parent.parent.parent)
                        if project_root not in sys.path:
                            sys.path.append(project_root)
                        
                        from main import run_pipeline
                        
                        results = run_pipeline(url, ["all"])
                        
                        if results:
                            st.session_state.df_topics = pd.read_csv(results["topics_path"], encoding="utf-8")
                            st.session_state.df_clean = pd.read_csv(results["cleaned_csv_path"], encoding="utf-8")
                            
                            if Path(results["topic_info_path"]).exists():
                                st.session_state.df_topic_info = pd.read_csv(results["topic_info_path"], encoding="utf-8")
                            else:
                                st.session_state.df_topic_info = None
                                
                            st.session_state.data_loaded = True
                            st.success("✅ Analysis complete! Data loaded.")
                            st.rerun()
                        else:
                            st.error("❌ Pipeline failed. Check console logs.")
                except Exception as e:
                    st.error(f"❌ Error running pipeline: {e}")

    elif input_method == "File Path":
        topics_path = st.text_input("Topics CSV path", placeholder="data/topics/video_topics.csv")
        csv_path = st.text_input("Cleaned CSV path", placeholder="data/cleaned/video.csv")
        topic_info_path = st.text_input("Topic Info CSV path (Optional)", placeholder="data/topics/video_topic_info.csv")
        
        if st.button("🔄 Load Data", type="primary", use_container_width=True):
            if not topics_path or not csv_path:
                st.error("⚠️ Please provide Topics and Cleaned CSV paths")
            elif not Path(topics_path).exists() or not Path(csv_path).exists():
                st.error("❌ Invalid file paths. Check again.")
            else:
                try:
                    with st.spinner("Loading data..."):
                        st.session_state.df_topics = pd.read_csv(topics_path, encoding="utf-8")
                        st.session_state.df_clean = pd.read_csv(csv_path, encoding="utf-8")
                        
                        if topic_info_path and Path(topic_info_path).exists():
                            st.session_state.df_topic_info = pd.read_csv(topic_info_path, encoding="utf-8")
                        else:
                            st.session_state.df_topic_info = None
                            
                        st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
    else:
        topics_file = st.file_uploader("Upload Topics CSV", type=['csv'])
        clean_file = st.file_uploader("Upload Cleaned CSV", type=['csv'])
        topic_info_file = st.file_uploader("Upload Topic Info CSV (Optional)", type=['csv'])
        
        if st.button("🔄 Load Data", type="primary", use_container_width=True):
            if topics_file and clean_file:
                try:
                    with st.spinner("Loading data..."):
                        st.session_state.df_topics = pd.read_csv(topics_file, encoding="utf-8")
                        st.session_state.df_clean = pd.read_csv(clean_file, encoding="utf-8")
                        
                        if topic_info_file:
                            st.session_state.df_topic_info = pd.read_csv(topic_info_file, encoding="utf-8")
                        else:
                            st.session_state.df_topic_info = None
                            
                        st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    show_outliers = st.checkbox("Show Outliers (Topic -1)", value=False)
    max_display = st.slider("Max comments to display", 5, 50, 10)

# ---------------------------
# Main Content
# ---------------------------
if st.session_state.data_loaded and st.session_state.df_topics is not None:
    df_topics = st.session_state.df_topics
    df_clean = st.session_state.df_clean
    
    # Filter outliers if needed
    if not show_outliers:
        df_topics = df_topics[df_topics["topic"] != -1]
        
    # Map topic names if available
    if st.session_state.df_topic_info is not None:
        topic_info = st.session_state.df_topic_info
        # Create mapping: Topic ID -> Name (or keywords)
        
        topic_map = {}
        for _, row in topic_info.iterrows():
            t_id = row.get("Topic")
            if t_id is not None:
                # Try to get a descriptive name
                if "Name" in row:
                    name = row["Name"]
                elif "Representation" in row:
                    name = str(row["Representation"])[:50] + "..." # Truncate if too long
                elif "Count" in row:
                    name = f"Topic {t_id}"
                else:
                    name = f"Topic {t_id}"
                
                # Clean up name if it's just the ID
                if str(name) == str(t_id):
                    name = f"Topic {t_id}"
                    
                topic_map[t_id] = name
        
        # Apply mapping
        df_topics["topic_label"] = df_topics["topic"].map(topic_map).fillna(df_topics["topic"].astype(str))
    else:
        df_topics["topic_label"] = "Topic " + df_topics["topic"].astype(str)
    
    # ---------------------------
    # Overview Metrics
    # ---------------------------
    st.header("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_comments = len(df_topics)
    num_topics = len(df_topics["topic"].unique()) - (1 if -1 in df_topics["topic"].unique() else 0)
    outliers = len(df_topics[df_topics["topic"] == -1]) if -1 in df_topics["topic"].unique() else 0
    avg_likes = df_clean["like_count"].mean() if "like_count" in df_clean.columns else 0
    
    with col1:
        st.metric("Total Comments", f"{total_comments:,}")
    with col2:
        st.metric("Topics Discovered", num_topics)
    with col3:
        st.metric("Outlier Comments", outliers)
    with col4:
        st.metric("Avg Likes", f"{avg_likes:.1f}")
    
    st.divider()
    
    # ---------------------------
    # Tabbed Interface
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Topic Analysis",
        "🔍 Topic Explorer", 
        "🔎 Search Comments",
        "📅 Timeline Analysis",
        "💬 Comment Details"
    ])
    
    # ---------------------------
    # Tab 1: Topic Analysis
    # ---------------------------
    with tab1:
        st.subheader("Topic Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive bar chart
            topic_counts = df_topics["topic_label"].value_counts()
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={"x": "Topic", "y": "Number of Comments"},
                title="Comments per Topic",
                color=topic_counts.values,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Topic Statistics")
            topic_stats = pd.DataFrame({
                "Topic": topic_counts.index,
                "Count": topic_counts.values,
                "Percentage": (topic_counts.values / total_comments * 100).round(2)
            })
            st.dataframe(topic_stats, hide_index=True, height=400)
        
        # Pie chart
        st.subheader("Topic Proportion")
        fig_pie = px.pie(
            values=topic_counts.values,
            names=topic_counts.index,
            title="Topic Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # ---------------------------
    # Tab 2: Topic Explorer
    # ---------------------------
    with tab2:
        st.subheader("Explore Topics in Detail")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            topic_labels = sorted(df_topics["topic_label"].unique())
            selected_topic_label = st.selectbox("Select a topic", topic_labels, key="topic_selector")
            
            # Topic info
            topic_count = len(df_topics[df_topics["topic_label"] == selected_topic_label])
            st.info(f"**{topic_count}** comments in this topic")
            
            # Sorting options
            sort_by = st.radio("Sort by", ["Random", "Likes", "Recent"], key="sort_option")
        
        with col2:
            filtered = df_topics[df_topics["topic_label"] == selected_topic_label].copy()
            
            # Merge with clean data to get likes and timestamps
            if "comment_id" in filtered.columns and "comment_id" in df_clean.columns:
                filtered = filtered.merge(
                    df_clean[["comment_id", "like_count", "published_at", "author"]], 
                    on="comment_id", 
                    how="left"
                )
            
            # Apply sorting
            if sort_by == "Likes" and "like_count" in filtered.columns:
                filtered = filtered.sort_values("like_count", ascending=False)
            elif sort_by == "Recent" and "published_at" in filtered.columns:
                filtered = filtered.sort_values("published_at", ascending=False)
            else:
                filtered = filtered.sample(min(max_display, len(filtered)))
            
            # Display comments
            st.markdown(f"### Showing top {min(max_display, len(filtered))} comments")
            
            for idx, row in filtered.head(max_display).iterrows():
                with st.container():
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(f"**💬 {row.get('author', 'Anonymous')}**")
                        st.write(row["text_clean"])
                    with col_b:
                        if "like_count" in row:
                            st.metric("👍 Likes", int(row["like_count"]))
                        if "topic_probability" in row:
                            st.metric("🎯 Confidence", f"{row['topic_probability']:.2%}")
                    st.divider()
    
    # ---------------------------
    # Tab 3: Search Comments
    # ---------------------------
    with tab3:
        st.subheader("Search Through Comments")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("🔍 Enter keyword or phrase", placeholder="Type to search...")
        
        with col2:
            case_sensitive = st.checkbox("Case sensitive", value=False)
        
        if query:
            try:
                if case_sensitive:
                    matched = df_topics[df_topics["text_clean"].str.contains(query, na=False)]
                else:
                    matched = df_topics[df_topics["text_clean"].str.contains(query, case=False, na=False)]
                
                st.success(f"✅ Found **{len(matched)}** matching comments")
                
                if len(matched) > 0:
                    # Topic distribution of search results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Topics in Results")
                        search_topic_dist = matched["topic_label"].value_counts()
                        fig = px.bar(
                            x=search_topic_dist.index,
                            y=search_topic_dist.values,
                            labels={"x": "Topic", "y": "Count"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Quick Stats")
                        st.metric("Total Matches", len(matched))
                        st.metric("Topics Covered", len(search_topic_dist))
                        if "like_count" in matched.columns:
                            st.metric("Avg Likes", f"{matched['like_count'].mean():.1f}")
                    
                    # Display results
                    st.markdown("### 📋 Search Results")
                    display_cols = ["topic_label", "text_clean"]
                    if "like_count" in matched.columns:
                        display_cols.append("like_count")
                    if "author" in matched.columns:
                        display_cols.append("author")
                    
                    st.dataframe(
                        matched[display_cols].head(50),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv = matched.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Search Results",
                        data=csv,
                        file_name=f"search_results_{query}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Search error: {e}")
    
    # ---------------------------
    # Tab 4: Timeline Analysis
    # ---------------------------
    with tab4:
        st.subheader("Comment Timeline Analysis")
        
        # Check if required columns exist
        has_timestamp = "published_at" in df_clean.columns
        has_comment_id = "comment_id" in df_topics.columns and "comment_id" in df_clean.columns
        
        if has_timestamp and has_comment_id:
            try:
                # Merge topics with timestamps
                timeline_df = df_topics.merge(
                    df_clean[["comment_id", "published_at"]], 
                    on="comment_id", 
                    how="left"
                )
                
                # Check if merge was successful and column exists
                if "published_at" not in timeline_df.columns:
                    st.warning("⚠️ Merge failed - comment IDs may not match between files")
                else:
                    timeline_df["published_at"] = pd.to_datetime(timeline_df["published_at"], errors="coerce")
                    timeline_df = timeline_df.dropna(subset=["published_at"])
                    
                    if len(timeline_df) == 0:
                        st.warning("⚠️ No valid timestamp data found")
                    else:
                        # Time series plot
                        timeline_df["date"] = timeline_df["published_at"].dt.date
                        daily_counts = timeline_df.groupby("date").size().reset_index(name="count")
                        
                        fig = px.line(
                            daily_counts,
                            x="date",
                            y="count",
                            title="Comments Over Time",
                            labels={"date": "Date", "count": "Number of Comments"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Topic evolution
                        st.markdown("### Topic Evolution Over Time")
                        topic_timeline = timeline_df.groupby([timeline_df["published_at"].dt.date, "topic_label"]).size().reset_index(name="count")
                        
                        fig2 = px.area(
                            topic_timeline,
                            x="published_at",
                            y="count",
                            color="topic_label",
                            title="Topic Distribution Over Time"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating timeline: {e}")
                st.info("Make sure both CSV files have matching comment_id values")
        else:
            missing = []
            if not has_timestamp:
                missing.append("'published_at' column in cleaned CSV")
            if not has_comment_id:
                missing.append("'comment_id' column in both CSVs")
            
            st.warning(f"⚠️ Timeline data not available. Missing: {', '.join(missing)}")
    
    # ---------------------------
    # Tab 5: Comment Details
    # ---------------------------
    with tab5:
        st.subheader("Detailed Comment Information")
        
        # Merge all data
        full_df = df_topics.merge(df_clean, on="comment_id", how="left")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_likes = st.number_input("Minimum Likes", min_value=0, value=0)
        with col2:
            selected_topics_filter = st.multiselect(
                "Filter by Topics",
                options=sorted(full_df["topic_label"].unique()),
                default=None
            )
        with col3:
            sort_order = st.selectbox("Sort by", ["Likes (High to Low)", "Likes (Low to High)", "Most Recent"])
        
        # Apply filters
        filtered_full = full_df.copy()
        if "like_count" in filtered_full.columns:
            filtered_full = filtered_full[filtered_full["like_count"] >= min_likes]
        if selected_topics_filter:
            filtered_full = filtered_full[filtered_full["topic_label"].isin(selected_topics_filter)]
        
        # Apply sorting
        if sort_order == "Likes (High to Low)" and "like_count" in filtered_full.columns:
            filtered_full = filtered_full.sort_values("like_count", ascending=False)
        elif sort_order == "Likes (Low to High)" and "like_count" in filtered_full.columns:
            filtered_full = filtered_full.sort_values("like_count", ascending=True)
        elif sort_order == "Most Recent" and "published_at" in filtered_full.columns:
            filtered_full = filtered_full.sort_values("published_at", ascending=False)
        
        st.info(f"Showing **{len(filtered_full)}** comments after filters")
        
        # Display
        display_columns = ["topic_label", "author", "text_clean", "like_count", "published_at"]
        available_columns = [col for col in display_columns if col in filtered_full.columns]
        
        st.dataframe(
            filtered_full[available_columns].head(100),
            use_container_width=True,
            height=500
        )
        
        # Download
        csv = filtered_full.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Filtered Data",
            data=csv,
            file_name="filtered_comments.csv",
            mime="text/csv"
        )

else:
    # Landing page when no data is loaded
    st.info("👈 Please load your data using the sidebar to get started")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    This dashboard helps you analyze YouTube comments with AI-powered topic modeling.
    
    **Required Files:**
    1. **Topics CSV** - Output from topic modeling (contains topic assignments)
    2. **Cleaned CSV** - Cleaned comment data
    3. **Topic Info CSV** (Optional) - Output from topic modeling (contains topic names)
    
    **Features:**
    - 📊 Interactive topic distribution visualizations
    - 🔍 Explore comments within each topic
    - 🔎 Search through all comments
    - 📅 Timeline analysis of comment patterns
    - 💬 Detailed comment filtering and export
    
    **How to use:**
    1. Enter file paths or upload CSVs in the sidebar
    2. Click "Load Data"
    3. Explore the different tabs for insights!
    """)
    
    # Sample data info
    with st.expander("📖 Expected Data Format"):
        st.markdown("""
        **Topics CSV should contain:**
        - `comment_id` - Unique identifier
        - `topic` - Topic assignment (-1 for outliers)
        - `text_clean` - Cleaned comment text
        - `topic_probability` (optional) - Confidence score
        
        **Cleaned CSV should contain:**
        - `comment_id` - Unique identifier
        - `author` - Comment author
        - `like_count` - Number of likes
        - `published_at` - Timestamp
        - `text_raw` - Original comment
        
        **Topic Info CSV should contain:**
        - `Topic` - Topic ID
        - `Name` or `Representation` - Topic Name/Keywords
        """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>YouTube Comment Intelligence Dashboard | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)