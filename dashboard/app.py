import streamlit as st

st.set_page_config(
    page_title="Visual Attention Economy",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, .metric-value { font-family: 'DM Mono', monospace; }

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiselect label { color: #aaa !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }

/* Main background */
.stApp { background: #f7f5f0; }

/* Headers */
h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -0.03em; }
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e8e4dc;
    border-radius: 2px;
    padding: 1rem 1.25rem;
    box-shadow: 2px 2px 0px #e8e4dc;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #0d0d0d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    padding: 0.5rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #888;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    border-bottom: 2px solid #0d0d0d !important;
    color: #0d0d0d !important;
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.25rem;
}

/* Insight box */
.insight-box {
    background: #0d0d0d;
    color: #f7f5f0;
    padding: 1rem 1.25rem;
    border-radius: 2px;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0.75rem 0;
}
.insight-box strong { color: #f0c040; }

/* Divider */
hr { border: none; border-top: 1px solid #e8e4dc; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁 Visual Attention\nEconomy Dashboard")
    st.markdown("---")
    st.markdown('<p class="section-label">Navigate</p>', unsafe_allow_html=True)
    page = st.radio(
        label="",
        options=[
            "🏠  Overview",
            "📉  Banner Blindness",
            "🗺  Spatial Effectiveness",
            "📐  Size vs Attention",
            "🖼  Image Explorer",
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<p class="section-label">Dataset</p>', unsafe_allow_html=True)
    st.markdown("**972** images · **8,447** bounding boxes")
    st.markdown("E-Commercial Dataset · Jiang et al. CVPR 2022")

# ── Route to pages ────────────────────────────────────────────────────────────
if "Overview" in page:
    from pages.overview import show; show()
elif "Banner Blindness" in page:
    from pages.banner_blindness import show; show()
elif "Spatial" in page:
    from pages.spatial import show; show()
elif "Size" in page:
    from pages.size_attention import show; show()
elif "Image Explorer" in page:
    from pages.image_explorer import show; show()
