
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="NOVA",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

    * { box-sizing: border-box; }

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #ffffff;
        color: #1d1d1f;
    }

    .main {
        background: #ffffff;
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    /* ── Hero ── */
    .nova-hero {
        margin-left: -6.5% !important;
        margin-right: -6.5% !important;
        background: #000000;
        min-height: 100vh;
        padding: 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .nova-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 40%, rgba(99,102,241,0.15) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 60%, rgba(168,85,247,0.1) 0%, transparent 50%);
        pointer-events: none;
    }

    .nova-wordmark {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.5);
        margin-bottom: 20px;
    }

    .nova-headline {
        font-family: 'DM Sans', sans-serif;
        font-size: clamp(2.5rem, 5vw, 4.5rem);
        font-weight: 300;
        color: #ffffff;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin-bottom: 16px;
    }

    .nova-headline span {
        font-weight: 600;
        background: linear-gradient(135deg, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .nova-subheadline {
        font-size: 1.1rem;
        font-weight: 300;
        color: rgba(255,255,255,0.55);
        max-width: 520px;
        margin: 0 auto 40px;
        line-height: 1.6;
    }

    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 60px;
        margin-top: 48px;
        padding-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.08);
    }

    .hero-stat-value {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.03em;
    }

    .hero-stat-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.4);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ── Nav tabs ── */
    .nova-nav {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid #e8e8ed;
        padding: 0 80px;
        position: sticky;
        top: 0;
        z-index: 100;
        display: flex;
        gap: 0;
    }

    /* ── Content area ── */
    .nova-content {
        padding: 60px 10%;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* ── Section titles ── */
    .section-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 8px;
    }

    .section-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .section-body {
        font-size: 1rem;
        font-weight: 300;
        color: #6e6e73;
        line-height: 1.7;
        max-width: 600px;
        margin-bottom: 40px;
    }

    /* ── Search input ── */
    .stTextInput input {
        background: #f5f5f7 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        font-size: 1rem !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #1d1d1f !important;
        box-shadow: none !important;
        transition: background 0.2s ease !important;
    }

    .stTextInput input:focus {
        background: #ebebf0 !important;
        outline: none !important;
        box-shadow: none !important;
    }

    .stTextInput input::placeholder {
        color: #aeaeb2 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: #f5f5f7 !important;
        border: none !important;
        border-radius: 12px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #1d1d1f !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 980px !important;
        padding: 10px 24px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        transition: background 0.2s ease !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        background: #3a3a3c !important;
    }

    /* ── Product cards ── */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px;
        margin-top: 24px;
    }

    .product-card {
        background: #ffffff;
        border: 1px solid #e8e8ed;
        border-radius: 18px;
        padding: 24px;
        transition: box-shadow 0.3s ease, transform 0.2s ease;
        position: relative;
    }

    .product-card:hover {
        box-shadow: 0 8px 40px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }

    .product-rank {
        position: absolute;
        top: 16px;
        right: 16px;
        width: 28px;
        height: 28px;
        background: #f5f5f7;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        color: #aeaeb2;
    }

    .product-category {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 8px;
    }

    .product-name {
        font-size: 0.95rem;
        font-weight: 400;
        color: #1d1d1f;
        line-height: 1.5;
        margin-bottom: 16px;
    }

    .product-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 16px;
        border-top: 1px solid #f5f5f7;
    }

    .product-price {
        font-size: 0.9rem;
        font-weight: 500;
        color: #1d1d1f;
    }

    .product-score {
        font-size: 0.78rem;
        color: #aeaeb2;
        font-weight: 400;
    }

    /* ── Metric tiles ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 40px;
    }

    .metric-tile {
        background: #f5f5f7;
        border-radius: 18px;
        padding: 28px;
    }

    .metric-tile-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.04em;
        line-height: 1;
        margin-bottom: 6px;
    }

    .metric-tile-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #aeaeb2;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-tile-delta {
        font-size: 0.82rem;
        color: #34c759;
        font-weight: 500;
        margin-top: 4px;
    }

    /* ── Tags ── */
    .tag {
        display: inline-block;
        background: #f5f5f7;
        border-radius: 980px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #6e6e73;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .tag-cold {
        background: #fff4e5;
        color: #c45e00;
    }

    .tag-warm {
        background: #e8faf0;
        color: #1a7a40;
    }

    .tag-purple {
        background: #f0eeff;
        color: #4f46e5;
    }

    /* ── Session log ── */
    .log-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid #f5f5f7;
        font-size: 0.88rem;
        color: #3a3a3c;
    }

    .log-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6366f1;
        flex-shrink: 0;
    }

    .log-event {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #aeaeb2;
        min-width: 80px;
    }

    /* ── Tabs override ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0;
        border-bottom: 1px solid #e8e8ed;
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        padding: 16px 24px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        color: #6e6e73;
        border-bottom: 2px solid transparent;
        border-radius: 0;
    }

    .stTabs [aria-selected="true"] {
        color: #1d1d1f !important;
        border-bottom: 2px solid #1d1d1f !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding: 40px 0 0 0;
    }

    /* ── Divider ── */
    .apple-divider {
        border: none;
        border-top: 1px solid #e8e8ed;
        margin: 48px 0;
    }

    /* ── Slider ── */
    .stSlider { padding: 0; }

    /* ── Checkbox ── */
    .stCheckbox label {
        font-size: 0.88rem;
        color: #6e6e73;
    }

    /* ── Info box ── */
    .info-box {
        background: #f5f5f7;
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 0.88rem;
        color: #6e6e73;
        line-height: 1.6;
    }

    /* ── Example pills ── */
    .example-pill {
        display: inline-block;
        background: #f0eeff;
        color: #4f46e5;
        border-radius: 980px;
        padding: 6px 16px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 4px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

ROOT      = Path(__file__).resolve().parent.parent
EMBED_DIR = ROOT / "data" / "processed" / "embeddings"
INDEX_DIR = ROOT / "data" / "processed" / "index"

INTERACTION_WEIGHTS = {"view": 0.1, "add_to_cart": 0.3, "purchase": 1.0}

@st.cache_resource(show_spinner="Loading NOVA...")
def load_assets():
    index       = faiss.read_index(str(INDEX_DIR / "nova_product.index"))
    product_ids = np.load(EMBED_DIR / "product_ids.npy", allow_pickle=True)
    embeddings  = np.load(EMBED_DIR / "product_embeddings.npy").astype("float32")
    metadata    = pd.read_csv(EMBED_DIR / "product_metadata.csv")
    centroids   = pd.read_csv(EMBED_DIR / "category_centroids.csv", index_col="category")
    model       = SentenceTransformer("all-MiniLM-L6-v2")
    pid_to_idx  = {pid: i for i, pid in enumerate(product_ids)}
    return index, product_ids, embeddings, metadata, centroids, model, pid_to_idx

index, product_ids, embeddings, metadata, centroids, stmodel, pid_to_idx = load_assets()

def search(query_vector, top_k=10, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    q = query_vector.astype("float32").reshape(1, -1)
    q = q / np.linalg.norm(q)
    scores, indices = index.search(q, top_k + len(exclude_ids) + 10)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1: continue
        pid = product_ids[idx]
        if pid in exclude_ids: continue
        row = metadata[metadata["product_id"] == pid]
        if len(row) == 0: continue
        results.append({
            "product_id":   pid,
            "category_en":  row.iloc[0]["category_en"],
            "product_text": row.iloc[0]["product_text"],
            "avg_price":    row.iloc[0]["avg_price"],
            "similarity":   float(score),
        })
        if len(results) == top_k: break
    return pd.DataFrame(results)

def render_product_grid(recs_df, top_k=10):
    if recs_df is None or len(recs_df) == 0:
        st.markdown('<div class="info-box">No results found.</div>', unsafe_allow_html=True)
        return
    cards_html = '<div class="product-grid">'
    for rank, (_, row) in enumerate(recs_df.head(top_k).iterrows(), 1):
        category = str(row.get("category_en", "")).replace("_", " ").upper()
        text     = str(row.get("product_text", ""))
        price    = row.get("avg_price")
        score    = row.get("similarity", 0)
        price_str = f"R${price:.0f}" if pd.notna(price) else "—"
        cards_html += f"""
        <div class="product-card">
            <div class="product-rank">{rank}</div>
            <div class="product-category">{category}</div>
            <div class="product-name">{text}</div>
            <div class="product-footer">
                <span class="product-price">{price_str}</span>
                <span class="product-score">{score:.4f}</span>
            </div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

category_options = sorted(centroids.index.tolist())
category_display = [c.replace("_", " ").title() for c in category_options]
cat_map = dict(zip(category_display, category_options))

st.markdown(f"""
<div class="nova-hero">
    <div class="nova-wordmark">✦ NOVA</div>
    <div class="nova-headline">Recommendations that work<br><span>from the first visit.</span></div>
    <div class="nova-subheadline">
        Neural Object-Vector Architecture for cold-start ecommerce recommendations.
        No purchase history needed.
    </div>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-value">{index.ntotal:,}</div>
            <div class="hero-stat-label">Products Indexed</div>
        </div>
        <div>
            <div class="hero-stat-value">{len(centroids)}</div>
            <div class="hero-stat-label">Categories</div>
        </div>
        <div>
            <div class="hero-stat-value">0.67ms</div>
            <div class="hero-stat-label">Median Latency</div>
        </div>
        <div>
            <div class="hero-stat-value">384</div>
            <div class="hero-stat-label">Embedding Dims</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="nova-content">', unsafe_allow_html=True)

top_k = st.slider("", 5, 20, 10, label_visibility="collapsed",
                  help="Number of recommendations to return")
st.markdown(f'<p style="font-size:0.78rem;color:#aeaeb2;margin-top:-8px;margin-bottom:32px">Showing top {top_k} results</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Search",
    "Browse",
    "Similar",
    "Session",
])

with tab1:
    st.markdown('<div class="section-eyebrow">Semantic Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Search by anything.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-body">Your query is embedded into the same vector space as the product catalog. No keywords needed — just describe what you\'re looking for.</div>', unsafe_allow_html=True)

    query = st.text_input("", placeholder="Try: affordable running shoes, baby toys, kitchen appliances...", key="search_query", label_visibility="collapsed")

    st.markdown("""
    <div style="margin: 12px 0 32px">
        <span class="example-pill">affordable kitchen appliances</span>
        <span class="example-pill">outdoor sports gear</span>
        <span class="example-pill">premium beauty products</span>
        <span class="example-pill">baby toys accessories</span>
        <span class="example-pill">small electronics</span>
    </div>
    """, unsafe_allow_html=True)

    if query:
        with st.spinner(""):
            qvec = stmodel.encode([query], normalize_embeddings=True)[0]
            recs = search(qvec, top_k=top_k)
        st.markdown(f'<p style="font-size:0.85rem;color:#6e6e73;margin-bottom:4px"><span class="tag tag-purple">{len(recs)} results</span> for "{query}"</p>', unsafe_allow_html=True)
        render_product_grid(recs, top_k)

with tab2:
    st.markdown('<div class="section-eyebrow">Category Cold-Start</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Land anywhere. Get recommendations instantly.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-body">Simulates a user landing on a category page with zero history. NOVA uses the category centroid vector to surface the most representative products.</div>', unsafe_allow_html=True)

    selected_display = st.selectbox("", category_display, key="cat_select", label_visibility="collapsed")
    selected_cat = cat_map[selected_display]

    centroid_vec = centroids.loc[selected_cat].values.astype("float32")
    centroid_vec = centroid_vec / np.linalg.norm(centroid_vec)
    recs = search(centroid_vec, top_k=top_k)

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:24px 0 8px">
        <span class="tag tag-cold">Cold Start</span>
        <span style="font-size:0.85rem;color:#6e6e73">{selected_display}</span>
    </div>
    """, unsafe_allow_html=True)
    render_product_grid(recs, top_k)

with tab3:
    st.markdown('<div class="section-eyebrow">Product Similarity</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">You might also like.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-body">Enter a product ID to surface the most semantically similar items in the catalog. Powers "related products" widgets.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        pid_input = st.text_input("", placeholder="Paste a product_id from the dataset...", key="pid_input", label_visibility="collapsed")
    with col2:
        find_btn = st.button("Find Similar", use_container_width=True)

    if st.checkbox("Show example product IDs", key="show_examples"):
        sample = metadata.groupby("category_en").first().reset_index()[["product_id", "category_en"]].head(10)
        st.dataframe(sample, use_container_width=True, hide_index=True)

    if pid_input and find_btn:
        if pid_input not in pid_to_idx:
            st.markdown('<div class="info-box">Product ID not found in index. Try one from the examples above.</div>', unsafe_allow_html=True)
        else:
            idx       = pid_to_idx[pid_input]
            query_vec = embeddings[idx]
            qrow      = metadata[metadata["product_id"] == pid_input].iloc[0]

            st.markdown(f"""
            <div class="product-card" style="max-width:480px;margin:24px 0 8px;border-color:#1d1d1f">
                <div class="product-category">Query Product</div>
                <div class="product-category" style="color:#6366f1">{str(qrow['category_en']).replace('_',' ').upper()}</div>
                <div class="product-name">{qrow['product_text']}</div>
            </div>
            """, unsafe_allow_html=True)

            recs = search(query_vec, top_k=top_k, exclude_ids={pid_input})
            st.markdown(f'<p style="font-size:0.85rem;color:#6e6e73;margin-bottom:4px"><span class="tag">{len(recs)} similar products</span></p>', unsafe_allow_html=True)
            render_product_grid(recs, top_k)

with tab4:
    st.markdown('<div class="section-eyebrow">Online Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Watch recommendations evolve.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-body">Each interaction shifts the user vector toward the interacted product\'s embedding. Purchases move it more than views. The system personalizes in real time without retraining.</div>', unsafe_allow_html=True)

    if "sv"  not in st.session_state: st.session_state.sv   = None
    if "log" not in st.session_state: st.session_state.log  = []
    if "snaps" not in st.session_state: st.session_state.snaps = []

    col_setup, col_recs = st.columns([1, 2], gap="large")

    with col_setup:
        st.markdown("**Setup**")
        init_cat = st.selectbox("Starting category", ["— none —"] + category_display, key="sess_cat", label_visibility="collapsed")

        if st.button("Reset Session", use_container_width=True):
            st.session_state.sv    = None
            st.session_state.log   = []
            st.session_state.snaps = []
            if init_cat != "— none —":
                raw = cat_map.get(init_cat)
                if raw and raw in centroids.index:
                    vec = centroids.loc[raw].values.astype("float32")
                    st.session_state.sv = vec / np.linalg.norm(vec)
            st.rerun()

        st.markdown("<hr class='apple-divider' style='margin:20px 0'>", unsafe_allow_html=True)
        st.markdown("**Add Interaction**")

        interact_pid   = st.text_input("", placeholder="Product ID...", key="int_pid", label_visibility="collapsed")
        interact_event = st.selectbox("", ["view", "add_to_cart", "purchase"], key="int_event", label_visibility="collapsed")

        if st.button("Record Interaction", use_container_width=True):
            if interact_pid not in pid_to_idx:
                st.error("Product ID not found.")
            else:
                pvec            = embeddings[pid_to_idx[interact_pid]]
                weight          = INTERACTION_WEIGHTS[interact_event]
                effective_alpha = 0.3 * weight
                cur             = st.session_state.sv
                new_vec         = pvec.copy() if cur is None else (1 - effective_alpha) * cur + effective_alpha * pvec
                new_vec         = new_vec / np.linalg.norm(new_vec)
                st.session_state.sv = new_vec

                cat = metadata[metadata["product_id"] == interact_pid]["category_en"].values
                cat = cat[0] if len(cat) > 0 else "unknown"
                st.session_state.log.append({"product_id": interact_pid, "event": interact_event, "category": cat})

                snap_recs = search(new_vec, top_k=10)
                if len(snap_recs) > 0:
                    st.session_state.snaps.append(snap_recs["category_en"].value_counts().head(5).to_dict())
                st.rerun()

        st.markdown("<hr class='apple-divider' style='margin:20px 0'>", unsafe_allow_html=True)

        n_interactions = len(st.session_state.log)
        is_cold = n_interactions == 0
        badge   = "tag-cold" if is_cold else "tag-warm"
        label   = "Cold Start" if is_cold else f"Warm · {n_interactions} interactions"
        st.markdown(f'<span class="tag {badge}">{label}</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.log:
            log_html = ""
            for item in reversed(st.session_state.log[-6:]):
                log_html += f"""
                <div class="log-item">
                    <div class="log-dot"></div>
                    <div class="log-event">{item['event']}</div>
                    <div>{item['category'].replace('_',' ').title()}</div>
                </div>"""
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">No interactions yet. Set a category and reset to begin.</div>', unsafe_allow_html=True)

    with col_recs:
        if st.session_state.sv is None:
            st.markdown('<div class="info-box" style="margin-top:48px">Set a starting category and click Reset Session to begin the simulation.</div>', unsafe_allow_html=True)
        else:
            recs = search(
                st.session_state.sv,
                top_k=top_k,
                exclude_ids={log["product_id"] for log in st.session_state.log}
            )
            render_product_grid(recs, top_k)

        if len(st.session_state.snaps) >= 2:
            st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
            st.markdown("**Recommendation Drift**")
            st.markdown('<p style="font-size:0.85rem;color:#aeaeb2;margin-bottom:16px">How top recommended categories shift with each interaction.</p>', unsafe_allow_html=True)

            snaps    = st.session_state.snaps
            all_cats = sorted(set(c for s in snaps for c in s))
            matrix   = np.array([[s.get(c, 0) for c in all_cats] for s in snaps])

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#fafafa")
            colors_drift = ["#6366f1", "#a78bfa", "#34c759", "#f59e0b", "#ef4444"]
            for i, cat in enumerate(all_cats):
                ax.plot(range(1, len(snaps)+1), matrix[:, i], marker="o",
                        label=cat.replace("_", " "), linewidth=2,
                        color=colors_drift[i % len(colors_drift)])
            ax.set_xlabel("Interaction #", fontsize=9, color="#aeaeb2")
            ax.set_ylabel("Count", fontsize=9, color="#aeaeb2")
            ax.set_xticks(range(1, len(snaps)+1))
            ax.tick_params(colors="#aeaeb2", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#e8e8ed")
            ax.spines["bottom"].set_color("#e8e8ed")
            ax.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:#000;padding:40px 80px;text-align:center;margin-top:60px;margin-left:-6.5%;margin-right:-6.5%;">
    <div style="font-size:0.75rem;color:rgba(255,255,255,0.3);letter-spacing:0.05em">
        ✦ NOVA · Neural Object-Vector Architecture · Built with PyTorch, Sentence-Transformers & FAISS
    </div>
</div>
""", unsafe_allow_html=True)
