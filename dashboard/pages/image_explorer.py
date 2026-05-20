import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
from utils.data import load_global, load_regions, get_image_path, QUAD_COLORS, QUAD_LABELS

def show():
    df   = load_global()
    df_r = load_regions()

    st.markdown("# Image Explorer")
    st.markdown("#### Inspect individual images: product, saliency map, and text bounding boxes")

    # ── Page structure explainer ──────────────────────────────────────────────
    st.markdown("""
    <div style="background:#f7f5f0; border-left:3px solid #0d0d0d; padding:0.9rem 1.1rem; margin-bottom:0.5rem; font-size:0.88rem; line-height:1.7;">
    <strong>📌 How this page is structured</strong><br><br>
    This explorer is divided into <strong>three sections</strong>, each building on the previous:
    </div>
    """, unsafe_allow_html=True)

    # Three-step visual guide
    st.markdown("""
    <div style="display:flex; gap:0.75rem; margin:0.5rem 0 1rem 0; font-size:0.85rem;">

      <div style="flex:1; background:#ffffff; border:1px solid #ddd; border-radius:6px; padding:0.9rem 1rem;">
        <div style="font-size:1.2rem; margin-bottom:0.4rem;">① 🎛️ Filters &amp; Selection</div>
        <strong>Top controls</strong><br>
        <span style="color:#555;">Narrow down the 1,000+ images by <em>text count</em> and <em>dominant quadrant</em>,
        sort the results, then pick a single image to inspect.</span>
      </div>

      <div style="flex:1; background:#ffffff; border:1px solid #ddd; border-radius:6px; padding:0.9rem 1rem;">
        <div style="font-size:1.2rem; margin-bottom:0.4rem;">② 📊 Image Metrics</div>
        <strong>Summary row</strong><br>
        <span style="color:#555;">Five key numbers for the selected image: Text Count, Attention Ratio,
        Clutter Index, and Dominant Quadrant — giving you a quick fingerprint before you look at the visuals.</span>
      </div>

      <div style="flex:1; background:#ffffff; border:1px solid #ddd; border-radius:6px; padding:0.9rem 1rem;">
        <div style="font-size:1.2rem; margin-bottom:0.4rem;">③ 🖼️ Three Visual Panels</div>
        <strong>Side-by-side views</strong><br>
        <span style="color:#555;"><strong>Product Image</strong> → <strong>Saliency Map</strong> → <strong>Bounding Boxes</strong><br>
        Read left to right: what the ad looked like, where eyes went, which text boxes captured attention.</span>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # Panel legend
    st.markdown("""
    <div style="background:#f0f4ff; border:1px solid #c8d4f0; border-radius:6px; padding:0.8rem 1.1rem; margin-bottom:0.5rem; font-size:0.85rem; line-height:1.7;">
    <strong>🖼 Panel guide</strong><br>
    &nbsp;&nbsp;• <strong>Product Image</strong> — the original e-commerce photo as shown to study participants<br>
    &nbsp;&nbsp;• <strong>Saliency Map</strong> — a heatmap derived from real eye-tracking data: <em>brighter = more fixations</em>.
       White/yellow areas are where eyes spent the most time; dark areas were largely ignored<br>
    &nbsp;&nbsp;• <strong>Bounding Boxes</strong> — each detected text region drawn as a coloured rectangle.
       Colour encodes how much of the image's total saliency that box captured:
       <span style="color:#4a90d9;"><strong>blue = low attention</strong></span>,
       <span style="color:#E85D04;"><strong>orange = high attention</strong></span>.
       The number inside each box is the <em>Region Attention Ratio</em> (share of total image saliency)
    </div>
    """, unsafe_allow_html=True)

    # Suggested workflow
    st.markdown("""
    <div style="background:#f0faf0; border:1px solid #b8d8b8; border-radius:6px; padding:0.75rem 1.1rem; margin-bottom:1rem; font-size:0.84rem; line-height:1.65;">
    💡 <strong>Suggested explorations:</strong><br>
    &nbsp;&nbsp;• Set <em>Text Count</em> to <strong>15+</strong> → see how attention scatters across many elements (Banner Blindness in action)<br>
    &nbsp;&nbsp;• Set <em>Text Count</em> to <strong>1–3</strong> → see how a single large text element dominates the saliency map<br>
    &nbsp;&nbsp;• Sort by <strong>Attention Ratio</strong> (descending) → find images where text completely outcompeted the product photo<br>
    &nbsp;&nbsp;• Sort by <strong>Clutter Index</strong> → explore the most visually cluttered ads
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 1: Filters & Selection ───────────────────────────────────────
    st.markdown("##### ① Filter & Select an Image")

    col_ctrl1, col_ctrl2 = st.columns([2, 2])

    with col_ctrl1:
        tc_range = st.slider("Filter images by Text Count", 0, 53, (1, 30))

    with col_ctrl2:
        quad_filter = st.selectbox(
            "Filter by Dominant Quadrant",
            options=['All'] + list(QUAD_LABELS.keys()),
            format_func=lambda x: 'All quadrants' if x == 'All' else QUAD_LABELS[x],
        )

    mask = (df['Text_Count'] >= tc_range[0]) & (df['Text_Count'] <= tc_range[1])
    if quad_filter != 'All':
        mask &= df['Dominant_Quadrant'] == quad_filter
    candidates = df[mask].copy()

    st.caption(f"**{len(candidates)}** images match the current filters.")

    if len(candidates) == 0:
        st.warning("No images match the current filters. Adjust the sliders.")
        return

    col_pick1, col_pick2 = st.columns([2, 2])
    with col_pick1:
        sort_by = st.selectbox(
            "Sort images by",
            ['Image_ID', 'Attention_Ratio', 'Clutter_Index', 'Text_Count'],
        )
        candidates = candidates.sort_values(sort_by, ascending=False if sort_by != 'Image_ID' else True)

    with col_pick2:
        selected_id = st.selectbox(
            "Select image",
            options=candidates['Image_ID'].tolist(),
            format_func=lambda x: f"Image {x}  ·  {int(df[df['Image_ID']==x]['Text_Count'].values[0])} text elements  ·  AR={df[df['Image_ID']==x]['Attention_Ratio'].values[0]:.2f}",
        )

    st.markdown("---")

    # ── Section 2: Image Metrics ──────────────────────────────────────────────
    st.markdown("##### ② Image Metrics")

    row = df[df['Image_ID'] == selected_id].iloc[0]
    regions = df_r[df_r['Image_ID'] == selected_id].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Image ID", selected_id)
    c2.metric("Text Elements", int(row['Text_Count']))
    c3.metric("Attention Ratio", f"{row['Attention_Ratio']:.3f}" if pd.notna(row['Attention_Ratio']) else "N/A")
    c4.metric("Clutter Index", f"{row['Clutter_Index']:.2f}" if pd.notna(row['Clutter_Index']) else "N/A")
    c5.metric("Dominant Quadrant", QUAD_LABELS.get(row['Dominant_Quadrant'], row['Dominant_Quadrant']))

    st.markdown("""
    <div style="font-size:0.8rem; color:#666; line-height:1.6; margin-top:0.3rem;">
    <strong>Attention Ratio</strong> = saliency on text / saliency on product
    &nbsp;|&nbsp; <strong>Clutter Index</strong> = measure of visual complexity (higher = more cluttered)
    &nbsp;|&nbsp; <strong>Dominant Quadrant</strong> = screen region containing the most text
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 3: Three Visual Panels ───────────────────────────────────────
    st.markdown("##### ③ Visual Panels")
    st.caption("Read left → right: original ad · where eyes went · which text boxes captured attention")

    stim_path   = get_image_path(selected_id, 'stimulus')
    fixmap_path = get_image_path(selected_id, 'fixmap')

    images_available = stim_path is not None or fixmap_path is not None

    if not images_available:
        st.info("📁 Images not found in the configured path. Showing bounding box canvas only.\n\nSet the `IMAGES_DIR` environment variable to point to the folder containing ALLSTIMULI and ALLFIXATIONMAPS.")

    col_img1, col_img2, col_img3 = st.columns(3)

    with col_img1:
        st.markdown("**🖼 Product Image**")
        if stim_path:
            img = Image.open(stim_path)
            st.image(img, use_column_width=True)
        else:
            st.markdown("_Image not available_")
        st.caption("The original product photo from the e-commerce dataset, as shown to study participants.")

    with col_img2:
        st.markdown("**👁 Saliency Map**")
        if fixmap_path:
            fixmap = Image.open(fixmap_path)
            st.image(fixmap, use_column_width=True)
        else:
            st.markdown("_Saliency map not available_")
        st.caption(
            "Eye-tracking fixation map from real viewer data. "
            "**Bright/white** = high fixation density. **Dark** = little or no attention."
        )

    with col_img3:
        st.markdown("**📦 Bounding Boxes**")
        if len(regions) > 0:
            fig = go.Figure()
            fig.add_shape(type='rect', x0=0, y0=0, x1=720, y1=720,
                          fillcolor='#1a1a1a', line_color='#333')

            if fixmap_path:
                import base64
                with open(fixmap_path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode()
                fig.add_layout_image(
                    dict(source=f"data:image/jpeg;base64,{encoded}",
                         xref="x", yref="y",
                         x=0, y=720, sizex=720, sizey=720,
                         sizing="stretch", opacity=0.6, layer="below")
                )

            max_rar = regions['Region_Attention_Ratio'].max() if len(regions) > 0 else 1
            for _, reg in regions.iterrows():
                x, y, w, h = reg['BBox_X'], reg['BBox_Y'], reg['BBox_W'], reg['BBox_H']
                y_plot  = 720 - y - h
                y2_plot = 720 - y
                intensity = reg['Region_Attention_Ratio'] / (max_rar + 1e-9)
                r_ch = int(74  + (228-74)  * intensity)
                g_ch = int(144 + (93-144)  * intensity)
                b_ch = int(217 + (4-217)   * intensity)
                color = f"rgba({r_ch},{g_ch},{b_ch},0.75)"
                fig.add_shape(type='rect',
                              x0=x, y0=y_plot, x1=x+w, y1=y2_plot,
                              line=dict(color=color, width=2),
                              fillcolor=color.replace('0.75', '0.2'))
                fig.add_annotation(
                    x=x + w/2, y=y2_plot + 4,
                    text=f"{reg['Region_Attention_Ratio']:.3f}",
                    showarrow=False, font=dict(size=7, color='white'),
                    bgcolor='rgba(0,0,0,0.5)',
                )

            fig.update_layout(
                width=360, height=360,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, 720], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[0, 720], showgrid=False, zeroline=False, showticklabels=False,
                           scaleanchor='x'),
                plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a',
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "🔵 **Blue** = low attention captured by this text region &nbsp;&nbsp;"
                "🟠 **Orange** = high attention captured by this text region. "
                "Number inside each box = Region Attention Ratio (share of total image saliency)."
            )
        else:
            st.markdown("_No text regions detected for this image_")

    # ── Section 4: Regions detail table ──────────────────────────────────────
    if len(regions) > 0:
        st.markdown("---")
        st.markdown("##### 📋 Bounding Box Details — per-region breakdown")
        st.markdown("""
        <div style="font-size:0.83rem; color:#555; line-height:1.6; margin-bottom:0.5rem;">
        Each row is one detected text region in the selected image.
        <strong>Area / Image</strong> = fraction of image area covered by this box.
        <strong>Avg Saliency (density)</strong> = mean eye-tracking intensity per pixel — how attention-dense the box is.
        <strong>Attention Share</strong> = fraction of the image's <em>total</em> saliency captured by this box (used for the colour encoding above).
        </div>
        """, unsafe_allow_html=True)

        display_cols = ['Region_Index', 'BBox_X', 'BBox_Y', 'BBox_W', 'BBox_H',
                        'Region_Area_Ratio', 'Region_Mean_Saliency',
                        'Region_Attention_Ratio', 'Quadrant']
        rename_cols = {
            'BBox_X': 'X position',
            'BBox_Y': 'Y position',
            'BBox_W': 'Width (px)',
            'BBox_H': 'Height (px)',
            'Region_Area_Ratio': 'Area / Image',
            'Region_Mean_Saliency': 'Avg Saliency (density)',
            'Region_Attention_Ratio': 'Attention Share',
            'Quadrant': 'Screen Position',
        }
        st.dataframe(
            regions[display_cols].rename(columns=rename_cols).round(4).set_index('Region_Index'),
            use_container_width=True,
            height=min(300, 40 + len(regions) * 35),
        )