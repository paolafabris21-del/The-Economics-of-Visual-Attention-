import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
from utils.data import load_global, load_regions, QUAD_COLORS, QUAD_LABELS

QUAD_ORDER = ['Q1_TopLeft', 'Q2_TopRight', 'Q3_BottomLeft', 'Q4_BottomRight']
NICE_LABELS = ['Q1\nTop-Left', 'Q2\nTop-Right', 'Q3\nBottom-Left', 'Q4\nBottom-Right']


def show():
    df  = load_global()
    df_r = load_regions()
    df_valid = df[df['Dominant_Quadrant'] != 'NoText'].copy()

    st.markdown("# Spatial Effectiveness")
    st.markdown("#### Which screen quadrant yields the highest visual attention?")
    st.markdown("""
    <div class="insight-box">
    Despite <strong>Q4 Bottom-Right</strong> being the most common text placement (n = 287),
    bounding boxes in <strong>Q1 Top-Left</strong> attract the highest median saliency (53.4 vs 29.7).
    &nbsp;·&nbsp; Kruskal-Wallis H = 85.2, <strong>p = 2.3×10⁻¹⁸</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Key concepts ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fff8f0; border:1px solid #f0e0c8; border-radius:6px; padding:1rem 1.2rem; margin:0.8rem 0 1rem 0; font-size:0.87rem; line-height:1.7;">
    <strong>📐 Key concepts for this page</strong><br><br>
    <table style="width:100%; border-collapse:collapse;">
      <tr>
        <td style="padding:0.3rem 0.8rem 0.3rem 0; width:33%; vertical-align:top;">
          <strong>Dominant Quadrant</strong><br>
          <span style="color:#555;">The screen quadrant (Top-Left, Top-Right, Bottom-Left, Bottom-Right) where the majority of an image's text bounding boxes are concentrated.</span>
        </td>
        <td style="padding:0.3rem 0.8rem 0.3rem 0; width:33%; vertical-align:top;">
          <strong>Region Mean Saliency</strong><br>
          <span style="color:#555;">The average eye-tracking intensity inside a single text bounding box. Higher = that specific text region attracted stronger, more sustained gaze.</span>
        </td>
        <td style="padding:0.3rem 0 0.3rem 0; width:33%; vertical-align:top;">
          <strong>Attention Ratio</strong><br>
          <span style="color:#555;">Saliency captured by all text ÷ saliency captured by the product. Values &gt; 1 mean text dominated the viewer's gaze.</span>
        </td>
      </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Level selector ────────────────────────────────────────────────────────
    level = st.radio(
        "Analysis level",
        ["Image level (Dominant Quadrant)", "Region level (per bounding box)"],
        horizontal=True,
    )
    st.caption(
        "**Image level** treats each image as one observation, classified by the "
        "quadrant where its text is concentrated (the image's *dominant* quadrant). "
        "**Region level** treats each individual text bounding box as one observation. "
        "The two levels can tell slightly different stories, so read them together."
    )

    # ── Suggested explorations ────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#f0faf0; border:1px solid #b8d8b8; border-radius:6px; padding:0.75rem 1.1rem; margin-bottom:0.5rem; font-size:0.84rem; line-height:1.65;">
    💡 <strong>Suggested explorations:</strong><br>
    &nbsp;&nbsp;• Start with <strong>Image level</strong> for the big picture — which quadrant is most used vs. which earns more attention<br>
    &nbsp;&nbsp;• Switch to <strong>Region level</strong> for the fine-grained story — how individual bounding boxes perform across quadrants<br>
    &nbsp;&nbsp;• On Region level, compare the <strong>violin widths</strong>: Q1 and Q2 have heavier upper tails (some very high-saliency outliers)<br>
    &nbsp;&nbsp;• Check the <strong>4×4 heatmap</strong> on Region level — it shows exactly where on the canvas attention concentrates most
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    if "Image level" in level:
        st.markdown("""
        <div style="background:#f7f5f0; border-left:3px solid #0d0d0d; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.88rem; line-height:1.6;">
        📌 <strong>What you're looking at (Image level):</strong> each of the 914 images with text is assigned to its <strong>dominant quadrant</strong> — the screen region where most of its text lands.
        The <strong>boxplot</strong> (left) shows how the Attention Ratio varies across those quadrant groups.
        The <strong>bar chart</strong> (right) shows how often designers actually place text in each quadrant, so you can spot the gap between design habit and attention reality.
        </div>
        """, unsafe_allow_html=True)

        # ── Image-level: boxplot of Attention_Ratio by dominant quadrant ──────
        with col_left:
            st.markdown("##### Attention Ratio by Dominant Quadrant")
            fig = go.Figure()
            for q in QUAD_ORDER:
                sub = df_valid[df_valid['Dominant_Quadrant'] == q]['Attention_Ratio'].dropna()
                fig.add_trace(go.Box(
                    y=sub, name=QUAD_LABELS[q],
                    marker_color=QUAD_COLORS[q],
                    line_color=QUAD_COLORS[q],
                    fillcolor=QUAD_COLORS[q],
                    opacity=0.6,
                    boxmean=True,
                    hoveron='points',
                    hovertemplate="<b>%{x}</b><br>Attention Ratio: %{y:.2f}<extra></extra>",
                ))
            fig.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font_family="Syne", height=440,
                margin=dict(l=0, r=0, t=44, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(title='Attention Ratio', showgrid=True, gridcolor='#f0ece4'),
                hovermode='closest',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0,
                            font_size=11),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each box shows the spread of the per-image **Attention Ratio** for the "
                "images whose text is concentrated in that quadrant. The solid line is the "
                "median, the dashed line the mean, and the dots are outliers. Images "
                "dominated by **Q1 Top-Left** reach a slightly higher median (4.90) than "
                "the other quadrants (~4.1) — a real but small gap (Kruskal-Wallis "
                "*p* = 0.044)."
            )

        with col_right:
            st.markdown("##### Image count & median Attention Ratio")
            stats_rows = []
            for q in QUAD_ORDER:
                sub = df_valid[df_valid['Dominant_Quadrant'] == q]['Attention_Ratio'].dropna()
                stats_rows.append({
                    'Quadrant': QUAD_LABELS[q],
                    'n Images': len(sub),
                    'Median AR': round(sub.median(), 3),
                    'Mean AR': round(sub.mean(), 3),
                    'Share': f"{len(sub)/len(df_valid)*100:.1f}%",
                })
            st.dataframe(
                pd.DataFrame(stats_rows).set_index('Quadrant'),
                use_container_width=True, height=200,
            )
            st.caption(
                "For each dominant quadrant: how many images fall into it, their median "
                "and mean Attention Ratio, and their share of the 914 images that contain "
                "text. Q4 Bottom-Right is the most common placement, Q2 Top-Right the rarest."
            )

            st.markdown("---")
            st.markdown("##### Quadrant frequency")
            freq = df['Dominant_Quadrant'].value_counts()
            fig2 = go.Figure()
            for q in QUAD_ORDER:
                n = int(freq.get(q, 0))
                fig2.add_trace(go.Bar(
                    x=[QUAD_LABELS[q]], y=[n],
                    name=QUAD_LABELS[q],
                    marker_color=QUAD_COLORS[q],
                    text=[n], textposition='outside',
                    hovertemplate="<b>%{x}</b><br>%{y} images<extra></extra>",
                ))
            fig2.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font_family="Syne", height=280,
                margin=dict(l=0, r=0, t=44, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f0ece4', title='Images'),
                hovermode='closest',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0,
                            font_size=11),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Number of images assigned to each dominant quadrant (the 58 'No Text' "
                "images are excluded). Text is placed most often in the bottom half of "
                "the canvas (Q3 + Q4), even though Q1 attracts marginally more attention."
            )

        st.markdown("""
        <div style="background:#fdf3e7; border-left:4px solid #E85D04; padding:0.8rem 1rem; margin-top:1.2rem; font-size:0.87rem; line-height:1.65;">
        <strong>🔍 Takeaway — Image level:</strong><br>
        Q4 Bottom-Right is the most used placement (287 images, ~31%), yet Q1 Top-Left earns a higher median Attention Ratio.
        The gap is statistically real (Kruskal-Wallis p = 0.044) though modest — placement alone does not explain everything.
        Switch to <strong>Region level</strong> for the stronger, more granular signal where all 8,447 bounding boxes are analysed individually.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#f7f5f0; border-left:3px solid #0d0d0d; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.88rem; line-height:1.6;">
        📌 <strong>What you're looking at (Region level):</strong> each of the 8,447 individual text bounding boxes becomes one data point.
        The <strong>violin plot</strong> (left) shows the full distribution of attention per box by quadrant — not just the average, but the shape of the spread and any outliers.
        The <strong>4×4 heatmap</strong> (right) breaks the canvas into a grid and shows where attention is hottest across the entire image surface.
        </div>
        """, unsafe_allow_html=True)

        # ── Region-level: violin + heatmap ────────────────────────────────────
        with col_left:
            st.markdown("##### Region Mean Saliency by Quadrant")
            fig = go.Figure()
            for q in QUAD_ORDER:
                sub = df_r[df_r['Quadrant'] == q]['Region_Mean_Saliency'].dropna().values
                fig.add_trace(go.Violin(
                    y=sub, name=QUAD_LABELS[q],
                    fillcolor='rgba({},{},{},0.73)'.format(int(QUAD_COLORS[q][1:3],16), int(QUAD_COLORS[q][3:5],16), int(QUAD_COLORS[q][5:7],16)),
                    line_color=QUAD_COLORS[q],
                    meanline_visible=True,
                    box_visible=True,
                    points='outliers',
                    hoveron='points',
                    hovertemplate="<b>%{x}</b><br>Saliency: %{y:.1f}<extra></extra>",
                ))
            kw_groups = [df_r[df_r['Quadrant'] == q]['Region_Mean_Saliency'].dropna().values
                         for q in QUAD_ORDER]
            kw_stat, kw_p = stats.kruskal(*kw_groups)
            fig.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font_family="Syne", height=440,
                margin=dict(l=0, r=0, t=44, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(title='Region Mean Saliency', showgrid=True, gridcolor='#f0ece4'),
                hovermode='closest',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0,
                            font_size=11),
                violingap=0.15,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Here every text **bounding box** is one observation ({len(df_r):,} in "
                "total, not just one per image). Each violin shows the distribution of "
                "Region Mean Saliency — the wider the shape at a given height, the more "
                "boxes have that value; the inner box marks the quartiles and median. "
                "Top quadrants (Q1, Q2) attract clearly stronger saliency than bottom "
                "ones (Q1 median 53.4 vs Q4 median 29.7). A Kruskal-Wallis test "
                f"(H = {kw_stat:.1f}, p = {kw_p:.1e}) confirms the difference is highly "
                "significant."
            )

        with col_right:
            st.markdown("##### Spatial canvas heatmap (4×4 grid)")
            df_r = df_r.copy()
            df_r['grid_x'] = pd.cut(df_r['Center_X'], bins=4,
                                     labels=['Left', 'Ctr-L', 'Ctr-R', 'Right'])
            df_r['grid_y'] = pd.cut(df_r['Center_Y'], bins=4,
                                     labels=['Top', 'Upper', 'Lower', 'Bottom'])
            pivot = df_r.groupby(['grid_y', 'grid_x'], observed=True)['Region_Mean_Saliency'].mean().unstack()
            fig3 = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale='YlOrRd',
                text=pivot.values.round(1),
                texttemplate="%{text}",
                hovertemplate="<b>%{y} / %{x}</b><br>Mean Saliency: %{z:.1f}<extra></extra>",
                colorbar=dict(title='Saliency'),
            ))
            fig3.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font_family="Syne", height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(autorange='reversed'),
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "The 720×720 canvas is split into a 4×4 grid by each box's centroid; "
                "every cell is the mean Region Mean Saliency of the boxes that fall in "
                "it. Rows run **Top → Bottom** and columns **Left → Right**, so the grid "
                "matches the screen. The warm band across the upper-centre cells confirms "
                "the headline: attention concentrates near the top-centre of the screen, "
                "not in the corners or at the bottom."
            )

            st.markdown("---")
            st.markdown("##### Region-level summary by quadrant")
            stats_rows = []
            for q in QUAD_ORDER:
                sub = df_r[df_r['Quadrant'] == q]['Region_Mean_Saliency'].dropna()
                stats_rows.append({
                    'Quadrant': QUAD_LABELS[q],
                    'n BBoxes': len(sub),
                    'Median Sal': round(sub.median(), 1),
                    'Mean Sal': round(sub.mean(), 1),
                })
            st.dataframe(
                pd.DataFrame(stats_rows).set_index('Quadrant'),
                use_container_width=True, height=200,
            )
            st.caption(
                "Per-quadrant count of text bounding boxes and their median / mean "
                "Region Mean Saliency. The ranking Q1 > Q2 > Q3 > Q4 mirrors the violin "
                "plot on the left."
            )

        st.markdown("""
        <div style="background:#fdf3e7; border-left:4px solid #E85D04; padding:0.8rem 1rem; margin-top:1.2rem; font-size:0.87rem; line-height:1.65;">
        <strong>🔍 Takeaway — Region level:</strong><br>
        At the bounding-box level the Q1 advantage is unmistakable: median saliency <strong>53.4 vs. 29.7</strong> for Q4
        (Kruskal-Wallis p = 2.3×10⁻¹⁸). The heatmap confirms the pattern spatially — the top-centre of the canvas
        is consistently the hottest zone for gaze, regardless of which image is shown.
        <br><br>
        <strong>For UX designers: if a text element must compete for attention, placing it in the top-left
        or top-centre gives it the strongest statistical advantage of being seen.</strong>
        </div>
        """, unsafe_allow_html=True)
