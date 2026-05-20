import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from utils.data import load_global, load_regions, QUAD_COLORS, QUAD_LABELS

def _fmt_pval(p: float) -> str:
    """Format p-values gracefully, handling floating-point underflow."""
    if p == 0.0:
        return "< 1e-300"
    elif p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"

def show():
    df   = load_global()
    df_r = load_regions()
    df_valid = df[df['Dominant_Quadrant'] != 'NoText'].copy()

    st.markdown("# Size vs. Attention")
    st.markdown("#### Does making a text element larger capture more attention?")

    # ── Main insight box ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="insight-box">
    <strong>In a nutshell:</strong> the answer depends on <em>what</em> you measure and <em>at what scale</em>.<br><br>
    At <strong>image level</strong>: more text area is weakly associated with <em>lower</em> attention ratio
    (r = −0.197) — consistent with Banner Blindness.<br>
    At <strong>region level</strong>: larger individual bounding boxes capture more <em>total</em> saliency
    (r = +0.71), but <em>not</em> more attention per pixel (r ≈ +0.03).<br><br>
    👉 Use the toggle below to switch between the two levels of analysis.
    </div>
    """, unsafe_allow_html=True)

    # ── Conceptual explainer ──────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fff8f0; border:1px solid #f0e0c8; border-radius:6px; padding:1rem 1.2rem; margin:0.8rem 0 1rem 0; font-size:0.87rem; line-height:1.7;">
    <strong>📐 Key concepts before you dive in</strong><br><br>
    <table style="width:100%; border-collapse:collapse;">
      <tr>
        <td style="padding:0.3rem 0.8rem 0.3rem 0; width:38%; vertical-align:top;">
          <strong>Text Area Ratio</strong><br>
          <span style="color:#555;">Fraction of the whole image covered by all text elements combined (0 = no text, 1 = entire image is text).</span>
        </td>
        <td style="padding:0.3rem 0.8rem 0.3rem 0; width:38%; vertical-align:top;">
          <strong>Attention Ratio</strong><br>
          <span style="color:#555;">Saliency captured by text ÷ saliency captured by the product. Values &gt; 1 mean text "won"; values &lt; 1 mean the product photo attracted more gaze.</span>
        </td>
        <td style="padding:0.3rem 0 0.3rem 0; width:24%; vertical-align:top;">
          <strong>OLS line</strong><br>
          <span style="color:#555;">Orange regression line showing the overall trend across all data points.</span>
        </td>
      </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    level = st.radio(
        "Analysis level",
        ["Image level", "Region level"],
        horizontal=True,
    )

    # ── Level description pills ───────────────────────────────────────────────
    if level == "Image level":
        st.markdown("""
        <div style="display:inline-block; background:#e8f4e8; border:1px solid #b8d8b8; border-radius:4px;
                    padding:0.25rem 0.75rem; font-size:0.82rem; color:#2d6a2d; margin-bottom:0.5rem;">
        📊 <strong>Image level</strong> — 1 dot = 1 product image &nbsp;|&nbsp; n = 914 images
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:inline-block; background:#e8eef8; border:1px solid #b8c8e8; border-radius:4px;
                    padding:0.25rem 0.75rem; font-size:0.82rem; color:#2d3d6a; margin-bottom:0.5rem;">
        📊 <strong>Region level</strong> — 1 dot = 1 bounding box &nbsp;|&nbsp; n = 8,447 boxes across all images
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if level == "Image level":
        # ── Image level ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#f7f5f0; border-left:3px solid #0d0d0d; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.88rem; line-height:1.6;">
        📌 <strong>What we are looking at:</strong> each dot represents one image.
        The X axis shows <strong>how much of the image is covered by text</strong>
        (0 = no text, 0.6 = 60% covered).
        The Y axis shows <strong>how much attention text captures compared to the product</strong>
        (values > 1 = text outcompetes the product, values < 1 = product wins).
        </div>
        """, unsafe_allow_html=True)

        data = df_valid[['Text_Area_Ratio', 'Attention_Ratio',
                         'Text_Count', 'Dominant_Quadrant', 'Image_ID']].dropna()
        r, p = stats.pearsonr(data['Text_Area_Ratio'], data['Attention_Ratio'])
        slope, intercept, *_ = stats.linregress(data['Text_Area_Ratio'], data['Attention_Ratio'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Pearson r", f"{r:.3f}")
        col2.metric("p-value", _fmt_pval(p))
        col3.metric("n images", len(data))

        st.caption(
            "⚠️ r = −0.197 is a **weak** correlation, but with p < 0.001 across 914 images "
            "it is statistically robust — not a coincidence. The trend exists, but other factors "
            "(position, text type, contrast) influence attention more than size alone."
        )

        fig = go.Figure()
        for q in ['Q1_TopLeft', 'Q2_TopRight', 'Q3_BottomLeft', 'Q4_BottomRight']:
            sub = data[data['Dominant_Quadrant'] == q]
            fig.add_trace(go.Scatter(
                x=sub['Text_Area_Ratio'], y=sub['Attention_Ratio'],
                mode='markers', name=QUAD_LABELS[q],
                marker=dict(color=QUAD_COLORS[q], size=6, opacity=0.55),
                hovertemplate=(
                    "<b>Image %{customdata[0]}</b><br>"
                    "Text Area Ratio: %{x:.3f}<br>"
                    "Attention Ratio: %{y:.2f}<br>"
                    "Text Count: %{customdata[1]}<extra></extra>"
                ),
                customdata=sub[['Image_ID', 'Text_Count']].values,
            ))
        x_line = np.linspace(data['Text_Area_Ratio'].min(), data['Text_Area_Ratio'].max(), 200)
        fig.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode='lines', name=f'OLS (r={r:.3f})',
            line=dict(color='#E85D04', width=2.5),
            hoverinfo='skip',
        ))
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family="Syne", height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title='Text Area Ratio (fraction of image covered by text)', showgrid=False),
            yaxis=dict(title='Attention Ratio', showgrid=True, gridcolor='#f0ece4'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "📖 **How to read this chart:** the orange line (OLS) shows the general trend — "
            "it slopes slightly downward, confirming that more text = less attention per element. "
            "Colors indicate the dominant quadrant of each image. No strong quadrant pattern emerges, "
            "suggesting the effect is general and does not depend on text position."
        )

        # ── Image-level takeaway ──────────────────────────────────────────────
        st.markdown("""
        <div style="background:#fdf3e7; border-left:4px solid #E85D04; padding:0.8rem 1rem; margin-top:1.2rem; font-size:0.87rem; line-height:1.65;">
        <strong>🔍 Takeaway — Image level:</strong><br>
        Adding more text to an ad does <em>not</em> make the text more compelling to viewers.
        On the contrary, higher text coverage is slightly associated with <em>less</em> relative attention going to text (r = −0.197).
        This is consistent with <strong>Banner Blindness</strong>: when text fills too much of the image,
        viewers start ignoring it and redirect gaze to the product photo.
        The effect is weak but statistically reliable across the full dataset (p &lt; 0.001).
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Region level ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#f7f5f0; border-left:3px solid #0d0d0d; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.88rem; line-height:1.6;">
        📌 <strong>What changes at region level:</strong> instead of one dot per image,
        here <strong>each dot is a single bounding box</strong> (8,447 total).
        We ask: does a larger bounding box capture more attention than a smaller one?
        The answer depends on how you define "attention" — use the Y-axis toggle to explore the paradox.
        </div>
        """, unsafe_allow_html=True)

        # ── Paradox explainer ─────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#f0f4ff; border:1px solid #c8d4f0; border-radius:6px; padding:0.9rem 1.1rem; margin-bottom:1rem; font-size:0.86rem; line-height:1.65;">
        <strong>⚖️ The Size Paradox — two ways to measure "attention":</strong><br><br>
        <strong>Total saliency share (r = +0.71):</strong> a bigger box captures more of the image's total fixation mass —
        but only because it has more pixels. Think of it like a bucket: a bigger bucket collects more rain even if
        the rain density is the same everywhere.<br><br>
        <strong>Attention density per pixel (r ≈ +0.03):</strong> divide total saliency by box area and the
        correlation nearly vanishes. Large boxes are <em>not intrinsically more eye-catching per pixel</em>
        than small ones. <strong>Size alone does not make text more salient — it just accumulates more
        fixations by offering a larger target.</strong>
        </div>
        """, unsafe_allow_html=True)

        col_ctrl1, col_ctrl2 = st.columns([2, 2])
        with col_ctrl1:
            y_metric = st.selectbox(
                "Y-axis metric — choose what 'attention' means:",
                ["Region_Attention_Ratio (share of total saliency)",
                 "Region_Mean_Saliency (attention density per pixel)"],
                help="Share of total saliency (r=+0.71): bigger box → more total attention captured. "
                     "Density per pixel (r≈+0.03): bigger box is NOT more attention-grabbing per pixel. "
                     "This is the paradox — switch between the two to see it."
            )
        y_col = 'Region_Attention_Ratio' if 'share' in y_metric else 'Region_Mean_Saliency'
        y_label = ('Region Attention Ratio (share of total saliency)'
                   if y_col == 'Region_Attention_Ratio'
                   else 'Region Mean Saliency (avg per pixel)')

        with col_ctrl2:
            quad_filter = st.multiselect(
                "Filter by Quadrant",
                options=list(QUAD_LABELS.keys())[:-1],
                default=list(QUAD_LABELS.keys())[:-1],
                format_func=lambda x: QUAD_LABELS[x],
            )

        data_r = df_r[df_r['Quadrant'].isin(quad_filter)][
            ['Region_Area_Ratio', 'Region_Attention_Ratio', 'Region_Mean_Saliency',
             'Quadrant', 'Image_ID', 'Region_Index']
        ].dropna()

        cap_x = data_r['Region_Area_Ratio'].quantile(0.99)
        cap_y = data_r[y_col].quantile(0.99)
        data_plot = data_r[(data_r['Region_Area_Ratio'] <= cap_x) & (data_r[y_col] <= cap_y)]

        r, p = stats.pearsonr(data_r['Region_Area_Ratio'], data_r[y_col])
        slope, intercept, *_ = stats.linregress(data_r['Region_Area_Ratio'], data_r[y_col])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pearson r", f"{r:.3f}")
        col2.metric("p-value", _fmt_pval(p))
        col3.metric("n bboxes", len(data_r))
        col4.metric("Plot capped at", "99th pct")

        if p == 0.0:
            st.caption(
                "ℹ️ **p-value shown as < 1e-300**: the true p-value is so small (well below 10⁻³⁰⁰) "
                "that it underflows Python's 64-bit float representation and rounds to exactly 0. "
                "This is expected with n = 8,447 and r = 0.71 — the result is overwhelmingly significant."
            )

        if y_col == 'Region_Attention_Ratio':
            st.caption(
                "📊 **r = +0.71** — strong positive correlation: larger bounding boxes capture a bigger "
                "**share** of the image's total saliency. But this does not mean they are more attention-grabbing "
                "per pixel. Switch to Region_Mean_Saliency to see the other side of the paradox."
            )
        else:
            st.caption(
                "📊 **r ≈ +0.03** — nearly zero correlation: bounding box size has almost no relationship "
                "with attention **density** per pixel. A small box can be just as salient per pixel as a large one. "
                "Size captures more total attention simply because there are more pixels — not because it is intrinsically more effective."
            )

        fig = go.Figure()
        for q in quad_filter:
            sub = data_plot[data_plot['Quadrant'] == q]
            fig.add_trace(go.Scatter(
                x=sub['Region_Area_Ratio'], y=sub[y_col],
                mode='markers', name=QUAD_LABELS[q],
                marker=dict(color=QUAD_COLORS[q], size=4, opacity=0.35),
                hovertemplate=(
                    f"<b>Image %{{customdata[0]}} · BBox %{{customdata[1]}}</b><br>"
                    f"Area Ratio: %{{x:.4f}}<br>"
                    f"{y_label}: %{{y:.4f}}<extra></extra>"
                ),
                customdata=sub[['Image_ID', 'Region_Index']].values,
            ))
        x_line = np.linspace(data_r['Region_Area_Ratio'].min(), cap_x, 200)
        fig.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode='lines', name=f'OLS (r={r:.3f})',
            line=dict(color='#E85D04', width=2.5),
            hoverinfo='skip',
        ))
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family="Syne", height=440,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title='Region Area Ratio (bbox area / image area)', showgrid=False),
            yaxis=dict(title=y_label, showgrid=True, gridcolor='#f0ece4'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        if y_col == 'Region_Attention_Ratio':
            st.caption(
                "📖 **How to read this chart:** each dot is one bounding box. "
                "The orange line slopes upward (r = +0.71) — bigger boxes do capture more total saliency. "
                "Plot is capped at the 99th percentile to remove extreme outliers and improve readability. "
                "Now switch Y-axis to Region_Mean_Saliency to see the paradox."
            )
        else:
            st.caption(
                "📖 **How to read this chart:** the orange line is nearly flat (r ≈ +0.03) — "
                "bounding box size tells us almost nothing about attention per pixel. "
                "The cloud of dots is evenly spread regardless of box size. "
                "This is the core finding: size captures more attention in absolute terms, but not in efficiency terms."
            )

        # ── Region-level takeaway ─────────────────────────────────────────────
        st.markdown("""
        <div style="background:#fdf3e7; border-left:4px solid #E85D04; padding:0.8rem 1rem; margin-top:1.2rem; font-size:0.87rem; line-height:1.65;">
        <strong>🔍 Takeaway — Region level:</strong><br>
        Making a text element bigger <em>does</em> accumulate more total fixations (r = +0.71),
        but only as a side-effect of having more pixels to land on — not because it is intrinsically
        more compelling. Attention density per pixel is unaffected by size (r ≈ +0.03).
        The practical implication: <strong>if you want to maximise attention <em>efficiency</em>,
        size is not the lever to pull</strong> — contrast, position, and message clarity matter more.
        </div>
        """, unsafe_allow_html=True)