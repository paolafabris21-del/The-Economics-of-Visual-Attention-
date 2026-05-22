import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.data import load_global, load_regions, QUAD_COLORS, QUAD_LABELS


def show():
    df = load_global()
    df_r = load_regions()
    df_valid = df[df['Dominant_Quadrant'] != 'NoText'].copy()

    # ── Hero Section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 2.5rem 0 1.5rem 0;">
        <p style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase;
                  color:#888; margin-bottom:0.6rem;">
            Data Visualization Project · University of Trento
        </p>
        <h1 style="font-size:2.8rem; font-weight:800; letter-spacing:-0.03em;
                   line-height:1.1; margin:0 0 1rem 0; color:#0d0d0d;">
            The Economics of<br>Visual Attention
        </h1>
        <p style="font-size:1.05rem; color:#444; max-width:700px; line-height:1.75;
                  margin:0 0 1rem 0;">
            When users browse e-commerce product pages, their eyes don't follow the
            designer's intentions — they follow invisible rules. This dashboard explores
            <strong>how text elements compete with product visuals for user attention</strong>,
            drawing on eye-tracking data from 972 real e-commerce images.
        </p>
        <p style="font-size:0.9rem; color:#777; max-width:700px; line-height:1.65;">
            Using saliency maps from the <em>E-Commercial Dataset</em> (Jiang et al., CVPR 2022),
            we measured how placement, quantity, and size of text bounding boxes affect
            how much attention each element actually captures. The findings have direct
            implications for UX and visual design in digital retail.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    pct_text_wins = (df_valid['Attention_Ratio'] > 1).mean() * 100
    avg_text_count = df_valid['Text_Count'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images Analyzed", "972")
    c2.metric("Text Elements", "8,447")
    c3.metric("Avg. Text per Image", f"{avg_text_count:.1f}")
    c4.metric("Images Where Text Beats Product", f"{pct_text_wins:.0f}%")

    st.caption(
        "Metrics based on the 914 images that contain at least one text element. "
        "The remaining 58 images (6%) contain no text bounding boxes."
    )

    st.markdown("---")

    # ── Three Key Findings ────────────────────────────────────────────────────
    st.markdown("### Key Findings")
    st.markdown(
        '<p style="color:#888; font-size:0.9rem; margin-top:-0.4rem; margin-bottom:1.5rem;">'
        'Three evidence-based insights for UX and visual design in e-commerce'
        '</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="insight-box">
            <div style="font-size:1.6rem; margin-bottom:0.6rem;">📉</div>
            <strong>Banner Blindness Is Real</strong><br><br>
            The more text elements an image contains, the less attention each one receives.
            Crowding a product image with labels, prices, and CTAs is a losing strategy —
            users progressively tune everything out.
            <br><br>
            <span style="color:#aaa; font-size:0.78rem;">Pearson r = −0.49 &nbsp;·&nbsp; p &lt; 0.001</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
            <div style="font-size:1.6rem; margin-bottom:0.6rem;">👁</div>
            <strong>Top-Left Is Prime Real Estate</strong><br><br>
            Text placed in the top-left corner (Q1) consistently attracts the highest attention —
            yet most designers cluster their text in the bottom-right. The most valuable
            position on the canvas is systematically underused.
            <br><br>
            <span style="color:#aaa; font-size:0.78rem;">Median saliency Q1: 53.4 &nbsp;·&nbsp; p = 2.3×10⁻¹⁸</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insight-box">
            <div style="font-size:1.6rem; margin-bottom:0.6rem;">📐</div>
            <strong>Bigger Isn't Proportionally Better</strong><br><br>
            Larger text elements capture more total attention — but not more per pixel.
            A full-width banner and a compact label are equally "dense" in how well
            they hold the viewer's gaze. Size alone doesn't buy quality of attention.
            <br><br>
            <span style="color:#aaa; font-size:0.78rem;">Total saliency r = +0.71 &nbsp;·&nbsp; Density r ≈ +0.03</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    # ── Chart 1: Placement vs Attention per Quadrant ──────────────────────────
    with col_a:
        st.markdown("##### Where Designers Put Text vs. Where Eyes Actually Go")
        st.caption(
            "Designers heavily favor the bottom-right corner — but eye-tracking shows "
            "the top-left consistently wins more attention. The gap between each pair "
            "of bars reveals a missed opportunity."
        )

        quad_order = ["Q1_TopLeft", "Q2_TopRight", "Q3_BottomLeft", "Q4_BottomRight"]
        quad_labels_short = ["Top-Left (Q1)", "Top-Right (Q2)", "Bottom-Left (Q3)", "Bottom-Right (Q4)"]

        # Placement: share of individual text elements per quadrant
        placement_counts = df_r['Quadrant'].value_counts().reindex(quad_order).fillna(0)
        placement_pct = (placement_counts / placement_counts.sum() * 100).values

        # Attention: median saliency per quadrant, normalized to sum to 100
        sal_medians = df_r.groupby('Quadrant')['Region_Mean_Saliency'].median().reindex(quad_order).fillna(0)
        sal_pct = (sal_medians / sal_medians.sum() * 100).values

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            name="Where text is placed",
            x=quad_labels_short,
            y=placement_pct,
            marker_color="#d6d0c4",
            marker_line_color="#0d0d0d",
            marker_line_width=1,
        ))
        fig1.add_trace(go.Bar(
            name="Attention actually captured",
            x=quad_labels_short,
            y=sal_pct,
            marker_color="#4A90D9",
        ))
        fig1.update_layout(
            barmode='group',
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=0, r=0, t=10, b=0),
            font_family="Syne",
            height=320,
            legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True, gridcolor='#f0ece4',
                title='Share of total (%)',
                ticksuffix='%'
            ),
        )
        # Annotate the Q1 attention bar
        fig1.add_annotation(
            x="Top-Left (Q1)",
            y=sal_pct[0],
            text="Highest<br>attention",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#E85D04",
            font=dict(color="#E85D04", size=10),
            ax=50, ay=-35,
        )
        # Annotate the Q4 placement bar
        fig1.add_annotation(
            x="Bottom-Right (Q4)",
            y=placement_pct[3],
            text="Most used<br>by designers",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#888",
            font=dict(color="#555", size=10),
            ax=-55, ay=-35,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Attention Ratio distribution ─────────────────────────────────
    with col_b:
        st.markdown("##### Does Text Outcompete the Product?")
        st.caption(
            "The Attention Ratio compares how much the eye is drawn to text vs. the product image itself. "
            "A ratio above 1 means text is winning the battle for attention — and in most images, it does. "
            "The dashed orange vertical line marks the threshold of 1: images to its right have text that "
            "outcompetes the product; images to the left have a product that draws more gaze than the text."
        )

        fig2 = px.histogram(
            df_valid.dropna(subset=['Attention_Ratio']),
            x='Attention_Ratio',
            nbins=50,
            color_discrete_sequence=["#4A90D9"],
            labels={
                'Attention_Ratio': 'Attention Ratio (text vs. product background)',
                'count': 'Number of Images',
            },
        )
        fig2.add_vline(
            x=1, line_dash="dash", line_color="#E85D04", line_width=2,
        )
        fig2.add_annotation(
            x=8,
            xref="x", yref="paper",
            y=0.82,
            text=f"<b>{pct_text_wins:.0f}%</b> of images:<br>text beats the product",
            showarrow=False,
            font=dict(color="#0d0d0d", size=12),
            bgcolor="#f0ece4",
            bordercolor="#d6d0c4",
            borderwidth=1,
            borderpad=8,
        )
        fig2.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=0, r=0, t=10, b=0),
            font_family="Syne",
            height=320,
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0ece4', title='Number of Images'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Clutter Effect ───────────────────────────────────────────────
    st.markdown("##### More Text = Less Attention per Element")
    st.caption(
        "Each dot is one e-commerce image. As the number of text elements grows, the average "
        "attention each one receives drops sharply. This is the banner blindness effect in action: "
        "users don't read every element — they stop reading altogether. "
        "Fewer, more intentional elements perform better."
    )

    df_scatter = df_valid.dropna(subset=['Text_Count', 'Avg_Text_Saliency']).copy()
    df_scatter = df_scatter[df_scatter['Text_Count'] <= 35]  # remove extreme outliers

    # Compute binned trend manually (no statsmodels needed)
    bins = np.arange(1, df_scatter['Text_Count'].max() + 2)
    df_scatter['bin'] = pd.cut(df_scatter['Text_Count'], bins=bins, labels=bins[:-1])
    trend = df_scatter.groupby('bin', observed=True)['Avg_Text_Saliency'].mean().dropna()
    trend_x = trend.index.astype(int).tolist()
    trend_y = trend.values.tolist()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_scatter['Text_Count'],
        y=df_scatter['Avg_Text_Saliency'],
        mode='markers',
        marker=dict(color="#4A90D9", size=5, opacity=0.3),
        name='Image',
        hovertemplate=(
            "Text elements: %{x}<br>"
            "Avg attention: %{y:.1f}<extra></extra>"
        ),
    ))
    fig3.add_trace(go.Scatter(
        x=trend_x,
        y=trend_y,
        mode='lines',
        line=dict(color="#E85D04", width=2.5),
        name='Average per count',
    ))
    fig3.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        text="<b>Orange line</b> = average attention per text count",
        showarrow=False,
        font=dict(color="#888", size=11),
        xanchor="right",
    )
    fig3.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=0, r=0, t=10, b=0),
        font_family="Syne",
        height=280,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0ece4'),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.78rem; color:#aaa; text-align:center;">'
        'Dataset: E-Commercial Dataset · Jiang et al., CVPR 2022 &nbsp;·&nbsp; '
        '972 images · 720×720 px · Eye-tracking saliency maps'
        '</p>',
        unsafe_allow_html=True
    )
