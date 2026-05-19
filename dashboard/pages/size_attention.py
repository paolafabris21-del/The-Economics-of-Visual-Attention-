import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from utils.data import load_global, load_regions, QUAD_COLORS, QUAD_LABELS

def show():
    df   = load_global()
    df_r = load_regions()
    df_valid = df[df['Dominant_Quadrant'] != 'NoText'].copy()

    st.markdown("# Size vs. Attention")
    st.markdown("#### Does making a text element larger capture more attention?")
    st.markdown("""
    <div class="insight-box">
    The answer depends on the level of analysis.
    At <strong>image level</strong>: r = −0.197 (more text area → lower Attention_Ratio, consistent with Banner Blindness).
    At <strong>region level</strong>: r = +0.71 (larger individual bboxes do capture more total saliency).
    But r ≈ +0.03 for attention <em>density</em> — larger elements are not more salient per pixel.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    level = st.radio(
        "Analysis level",
        ["Image level", "Region level"],
        horizontal=True,
    )

    st.markdown("---")

    if level == "Image level":
        # ── Image level: Text_Area_Ratio vs Attention_Ratio ───────────────────
        data = df_valid[['Text_Area_Ratio', 'Attention_Ratio',
                         'Text_Count', 'Dominant_Quadrant', 'Image_ID']].dropna()
        r, p = stats.pearsonr(data['Text_Area_Ratio'], data['Attention_Ratio'])
        slope, intercept, *_ = stats.linregress(data['Text_Area_Ratio'], data['Attention_Ratio'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Pearson r", f"{r:.3f}")
        col2.metric("p-value", f"{p:.2e}")
        col3.metric("n images", len(data))

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
        st.caption("**Interpretation:** At image level, more text coverage is weakly associated with lower Attention_Ratio — consistent with Banner Blindness. More text area usually means more text elements, which dilutes per-element attention.")

    else:
        # ── Region level: Region_Area_Ratio vs Region_Attention_Ratio ────────
        col_ctrl1, col_ctrl2 = st.columns([2, 2])
        with col_ctrl1:
            y_metric = st.selectbox(
                "Y-axis metric",
                ["Region_Attention_Ratio (share of total saliency)",
                 "Region_Mean_Saliency (attention density per pixel)"],
            )
        y_col = 'Region_Attention_Ratio' if 'share' in y_metric else 'Region_Mean_Saliency'
        y_label = 'Region Attention Ratio (share of total saliency)' if y_col == 'Region_Attention_Ratio' else 'Region Mean Saliency (avg per pixel)'

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

        # Cap at 99th pct for readability
        cap_x = data_r['Region_Area_Ratio'].quantile(0.99)
        cap_y = data_r[y_col].quantile(0.99)
        data_plot = data_r[(data_r['Region_Area_Ratio'] <= cap_x) & (data_r[y_col] <= cap_y)]

        r, p = stats.pearsonr(data_r['Region_Area_Ratio'], data_r[y_col])
        slope, intercept, *_ = stats.linregress(data_r['Region_Area_Ratio'], data_r[y_col])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pearson r", f"{r:.3f}")
        col2.metric("p-value", f"{p:.2e}")
        col3.metric("n bboxes", len(data_r))
        col4.metric("Plot capped at", "99th pct")

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
            st.caption("**r = +0.71** — larger bboxes capture a bigger **share** of the image's total saliency. But switch to Region_Mean_Saliency to see that attention **density** per pixel is nearly uncorrelated with size (r ≈ +0.03).")
        else:
            st.caption("**r ≈ +0.03** — bounding box size has almost no relationship with **attention density** (mean saliency per pixel). Larger elements are not inherently more salient per pixel than smaller ones.")
