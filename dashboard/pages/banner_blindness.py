import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
from utils.data import load_global, QUAD_COLORS, QUAD_LABELS

def show():
    df = load_global()
    df_valid = df[df['Dominant_Quadrant'] != 'NoText'].copy()

    st.markdown("# Banner Blindness")
    st.markdown("#### Does adding more text elements dilute attention per element?")
    st.markdown("""
    <div class="insight-box">
    The <strong>Clutter Index</strong> measures average saliency per text element
    (Avg_Text_Saliency / Text_Count). The Banner Blindness hypothesis predicts it decreases
    as Text_Count increases. &nbsp;·&nbsp; <strong>r = −0.489, p &lt; 0.001, n = 914</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        text_range = st.slider(
            "Filter: Text Count range",
            min_value=1, max_value=53,
            value=(1, 53), step=1
        )
    with col_ctrl2:
        quad_filter = st.multiselect(
            "Filter: Dominant Quadrant",
            options=list(QUAD_LABELS.keys())[:-1],
            default=list(QUAD_LABELS.keys())[:-1],
            format_func=lambda x: QUAD_LABELS[x],
        )
    with col_ctrl3:
        log_scale = st.toggle("Log scale (Y axis)", value=False)

    # ── Filter data ───────────────────────────────────────────────────────────
    mask = (
        (df_valid['Text_Count'] >= text_range[0]) &
        (df_valid['Text_Count'] <= text_range[1]) &
        (df_valid['Dominant_Quadrant'].isin(quad_filter))
    )
    filtered = df_valid[mask].dropna(subset=['Text_Count', 'Clutter_Index'])

    # Recompute r on filtered data
    if len(filtered) > 2:
        r, p = stats.pearsonr(filtered['Text_Count'], filtered['Clutter_Index'])
        slope, intercept, _, _, _ = stats.linregress(filtered['Text_Count'], filtered['Clutter_Index'])
    else:
        r, p, slope, intercept = 0, 1, 0, 0

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images in selection", len(filtered))
    c2.metric("Pearson r", f"{r:.3f}")
    c3.metric("p-value", f"{p:.2e}")
    c4.metric("Avg Clutter Index", f"{filtered['Clutter_Index'].mean():.2f}")

    st.markdown("---")

    # ── Main scatter ──────────────────────────────────────────────────────────
    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown("##### Text Count vs. Clutter Index")

        fig = go.Figure()

        # Points colored by quadrant
        for q in quad_filter:
            sub = filtered[filtered['Dominant_Quadrant'] == q]
            fig.add_trace(go.Scatter(
                x=sub['Text_Count'], y=sub['Clutter_Index'],
                mode='markers',
                name=QUAD_LABELS[q],
                marker=dict(color=QUAD_COLORS[q], size=6, opacity=0.55,
                            line=dict(width=0.5, color='white')),
                hovertemplate=(
                    "<b>Image %{customdata[0]}</b><br>"
                    "Text Count: %{x}<br>"
                    "Clutter Index: %{y:.2f}<br>"
                    "Attention Ratio: %{customdata[1]:.2f}<br>"
                    "<extra></extra>"
                ),
                customdata=sub[['Image_ID', 'Attention_Ratio']].values,
            ))

        # OLS trend line
        if len(filtered) > 2:
            x_range = np.linspace(filtered['Text_Count'].min(), filtered['Text_Count'].max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range, y=slope * x_range + intercept,
                mode='lines', name=f'OLS trend (r={r:.3f})',
                line=dict(color='#E85D04', width=2.5, dash='solid'),
                hoverinfo='skip',
            ))

        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family="Syne", height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title='Text Count (# text elements per image)', showgrid=False, zeroline=False),
            yaxis=dict(
                title='Clutter Index (avg saliency per element)',
                showgrid=True, gridcolor='#f0ece4', zeroline=False,
                type='log' if log_scale else 'linear',
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown("##### Clutter Index by Text Count bin")
        bins = [1, 3, 6, 10, 15, 54]
        labels = ['1–2', '3–5', '6–9', '10–14', '15+']
        filtered['bin'] = pd.cut(filtered['Text_Count'], bins=bins, labels=labels, right=False)
        bin_stats = filtered.groupby('bin', observed=True)['Clutter_Index'].median().reset_index()

        fig2 = go.Figure(go.Bar(
            x=bin_stats['Clutter_Index'], y=bin_stats['bin'],
            orientation='h',
            marker_color=['#4A90D9', '#5BA4A4', '#6EC6A0', '#A8D8B9', '#E85D04'],
            text=bin_stats['Clutter_Index'].round(1),
            textposition='outside',
        ))
        fig2.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family="Syne", height=420,
            margin=dict(l=0, r=20, t=10, b=0),
            xaxis=dict(title='Median Clutter Index', showgrid=True, gridcolor='#f0ece4'),
            yaxis=dict(title='Text Count range', showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Interpretation")
    direction = "confirms" if r < -0.3 else "weakly supports" if r < 0 else "does not support"
    st.markdown(f"""
    On the selected **{len(filtered)} images**, the correlation is **r = {r:.3f}**
    (p = {p:.2e}), which **{direction}** the Banner Blindness hypothesis.
    Images with 1–2 text elements show a median Clutter Index of
    **{bin_stats[bin_stats['bin']=='1–2']['Clutter_Index'].values[0]:.1f}** compared to
    **{bin_stats[bin_stats['bin']=='15+']['Clutter_Index'].values[0]:.1f}** for images with 15+ elements —
    a {'significant' if abs(r) > 0.3 else 'modest'} reduction in per-element attention.
    """)
