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

    st.markdown("---")

    # ── Level selector ────────────────────────────────────────────────────────
    level = st.radio(
        "Analysis level",
        ["Image level (Dominant Quadrant)", "Region level (per bounding box)"],
        horizontal=True,
    )

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    if "Image level" in level:
        # ── Image-level: boxplot of Attention_Ratio by dominant quadrant ──────
        with col_left:
            st.markdown("##### Attention Ratio by Dominant Quadrant")
            fig = go.Figure()
            for q, label in zip(QUAD_ORDER, NICE_LABELS):
                sub = df_valid[df_valid['Dominant_Quadrant'] == q]['Attention_Ratio'].dropna()
                fig.add_trace(go.Box(
                    y=sub, name=label.replace('\n', ' '),
                    marker_color=QUAD_COLORS[q],
                    line_color=QUAD_COLORS[q],
                    fillcolor=QUAD_COLORS[q],
                    opacity=0.6,
                    boxmean=True,
                    hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>",
                ))
            fig.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font_family="Syne", height=420,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(title='Attention Ratio', showgrid=True, gridcolor='#f0ece4'),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

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

            st.markdown("---")
            st.markdown("##### Quadrant frequency")
            freq = df['Dominant_Quadrant'].value_counts().reset_index()
            freq.columns = ['Quadrant', 'Count']
            freq = freq[freq['Quadrant'] != 'NoText']
            freq['Label'] = freq['Quadrant'].map(QUAD_LABELS)
            freq['Color'] = freq['Quadrant'].map(QUAD_COLORS)
            fig2 = go.Figure(go.Bar(
                x=freq['Label'], y=freq['Count'],
                marker_color=freq['Color'],
                text=freq['Count'], textposition='outside',
            ))
            fig2.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font_family="Syne", height=220,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f0ece4', title='Images'),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        # ── Region-level: violin + heatmap ────────────────────────────────────
        with col_left:
            st.markdown("##### Region Mean Saliency by Quadrant")
            fig = go.Figure()
            for q, label in zip(QUAD_ORDER, NICE_LABELS):
                sub = df_r[df_r['Quadrant'] == q]['Region_Mean_Saliency'].values
                fig.add_trace(go.Violin(
                    y=sub, name=label.replace('\n', ' '),
                    fillcolor='rgba({},{},{},0.73)'.format(int(QUAD_COLORS[q][1:3],16), int(QUAD_COLORS[q][3:5],16), int(QUAD_COLORS[q][5:7],16)),
                    line_color=QUAD_COLORS[q],
                    meanline_visible=True,
                    box_visible=True,
                    hovertemplate="<b>%{x}</b><br>%{y:.1f}<extra></extra>",
                ))
            kw_groups = [df_r[df_r['Quadrant']==q]['Region_Mean_Saliency'].values for q in QUAD_ORDER]
            kw_stat, kw_p = stats.kruskal(*kw_groups)
            fig.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font_family="Syne", height=420,
                margin=dict(l=0, r=0, t=40, b=0),
                title=dict(text=f"Kruskal-Wallis H={kw_stat:.1f}, p={kw_p:.2e}", font_size=12, x=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(title='Region Mean Saliency', showgrid=True, gridcolor='#f0ece4'),
                showlegend=False,
                violingap=0.15,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("##### Spatial canvas heatmap (4×4 grid)")
            st.caption("Mean Region_Mean_Saliency by bounding box centroid position")
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
                font_family="Syne", height=300,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Region-level stats table
            stats_rows = []
            for q in QUAD_ORDER:
                sub = df_r[df_r['Quadrant'] == q]['Region_Mean_Saliency']
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
