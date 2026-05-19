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

    st.markdown("# Visual Attention Economy")
    st.markdown("#### How text elements compete for user attention in e-commerce imagery")
    st.markdown("---")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images", "972")
    c2.metric("Bounding Boxes", "8,447")
    c3.metric("Avg Text Elements", f"{df_valid['Text_Count'].mean():.1f}")
    c4.metric("Avg Attention Ratio", f"{df_valid['Attention_Ratio'].mean():.2f}×")
    c5.metric("No-Text Images", "58 (6%)")

    st.markdown("---")

    # ── Three key findings ────────────────────────────────────────────────────
    st.markdown("### Three Central Findings")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="insight-box">
        <strong>Banner Blindness</strong><br>
        As text element count increases, average attention per element decreases.<br><br>
        r = −0.489 &nbsp;·&nbsp; p &lt; 0.001
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>Q1 Placement Advantage</strong><br>
        Top-Left bounding boxes attract the highest median saliency (53.4), despite Bottom-Right being the most common placement.<br><br>
        Kruskal-Wallis p = 2.3×10⁻¹⁸
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insight-box">
        <strong>Size vs Density Paradox</strong><br>
        Larger bboxes capture more total saliency (r = +0.71) but not higher attention density per pixel (r ≈ +0.03).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Two overview charts ───────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Text Count Distribution")
        fig = px.histogram(
            df[df['Text_Count'] > 0], x='Text_Count', nbins=30,
            color_discrete_sequence=["#0d0d0d"],
            labels={'Text_Count': 'Text Elements per Image', 'count': 'Images'},
        )
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=0, r=0, t=10, b=0),
            font_family="Syne",
            bargap=0.05,
            showlegend=False,
            xaxis=dict(showgrid=False, title='Text Elements per Image'),
            yaxis=dict(showgrid=True, gridcolor='#f0ece4', title='Images'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Dominant Quadrant Distribution")
        quad_counts = df['Dominant_Quadrant'].value_counts().reset_index()
        quad_counts.columns = ['Quadrant', 'Count']
        quad_counts['Label'] = quad_counts['Quadrant'].map(QUAD_LABELS)
        quad_counts['Color'] = quad_counts['Quadrant'].map(QUAD_COLORS)

        fig2 = go.Figure(go.Bar(
            x=quad_counts['Label'],
            y=quad_counts['Count'],
            marker_color=quad_counts['Color'],
            text=quad_counts['Count'],
            textposition='outside',
        ))
        fig2.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=0, r=0, t=10, b=0),
            font_family="Syne",
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0ece4', title='Images'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Attention Ratio distribution ──────────────────────────────────────────
    st.markdown("##### Attention Ratio Distribution (text images only)")
    st.caption("Values > 1 mean text attracts more attention per pixel than the product background.")

    fig3 = px.histogram(
        df_valid.dropna(subset=['Attention_Ratio']),
        x='Attention_Ratio', nbins=50,
        color_discrete_sequence=["#4A90D9"],
        labels={'Attention_Ratio': 'Attention Ratio', 'count': 'Images'},
    )
    # Add vertical line at 1
    fig3.add_vline(x=1, line_dash="dash", line_color="#E85D04",
                   annotation_text="Ratio = 1 (text = product)", annotation_position="top right")
    fig3.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=0, r=0, t=10, b=30),
        font_family="Syne", height=220,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0ece4'),
    )
    st.plotly_chart(fig3, use_container_width=True)
    pct_above_1 = (df_valid['Attention_Ratio'] > 1).mean() * 100
    st.caption(f"**{pct_above_1:.1f}%** of images have Attention Ratio > 1 — text outcompetes the product for attention.")
