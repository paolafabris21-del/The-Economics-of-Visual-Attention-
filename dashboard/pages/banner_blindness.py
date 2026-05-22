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
    <div style="
        font-size: 0.95rem;
        color: #5F6472;
        line-height: 1.65;
        max-width: 860px;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
        layout="wide"
    ">
        Banner blindness is a well-documented cognitive phenomenon: when a visual scene becomes
        crowded with text, the human eye stops processing each element individually and begins
        to ignore them as background noise. This section investigates whether that effect is
        measurable in our dataset. Using the <strong>Clutter Index</strong>
        (average saliency per text element) and the <strong>Attention Ratio</strong>
        (text attention vs. product attention), we test whether images with a higher number
        of text elements systematically receive less visual attention per element —
        and whether occupying more screen area is enough to compensate.
    </div>
    """, unsafe_allow_html=True)

    # ── Key concepts ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fff8f0; border:1px solid #f0e0c8; border-radius:6px; padding:1rem 1.2rem; margin:0.8rem 0 1rem 0; font-size:0.87rem; line-height:1.7;">
    <strong>📐 Key concepts for this page</strong><br><br>
    <table style="width:100%; border-collapse:collapse;">
      <tr>
        <td style="padding:0.3rem 0.8rem 0.3rem 0; width:33%; vertical-align:top;">
          <strong>Text Count</strong><br>
          <span style="color:#555;">The number of distinct text elements (bounding boxes) detected in a single product image.</span>
        </td>
        <td style="padding:0.3rem 0.8rem 0.3rem 0; width:33%; vertical-align:top;">
          <strong>Clutter Index</strong><br>
          <span style="color:#555;">Average visual saliency per text element. High = each element is clearly noticed; low = attention is spread too thin and elements fade into the background.</span>
        </td>
        <td style="padding:0.3rem 0 0.3rem 0; width:33%; vertical-align:top;">
          <strong>Attention Ratio</strong><br>
          <span style="color:#555;">Saliency captured by text ÷ saliency captured by the product. Values &gt; 1 mean text "won" the battle for gaze; values &lt; 1 mean the product photo dominated.</span>
        </td>
      </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([2, 2])
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

    # ── Filter data ───────────────────────────────────────────────────────────
    mask = (
        (df_valid['Text_Count'] >= text_range[0]) &
        (df_valid['Text_Count'] <= text_range[1]) &
        (df_valid['Dominant_Quadrant'].isin(quad_filter))
    )
    filtered = df_valid[mask].dropna(subset=['Text_Count', 'Clutter_Index'])
    if filtered.empty:
        st.warning("No images match the selected filters. Please adjust the filters.")
        return

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

    # ── Suggested explorations ────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#f0faf0; border:1px solid #b8d8b8; border-radius:6px; padding:0.75rem 1.1rem; margin-bottom:0.5rem; font-size:0.84rem; line-height:1.65;">
    💡 <strong>Suggested explorations:</strong><br>
    &nbsp;&nbsp;• Set <em>Text Count</em> to <strong>1–5</strong> → see how attention is high when few elements compete<br>
    &nbsp;&nbsp;• Set <em>Text Count</em> to <strong>15–53</strong> → watch the Clutter Index collapse below the banner blindness threshold<br>
    &nbsp;&nbsp;• Filter by <strong>Q1 Top-Left</strong> only → check whether a better screen position can counteract clutter<br>
    &nbsp;&nbsp;• Compare <strong>Q1 vs Q4</strong> to see if placement modifies the banner blindness effect
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # ── Main scatter ──────────────────────────────────────────────────────────
# ── Palette & bin setup ──────────────────────────────────────────────────
    BINS       = [1, 3, 6, 10, 15, 54]
    BIN_LABELS = ["1–2", "3–5", "6–9", "10–14", "15+"]
    BIN_COLORS = ["#C1440E", "#D85A30", "#E89B6A", "#378ADD", "#185FA5"]
    SWATCH     = ["#C1440E", "#D85A30", "#E89B6A", "#378ADD", "#185FA5"]
    THRESHOLD  = 7.0   # banner-blindness critical line

    # ── Shared bin computation ───────────────────────────────────────────────
    filtered_bins = filtered.copy()
    filtered_bins["bin"] = pd.cut(
        filtered_bins["Text_Count"],
        bins=BINS, labels=BIN_LABELS, right=False,
    )

    bin_stats = (
        filtered_bins
        .groupby("bin", observed=True)
        .agg(
            median_ci  = ("Clutter_Index",   "median"),
            q1_ci      = ("Clutter_Index",   lambda s: s.quantile(0.25)),
            q3_ci      = ("Clutter_Index",   lambda s: s.quantile(0.75)),
            median_ar  = ("Attention_Ratio", "median"),
            q1_ar      = ("Attention_Ratio", lambda s: s.quantile(0.25)),
            q3_ar      = ("Attention_Ratio", lambda s: s.quantile(0.75)),
            n          = ("Clutter_Index",   "count"),
        )
        .reset_index()
    )
    bin_stats["bin"]    = bin_stats["bin"].astype(str)
    ref_ci = bin_stats["median_ci"].iloc[0]
    ref_ar = bin_stats["median_ar"].iloc[0]
    bin_stats["pct_ci"] = ((bin_stats["median_ci"] - ref_ci) / ref_ci * 100).round(1)
    bin_stats["pct_ar"] = ((bin_stats["median_ar"] - ref_ar) / ref_ar * 100).round(1)

    # ════════════════════════════════════════════════════════════════════════
    # CHART 1 — Binned bar + IQR + trend  (full width)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:#f7f5f0; border-left:3px solid #0d0d0d; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.88rem; line-height:1.6;">
    📌 <strong>What you're looking at:</strong> images are grouped into five text-count ranges (1–2, 3–5, 6–9, 10–14, 15+).
    Each bar shows the <strong>median Clutter Index</strong> for that group — how much attention each text element receives on average.
    The shaded band is the interquartile range (middle 50% of images), and the dashed orange trend line connects the medians.
    The red dashed line marks the <strong>banner blindness threshold (~7)</strong>: below it, individual elements are practically invisible.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Text Count vs Clutter Index")

    fig_main = go.Figure()

    # IQR band
    fig_main.add_trace(go.Scatter(
        x   = BIN_LABELS + BIN_LABELS[::-1],
        y   = list(bin_stats["q3_ci"]) + list(bin_stats["q1_ci"])[::-1],
        fill      = "toself",
        fillcolor = "rgba(180,180,180,0.18)",
        line      = dict(color="rgba(0,0,0,0)"),
        hoverinfo = "skip",
        showlegend= True,
        name      = "IQR (Q1–Q3)",
    ))

    # Bars
    fig_main.add_trace(go.Bar(
        x             = BIN_LABELS,
        y             = bin_stats["median_ci"],
        marker_color  = BIN_COLORS,
        marker_line_width = 0,
        name          = "Median Clutter Index",
        text          = [f"{v:.1f}" for v in bin_stats["median_ci"]],
        textposition  = "outside",
        textfont      = dict(size=12, color="#5F6472"),
        hovertemplate = (
            "<b>Range:</b> %{x} elements<br>"
            "<b>Median CI:</b> %{y:.2f}<br>"
            "<b>Images:</b> %{customdata[0]}<br>"
            "<b>Δ vs 1–2 elements:</b> %{customdata[1]:.1f}%"
            "<extra></extra>"
        ),
        customdata = bin_stats[["n", "pct_ci"]].values,
    ))

    # Trend line
    fig_main.add_trace(go.Scatter(
        x    = BIN_LABELS,
        y    = bin_stats["median_ci"],
        mode = "lines+markers",
        name = "Median trend",
        line = dict(color="#E85D04", width=2.5, dash="dot"),
        marker = dict(size=8, color="white", line=dict(color="#E85D04", width=2.5)),
        hoverinfo = "skip",
    ))

    # Banner-blindness threshold
    fig_main.add_hline(
        y=THRESHOLD,
        line     = dict(color="#A32D2D", width=1.5, dash="dash"),
        annotation_text     = f"⚠ Banner blindness threshold (~{THRESHOLD:.0f})",
        annotation_position = "top right",
        annotation_font     = dict(size=11, color="#A32D2D"),
    )

    fig_main.update_layout(
        title = dict(
            text     = "   ",
            font     = dict(size=15, family="Syne"),
            x        = 0,
            xanchor  = "left",
            pad      = dict(b=10),
        ),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        font_family   = "Syne",
        height        = 420,
        margin        = dict(l=10, r=10, t=75, b=20),
        xaxis = dict(
            title    = "Text elements per image (range)",
            showgrid = False,
            zeroline = False,
        ),
        yaxis = dict(
            title     = "Median Clutter Index",
            showgrid  = True,
            gridcolor = "#f0ece4",
            zeroline  = False,
            range     = [0, bin_stats["q3_ci"].max() * 1.3],
        ),
        legend = dict(
            orientation="h", yanchor="bottom", y=1.06,
            xanchor="left", x=0, font=dict(size=11),
        ),
        bargap = 0.35,
    )

    st.plotly_chart(fig_main, use_container_width=True)

    st.markdown(f"""
    <div style="
        font-size:0.88rem; color:#5F6472; line-height:1.5;
        border-left: 3px solid #E85D04; padding-left: 12px;
        margin-top:-0.4rem; margin-bottom:1.2rem;
    ">
        The <strong>Clutter Index</strong> measures average visual saliency per text element
        (Avg_Text_Saliency / Text_Count). Bars show the median per range: with 1–2 elements
        the value reaches ~<strong>{bin_stats['median_ci'].iloc[0]:.0f}</strong>, while with
        15+ elements it drops to ~<strong>{bin_stats['median_ci'].iloc[-1]:.1f}</strong>
        (<strong>{abs(bin_stats['pct_ci'].iloc[-1]):.0f}% less</strong>).
        The dashed red line marks the critical threshold: below ~7, attention per element is
        so diluted that each text becomes practically invisible — the <em>banner blindness</em>
        effect. Sample: <strong>{len(filtered)} images</strong>
        (r = {r:.3f}, p = {p:.2e}).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #e8e4dc;margin:0.5rem 0 1.2rem'>",
                unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # CHART 2 + CHART 3  side-by-side
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:#f7f5f0; border-left:3px solid #0d0d0d; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.88rem; line-height:1.6;">
    📌 <strong>Two complementary views of the same effect:</strong>
    <strong>Chart 2</strong> (left) places each image as a dot — how much screen space text occupies vs. how much attention it actually wins.
    <strong>Chart 3</strong> (right) traces how the Attention Ratio evolves as text count grows, with the spread (IQR band) showing how consistent the pattern is.
    Read them together: Chart 2 shows the raw data, Chart 3 shows the trend.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # ── CHART 2 — Text_Area_Ratio vs Attention_Ratio scatter ────────────────
    with col_left:
        st.markdown("##### Text Area vs. Attention")

        scatter_df = filtered_bins.dropna(subset=["Text_Area_Ratio", "Attention_Ratio", "bin"])

        fig2 = go.Figure()

        # Parity reference line (AR = 1)
        fig2.add_hline(
            y=1,
            line     = dict(color="#A32D2D", width=1.5, dash="dash"),
            annotation_text     = "Text = Product (AR = 1)",
            annotation_position = "top right",
            annotation_font     = dict(size=10, color="#A32D2D"),
        )

        # One trace per bin
        for i, label in enumerate(BIN_LABELS):
            sub = scatter_df[scatter_df["bin"] == label]
            if sub.empty:
                continue
            fig2.add_trace(go.Scatter(
                x    = sub["Text_Area_Ratio"],
                y    = sub["Attention_Ratio"],
                mode = "markers",
                name = f"{label} elements",
                marker = dict(
                    color   = BIN_COLORS[i],
                    size    = 5,
                    opacity = 0.45,
                    line    = dict(width=0),
                ),
                hovertemplate = (
                    "<b>Image:</b> %{customdata[0]}<br>"
                    "Text Area Ratio: %{x:.3f}<br>"
                    "Attention Ratio: %{y:.3f}<br>"
                    "Text Count: %{customdata[1]}"
                    "<extra></extra>"
                ),
                customdata = sub[["Image_ID", "Text_Count"]].values,
            ))

        # Bin medians as diamond markers
        bin_area = (
            scatter_df.groupby("bin", observed=True)
            .agg(med_x=("Text_Area_Ratio", "median"), med_y=("Attention_Ratio", "median"))
            .reset_index()
        )
        fig2.add_trace(go.Scatter(
            x    = bin_area["med_x"],
            y    = bin_area["med_y"],
            mode = "markers+text",
            name = "Bin median",
            marker = dict(
                size   = 14,
                color  = BIN_COLORS[:len(bin_area)],
                symbol = "diamond",
                line   = dict(color="white", width=1.5),
            ),
            text         = bin_area["bin"].astype(str),
            textposition = "top center",
            textfont     = dict(size=9, color="#5F6472"),
            hovertemplate = (
                "<b>Bin:</b> %{text}<br>"
                "Median Text Area Ratio: %{x:.3f}<br>"
                "Median Attention Ratio: %{y:.3f}"
                "<extra></extra>"
            ),
        ))

        fig2.update_layout(
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            font_family   = "Syne",
            height        = 320,
            margin        = dict(l=10, r=10, t=20, b=20),
            xaxis = dict(
                title    = "Text Area Ratio (fraction of image covered by text)",
                showgrid = True, gridcolor="#f0ece4", zeroline=False,
            ),
            yaxis = dict(
                title    = "Attention Ratio (text vs. product)",
                showgrid = True, gridcolor="#f0ece4", zeroline=False,
            ),
            legend = dict(
                orientation="h", yanchor="bottom", y=1.04,
                xanchor="left", x=0, font=dict(size=10),
            ),
        )

        st.plotly_chart(fig2, use_container_width=True)

        above_1 = (scatter_df["Attention_Ratio"] > 1).mean() * 100
        st.markdown(f"""
        <div style="font-size:0.84rem; color:#6B7280; line-height:1.5; margin-top:-0.3rem;">
            Each dot is one image. The x-axis shows how much of the image area is occupied
            by text; the y-axis shows whether text outcompetes the product for attention
            (Attention Ratio = Avg_Text_Saliency / avg_product_saliency).
            The dashed red line marks parity (AR&nbsp;=&nbsp;1): points <strong>above</strong>
            it mean text wins, <strong>below</strong> means the product wins.
            Diamond markers show the median per text-count range. Only
            <strong>{above_1:.0f}%</strong> of images have text that outcompetes the
            product — suggesting that occupying more screen space does not reliably
            translate into more attention.
        </div>
        """, unsafe_allow_html=True)

    # ── CHART 3 — Attention Ratio per bin (area + dots) ─────────────────────
    with col_right:
        st.markdown("##### Attention Ratio per element by range")

        fig3 = go.Figure()

        # Shaded area under the curve
        fig3.add_trace(go.Scatter(
            x         = BIN_LABELS,
            y         = bin_stats["median_ar"],
            fill      = "tozeroy",
            fillcolor = "rgba(216, 90, 48, 0.12)",
            line      = dict(color="#D85A30", width=2.5),
            mode      = "lines",
            name      = "Median Attention Ratio",
            hoverinfo = "skip",
            showlegend= False,
        ))

        # IQR band for attention ratio
        fig3.add_trace(go.Scatter(
            x   = BIN_LABELS + BIN_LABELS[::-1],
            y   = list(bin_stats["q3_ar"]) + list(bin_stats["q1_ar"])[::-1],
            fill      = "toself",
            fillcolor = "rgba(216, 90, 48, 0.15)",
            line      = dict(color="rgba(0,0,0,0)"),
            hoverinfo = "skip",
            showlegend= True,
            name      = "IQR (Q1–Q3)",
        ))

        # Q3 boundary line
        fig3.add_trace(go.Scatter(
            x         = BIN_LABELS,
            y         = bin_stats["q3_ar"],
            mode      = "lines",
            line      = dict(color="#D85A30", width=1, dash="dot"),
            name      = "Q3",
            hovertemplate = "<b>Q3:</b> %{y:.2f}<extra></extra>",
        ))
 
        # Q1 boundary line
        fig3.add_trace(go.Scatter(
            x         = BIN_LABELS,
            y         = bin_stats["q1_ar"],
            mode      = "lines",
            line      = dict(color="#D85A30", width=1, dash="dot"),
            name      = "Q1",
            hovertemplate = "<b>Q1:</b> %{y:.2f}<extra></extra>",
        ))
 
        # Q3 label on last point
        fig3.add_annotation(
            x=BIN_LABELS[-1], y=bin_stats["q3_ar"].iloc[-1],
            text="Q3", showarrow=False,
            xanchor="left", xshift=6,
            font=dict(size=10, color="#D85A30"),
        )
 
        # Q1 label on last point
        fig3.add_annotation(
            x=BIN_LABELS[-1], y=bin_stats["q1_ar"].iloc[-1],
            text="Q1", showarrow=False,
            xanchor="left", xshift=6,
            font=dict(size=10, color="#D85A30"),
        )


        # Markers with value labels
        fig3.add_trace(go.Scatter(
            x    = BIN_LABELS,
            y    = bin_stats["median_ar"],
            mode = "markers+text",
            marker = dict(
                size  = 10,
                color = BIN_COLORS,
                line  = dict(color="white", width=2),
            ),
            text         = [f"{v:.2f}" for v in bin_stats["median_ar"]],
            textposition = "top center",
            textfont     = dict(size=10, color="#5F6472"),
            name         = "Median",
            hovertemplate = (
                "<b>Range:</b> %{x}<br>"
                "<b>Median AR:</b> %{y:.3f}<br>"
                "<b>Δ vs baseline:</b> %{customdata:.1f}%<extra></extra>"
            ),
            customdata = bin_stats["pct_ar"],
        ))

        fig3.update_layout(
            plot_bgcolor  = "white",
            paper_bgcolor = "white",
            font_family   = "Syne",
            height        = 320,
            margin        = dict(l=10, r=10, t=20, b=20),
            xaxis = dict(
                title    = "Text elements per image (range)",
                showgrid = False, zeroline=False,
            ),
            yaxis = dict(
                title     = "Median Attention Ratio",
                showgrid  = True, gridcolor="#f0ece4",
                zeroline  = False,
                range     = [0, bin_stats["q3_ar"].max() * 1.35],
            ),
            legend = dict(
                orientation="h", yanchor="bottom", y=1.04,
                xanchor="left", x=0, font=dict(size=10),
            ),
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(f"""
        <div style="font-size:0.84rem; color:#6B7280; line-height:1.5; margin-top:-0.3rem;">
            The <strong>Attention Ratio</strong> measures the share of visual attention
            captured by each text element relative to the entire scene. The shaded area shows
            how quickly this share shrinks: from
            <strong>{bin_stats['median_ar'].iloc[0]:.2f}</strong> with 1–2 elements down to
            <strong>{bin_stats['median_ar'].iloc[-1]:.2f}</strong> with 15+
            (<strong>{abs(bin_stats['pct_ar'].iloc[-1]):.0f}% less</strong>).
            The light band marks the interquartile range — even in the best cases (Q3),
            images with many text elements remain well below the lower ranges.
        </div>
        """, unsafe_allow_html=True)

    # ── Takeaway ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fdf3e7; border-left:4px solid #E85D04; padding:0.8rem 1rem; margin-top:1.5rem; font-size:0.87rem; line-height:1.65;">
    <strong>🔍 Takeaway — Banner Blindness:</strong><br>
    More text does not mean more communication. Each additional element competes with all the others
    for a finite pool of viewer attention — and beyond ~7 elements, the Clutter Index drops below the
    banner blindness threshold, meaning individual text becomes practically invisible.
    The Attention Ratio tells the same story from a different angle: images with 15+ elements see
    their per-element share of attention reduced by roughly <strong>{pct}%</strong> compared to
    images with just 1–2 elements.
    <br><br>
    <strong>For UX designers: fewer, bolder text elements consistently outperform crowded layouts.
    Prioritise one clear message per image.</strong>
    </div>
    """.format(pct=int(abs(bin_stats['pct_ar'].iloc[-1]))), unsafe_allow_html=True)
