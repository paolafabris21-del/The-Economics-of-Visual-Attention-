# The Economics of Visual Attention — Interactive Dashboard

Streamlit dashboard for **Deliverable 3** of the Data Visualization course, University of Trento.

Explores how text elements in e-commerce product images compete with the product itself for user attention, drawing on eye-tracking data from 972 real images (E-Commercial Dataset · Jiang et al., CVPR 2022).

---

## Project structure

```
dashboard/
├── app.py                  # Entry point: sidebar navigation and global CSS
├── pages/
│   ├── overview.py         # Hero section, KPIs, 3 key findings, summary charts
│   ├── banner_blindness.py # Text Count vs Clutter Index and Attention Ratio
│   ├── spatial.py          # Quadrant analysis: where do eyes actually go?
│   ├── size_attention.py   # Text area vs attention captured
│   └── image_explorer.py  # Per-image viewer: stimulus, saliency map, bboxes
├── utils/
│   └── data.py             # Cached CSV loading, image path helper
├── data/                   # ← PUT CSV FILES HERE
│   ├── metrics_global_v3.csv
│   └── metrics_regions_v2.csv
├── images/                 # ← PUT IMAGE FOLDERS HERE
│   ├── ALLSTIMULI/         # 1.jpg, 2.jpg, ... 972.jpg
│   └── ALLFIXATIONMAPS/    # 1_fixMap.jpg, 1_fixPts.jpg, ...
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Dashboard pages

### 🏠 Overview
Landing page with narrative context, top-level KPIs, and the three key findings of the project:
- **Banner Blindness** — more text elements → less attention per element (r = −0.49)
- **Top-Left is prime real estate** — Q1 achieves the highest median saliency (53.4) despite being the least used quadrant by designers
- **Bigger isn't proportionally better** — larger elements capture more total saliency but not more saliency per pixel

Also includes two summary charts: placement vs. attention by quadrant, and the Attention Ratio distribution across all images.

### 📉 Banner Blindness
Quantitative analysis of the phenomenon: how much does each text element lose in effectiveness as the number of texts in the same image grows?

- Interactive slider on Text Count and filter by dominant quadrant
- **Chart 1:** Median Clutter Index by Text Count range (bars + trend line + banner blindness threshold ~7)
- **Chart 2:** Text Area Ratio vs Attention Ratio per image (scatter coloured by range)
- **Chart 3:** Attention Ratio trend by range with IQR band
- Live KPIs: Pearson r, p-value, and mean Clutter Index updated on every filter change

### 🗺 Spatial Effectiveness
Comparison of attention captured across the four screen quadrants (Top-Left, Top-Right, Bottom-Left, Bottom-Right).

- Analysis at **image level** (dominant quadrant) and **region level** (per individual bounding box)
- Kruskal-Wallis H = 85.2, p = 2.3×10⁻¹⁸: differences across quadrants are highly significant
- Q1 Top-Left: median saliency 53.4 vs Q4 Bottom-Right: 29.7 — even though Q4 is the most used by designers (n = 287)

### 📐 Size vs. Attention
Answers the question: does making a text element larger capture more attention?

- **Image level:** Text Area Ratio vs Attention Ratio (r = −0.197, consistent with Banner Blindness)
- **Region level:** individual bbox area vs total saliency (r = +0.71) vs saliency per pixel (r ≈ +0.03)
- Conclusion: size increases absolute saliency, but not attention density

### 🖼 Image Explorer
Per-image viewer with three sections:
1. **Filters & selection** — filter by Text Count and dominant quadrant, sort results, pick an image
2. **Image metrics** — Text Count, Attention Ratio, Clutter Index, dominant quadrant
3. **Visualisation** — stimulus image, saliency map (fixMap), fix points, and overlay of text bounding boxes coloured by quadrant

Requires image files in `dashboard/images/`.

---

## Setup — Option A: Docker (recommended for submission)

### 1. Copy the data files

```bash
# CSV files
cp /path/to/metrics_global_v3.csv  dashboard/data/
cp /path/to/metrics_regions_v2.csv dashboard/data/

# Image folders (copy from your ECdata folder)
cp -r /path/to/ECdata/ALLSTIMULI        dashboard/images/
cp -r /path/to/ECdata/ALLFIXATIONMAPS   dashboard/images/
```

### 2. Build and run

```bash
cd dashboard
docker compose up --build
```

### 3. Open in browser

```
http://localhost:8501
```

---

## Setup — Option B: Local Python (for development)

```bash
cd dashboard
pip install -r requirements.txt

# Set env variables (or edit utils/data.py directly)
export DATA_DIR=./data
export IMAGES_DIR=./images

streamlit run app.py
```

---

## Dependencies

| Library | Version |
|---|---|
| streamlit | 1.35.0 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| plotly | 5.22.0 |
| scipy | 1.13.1 |
| Pillow | 10.3.0 |
| opencv-python-headless | 4.9.0.80 |

---

## Notes

- The dashboard works **without images** for the Overview, Banner Blindness, Spatial, and Size vs Attention pages. Only the Image Explorer requires image files.
- Images are mounted as **read-only volumes** — the container never modifies your original data.
- `@st.cache_data` ensures CSV files are loaded only once per session.
- Navigation is handled entirely by `app.py` via the sidebar radio; Streamlit's automatic pages/ navigation is disabled (`[data-testid="stSidebarNav"]` hidden via CSS).

---

## Dataset

**E-Commercial Dataset** · Jiang et al., CVPR 2022  
972 e-commerce product images · 720×720 px · 8,447 text bounding boxes · Eye-tracking saliency maps
