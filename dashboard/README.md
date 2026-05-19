# Visual Attention Economy — Interactive Dashboard

Streamlit dashboard for Deliverable 3 of the Data Visualization project.

---

## Project structure

```
dashboard/
├── app.py                  # Main entry point & navigation
├── pages/
│   ├── overview.py         # KPIs and dataset summary
│   ├── banner_blindness.py # Text Count vs Clutter Index
│   ├── spatial.py          # Quadrant analysis
│   ├── size_attention.py   # Region/image size vs attention
│   └── image_explorer.py  # Per-image viewer with bboxes
├── utils/
│   └── data.py             # Data loading & path utilities
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

## Notes

- The dashboard works **without images** (Overview, Banner Blindness, Spatial, Size vs Attention pages). Only the Image Explorer requires the image files.
- Images are mounted as **read-only volumes** — the container never modifies your data.
- `@st.cache_data` ensures CSV files are loaded only once per session.
