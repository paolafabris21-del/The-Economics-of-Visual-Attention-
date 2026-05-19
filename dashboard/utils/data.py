import streamlit as st
import pandas as pd
import numpy as np
import os

DATA_DIR = os.environ.get("DATA_DIR", "./data")

@st.cache_data
def load_global():
    path = os.path.join(DATA_DIR, "metrics_global_v3.csv")
    df = pd.read_csv(path)
    df['Image_ID'] = df['Image_ID'].astype(int)
    return df

@st.cache_data
def load_regions():
    path = os.path.join(DATA_DIR, "metrics_regions_v2.csv")
    df = pd.read_csv(path)
    df['Image_ID'] = df['Image_ID'].astype(int)
    return df

def get_image_path(image_id: int, kind: str = "stimulus") -> str | None:
    """
    kind: 'stimulus' | 'fixmap' | 'fixpts'
    Returns path if file exists, else None.
    """
    images_dir = os.environ.get("IMAGES_DIR", "./images")
    stimuli_dir = os.path.join(images_dir, "ALLSTIMULI")
    fixmap_dir  = os.path.join(images_dir, "ALLFIXATIONMAPS")

    if kind == "stimulus":
        p = os.path.join(stimuli_dir, f"{image_id}.jpg")
    elif kind == "fixmap":
        p = os.path.join(fixmap_dir, f"{image_id}_fixMap.jpg")
    elif kind == "fixpts":
        p = os.path.join(fixmap_dir, f"{image_id}_fixPts.jpg")
    else:
        return None

    return p if os.path.exists(p) else None

QUAD_COLORS = {
    "Q1_TopLeft":     "#4A90D9",
    "Q2_TopRight":    "#5BA4A4",
    "Q3_BottomLeft":  "#6EC6A0",
    "Q4_BottomRight": "#A8D8B9",
    "NoText":         "#CCCCCC",
}

QUAD_LABELS = {
    "Q1_TopLeft":     "Q1 Top-Left",
    "Q2_TopRight":    "Q2 Top-Right",
    "Q3_BottomLeft":  "Q3 Bottom-Left",
    "Q4_BottomRight": "Q4 Bottom-Right",
    "NoText":         "No Text",
}
