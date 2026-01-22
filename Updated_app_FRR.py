
import time
from io import BytesIO
from collections import OrderedDict
import re

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ----------------------------- Page Config -----------------------------
st.set_page_config(page_title="AFRR Bids Explorer", layout="wide")
st.title("AFRR Auction Results Explorer")

# ----------------------------- Constants ------------------------------
PERIODS_4H = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']  # six 4h periods
DATASET = "ods125"
BASE_URL = "https://opendata.elia.be/api/records/1.0/search/"

# ----------------------------- Helpers -------------------------------

def to_float(series, default=0.0):
    """Safe numeric conversion to float with fallback default."""
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df

def coerce_selected(s: pd.Series) -> pd.Series:
    """Coerce selectedbyoptimizer to boolean robustly."""
    return s.map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))

def initial_ymax(v):
    """Initial y-max = rounded(max * 1.1), with safe lower bound."""
    if v <= 0:
        return 500
