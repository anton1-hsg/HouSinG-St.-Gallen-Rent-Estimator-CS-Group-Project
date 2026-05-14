"""
St. Gallen Apartment Valuator · CS Group Project (HSG).

Estimates fair rent for homegate.ch listings via XGBoost and flags
over/undervalued ones. Users can rank by walking time to HSG, transit,
groceries, and gyms.

Data: 315 listings (stgallen_listings.csv); distances via OSM Overpass.
Model: XGBoost, 5-fold CV. Gap = (actual − predicted) / predicted × 100%.
       >+5% = overvalued, <−5% = undervalued.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict

from sklearn.metrics import r2_score, mean_absolute_error

# Set the page title, icon, and layout.
st.set_page_config(
    page_title="HouSinG",
    page_icon="house",
    layout="wide",
)

# App-wide CSS for fonts, colours, and component styling.
st.markdown("""
<style>
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* Hide sidebar collapse button */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
button[data-testid="baseButton-header"] { display: none !important; }
section[data-testid="stSidebar"] button[kind="header"] { display: none !important; }
.stSidebarCollapseButton { display: none !important; }
/* Fallback when Material Icons font fails */
[data-testid="stSidebar"] > div > div > div > button:first-of-type { display: none !important; }
button[aria-label="Collapse sidebar"] { display: none !important; }
button[aria-label="Close sidebar"] { display: none !important; }
header button[data-testid] { display: none !important; }

/* Global font */
html, body, [class*="css"], .stMarkdown, .stCaption, p, li, span, label, td, th {
    font-family: 'Inter', sans-serif !important;
}
h1, h2, h3, h4, h5, h6,
[data-testid="stMetricValue"],
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
.stTitle {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em !important;
}

/* Sidebar: forest-green */
[data-testid="stSidebar"] {
    background-color: #0d3321 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #e8f5e9 !important;
}
[data-testid="stSidebar"] .stMetric label,
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #a5d6a7 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2d6a42 !important;
}
/* Nav uses buttons (see below) */

/* Main content */
.main .block-container {
    background: #ffffff;
    padding-top: 2rem;
}

/* Headings */
h1, h2, h3, h4 {
    color: #0d3321 !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
h1 { font-size: 2rem !important; }

/* HR */
hr {
    border-color: #d4e8d8 !important;
    margin: 1.2rem 0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #f2f5f2;
    border-radius: 10px;
    padding: 12px 16px !important;
    border-left: 4px solid #1a5c2a;
}
[data-testid="stMetricValue"] {
    color: #0d3321 !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #4a7c5a !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Buttons */
.stButton > button {
    background-color: #1a5c2a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background-color: #0d3321 !important;
}

/* Sliders & selects */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #1a5c2a !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-highlighted] {
    background: #1a5c2a !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    border-bottom: 2px solid #d4e8d8 !important;
    gap: 4px;
}
[data-baseweb="tab"] {
    background: #f2f5f2 !important;
    border-radius: 6px 6px 0 0 !important;
    color: #4a7c5a !important;
    font-weight: 500 !important;
    border: none !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1a5c2a !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] thead tr th {
    background: #0d3321 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Expander */
[data-testid="stExpander"] > details > summary {
    background: #f2f5f2 !important;
    border-radius: 6px !important;
    color: #0d3321 !important;
    font-weight: 600 !important;
    border-left: 3px solid #1a5c2a !important;
}

/* Select tags */
[data-baseweb="tag"] {
    background: #1a5c2a !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Path to the CSV file containing the listings data.
CSV_PATH = os.path.join(os.path.dirname(__file__), "stgallen_listings.csv")

@st.cache_data
def load_data(csv_version=None):
    """Load the listings CSV; csv_version (file mtime) busts the cache on file change."""
    return pd.read_csv(CSV_PATH)

# Use file mtime as the cache key so data reloads automatically when the CSV changes.
_mtime = os.path.getmtime(CSV_PATH)

# XGBoost model for predicting gross monthly rent.
@st.cache_resource
def train_model(csv_version):
    """Train the main XGBoost model using a two-pass pipeline to remove outliers first.
    Returns model, importances, r2, mae, all_preds, outlier_mask."""
    df = load_data(csv_version=csv_version)
    df = df.copy()

    feature_cols = [
        "rooms", "living_space", "floor",
        "lat", "lng",
        "dist_transit_m", "dist_grocery_m", "dist_gym_m", "dist_hsg_m",
        "dist_center_m",
    ]
    feature_names = [
        "Rooms", "Living space (m²)", "Floor",
        "Latitude", "Longitude",
        "Dist. transit", "Dist. grocery", "Dist. gym", "Dist. HSG",
        "Dist. city centre",
    ]

    X_all = df[feature_cols].values
    y_all = df["rent_gross"].values

    # Hyperparameters pre-tuned via 80-iteration RandomizedSearchCV with 5-fold CV.
    _params = dict(
        n_estimators=800, max_depth=4, learning_rate=0.02,
        subsample=1.0, colsample_bytree=0.9, min_child_weight=2,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0,
        random_state=42, verbosity=0,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. First-pass CV → outlier mask
    _m0 = XGBRegressor(**_params)
    _p0 = cross_val_predict(_m0, X_all, y_all, cv=cv)
    _resid = np.abs(_p0 - y_all)
    _q25, _q75 = np.percentile(_resid, 25), np.percentile(_resid, 75)
    outlier_mask = _resid > (_q75 + 2.0 * (_q75 - _q25))

    # 2. CV on clean set
    X_clean = X_all[~outlier_mask]
    y_clean = y_all[~outlier_mask]
    model = XGBRegressor(**_params)
    y_pred_cv = cross_val_predict(model, X_clean, y_clean, cv=cv)

    # 3. Fit final model on clean set
    model.fit(X_clean, y_clean)

    # 4. Predict all rows (incl. outliers)
    all_preds = model.predict(X_all)

    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    r2  = r2_score(y_clean, y_pred_cv)
    mae = mean_absolute_error(y_clean, y_pred_cv)
    return model, importances, r2, mae, all_preds, outlier_mask


@st.cache_resource
def train_model_per_room(csv_version):
    """Train a second XGBoost model on per-room rent (rent / rooms) for the shared-flat toggle.
    Returns model_pr, all_preds_pr, r2_pr, mae_pr, outlier_mask."""
    _df = load_data(csv_version=csv_version).copy()
    _df["per_room_rent"] = _df["rent_gross"] / _df["rooms"].apply(lambda r: max(1.0, np.floor(r)))

    feature_cols = [
        "rooms", "living_space", "floor",
        "lat", "lng",
        "dist_transit_m", "dist_grocery_m", "dist_gym_m", "dist_hsg_m",
        "dist_center_m",
    ]
    X_all = _df[feature_cols].values
    y_all = _df["per_room_rent"].values

    _params = dict(
        n_estimators=800, max_depth=4, learning_rate=0.02,
        subsample=1.0, colsample_bytree=0.9, min_child_weight=2,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0,
        random_state=42, verbosity=0,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Outliers from per-room residuals
    _p0 = cross_val_predict(XGBRegressor(**_params), X_all, y_all, cv=cv)
    _resid = np.abs(_p0 - y_all)
    _q25, _q75 = np.percentile(_resid, 25), np.percentile(_resid, 75)
    outlier_mask = _resid > (_q75 + 2.0 * (_q75 - _q25))

    # 2. Clean CV + final fit
    X_clean, y_clean = X_all[~outlier_mask], y_all[~outlier_mask]
    model_pr = XGBRegressor(**_params)
    y_pred_cv = cross_val_predict(model_pr, X_clean, y_clean, cv=cv)
    model_pr.fit(X_clean, y_clean)
    all_preds_pr = model_pr.predict(X_all)

    r2_pr  = r2_score(y_clean, y_pred_cv)
    mae_pr = mean_absolute_error(y_clean, y_pred_cv)
    return model_pr, all_preds_pr, r2_pr, mae_pr, outlier_mask


df = load_data(csv_version=_mtime)
model,    importances, r2,    mae,    _all_preds,    _outlier_mask    = train_model(_mtime)
_, _all_preds_pr, _, _, _outlier_mask_pr = train_model_per_room(_mtime)

# Add predicted rent, valuation gap, and label columns to the dataframe.
df["predicted_rent"]  = _all_preds.round(0)
df["valuation_gap"]   = ((df["rent_gross"] - df["predicted_rent"])
                          / df["predicted_rent"] * 100).round(1)
df["valuation_label"] = df["valuation_gap"].apply(
    lambda g: "overvalued" if g > 5 else ("undervalued" if g < -5 else "fair")
)
df["is_outlier"] = _outlier_mask

# Add per-room rent and valuation gap columns for shared-flat mode.
df["per_room_rent"]           = (df["rent_gross"] / df["rooms"].apply(lambda r: max(1.0, np.floor(r)))).round(0)
df["predicted_per_room_rent"] = _all_preds_pr.round(0)
df["per_room_gap"]            = (
    (df["per_room_rent"] - df["predicted_per_room_rent"])
    / df["predicted_per_room_rent"] * 100
).round(1)
df["per_room_label"] = df["per_room_gap"].apply(
    lambda g: "expensive" if g > 5 else ("affordable" if g < -5 else "average")
)

# Classify each listing as Studio or Shared flat based on room count.
df["listing_type"] = df["rooms"].apply(
    lambda r: "Studio" if r < 2 else "Shared flat"
)

# Convert distances in metres to walking time in minutes.
WALK_SPEED = 80  # m/min ≈ 5 km/h
df["walk_hsg_min"]     = (df["dist_hsg_m"]     / WALK_SPEED).round(1)
df["walk_transit_min"] = (df["dist_transit_m"] / WALK_SPEED).round(1)
df["walk_grocery_min"] = (df["dist_grocery_m"] / WALK_SPEED).round(1)
df["walk_gym_min"]     = (df["dist_gym_m"]     / WALK_SPEED).round(1)
df["walk_center_min"]  = (df["dist_center_m"]  / WALK_SPEED).round(1)

# Sidebar CSS for navigation buttons and stat cards.
st.markdown("""
<style>
/* Sidebar shell */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f09 0%, #0a1f12 35%, #0d2a18 100%) !important;
    border-right: 1px solid rgba(76,175,80,0.12) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* Sidebar nav buttons (liquid-glass) */
[data-testid="stSidebar"] .stButton > button {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.10) 0%,
        rgba(255,255,255,0.04) 50%,
        rgba(76,175,80,0.06) 100%
    ) !important;
    backdrop-filter: blur(16px) saturate(1.6) !important;
    -webkit-backdrop-filter: blur(16px) saturate(1.6) !important;
    border-radius: 14px !important;
    padding: 11px 16px !important;
    color: rgba(200,230,205,0.75) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.22s cubic-bezier(0.25,0.46,0.45,0.94) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-top: 1px solid rgba(255,255,255,0.22) !important;
    cursor: pointer !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow:
        0 2px 8px rgba(0,0,0,0.25),
        0 1px 0 rgba(255,255,255,0.08) inset,
        0 -1px 0 rgba(0,0,0,0.15) inset !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.18) 0%,
        rgba(255,255,255,0.08) 50%,
        rgba(76,175,80,0.12) 100%
    ) !important;
    border-color: rgba(255,255,255,0.22) !important;
    border-top-color: rgba(255,255,255,0.35) !important;
    color: #ffffff !important;
    box-shadow:
        0 4px 16px rgba(0,0,0,0.3),
        0 1px 0 rgba(255,255,255,0.15) inset,
        0 -1px 0 rgba(0,0,0,0.2) inset,
        0 0 0 1px rgba(76,175,80,0.15) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow:
        0 1px 4px rgba(0,0,0,0.3),
        0 1px 0 rgba(255,255,255,0.08) inset !important;
}
[data-testid="stSidebar"] .stButton > button.nav-active,
[data-testid="stSidebar"] .nav-active-btn button {
    background: linear-gradient(
        135deg,
        rgba(76,175,80,0.30) 0%,
        rgba(76,175,80,0.14) 50%,
        rgba(76,175,80,0.08) 100%
    ) !important;
    border-color: rgba(76,175,80,0.40) !important;
    border-top-color: rgba(140,220,140,0.45) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow:
        0 4px 20px rgba(76,175,80,0.20),
        0 1px 0 rgba(160,230,160,0.25) inset,
        0 -1px 0 rgba(0,0,0,0.15) inset !important;
}

/* Stat cards */
.sb-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
    padding: 0;
}
.sb-stat {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(76,175,80,0.15);
    border-radius: 12px;
    padding: 11px 13px;
    transition: border-color 0.2s;
}
.sb-stat:hover { border-color: rgba(76,175,80,0.35); }
.sb-stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.sb-stat-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    color: #4a8a5e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
    font-weight: 600;
}

/* Divider */
.sb-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(76,175,80,0.22), transparent);
    margin: 16px 0;
}

/* Section label */
.sb-section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    color: #2e5c3a;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    padding: 0 4px;
    margin-bottom: 8px;
}

/* Tagline pill */
.sb-tagline {
    display: inline-block;
    background: rgba(76,175,80,0.14);
    border: 1px solid rgba(76,175,80,0.28);
    border-radius: 100px;
    padding: 4px 12px;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    color: #6abf78;
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-top: 8px;
}

/* Footer */
.sb-footer {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: #2a4a32;
    text-align: center;
    padding: 4px 0 2px 0;
    line-height: 1.6;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# JS fallback to hide the sidebar collapse button.
st.markdown("""
<script>
(function() {
    function hideToggle() {
        document.querySelectorAll('button').forEach(function(btn) {
            if (btn.textContent.trim().includes('keyboard_double')) {
                btn.style.setProperty('display', 'none', 'important');
            }
        });
        ['stSidebarCollapseButton','collapsedControl'].forEach(function(id) {
            var el = document.querySelector('[data-testid="' + id + '"]');
            if (el) el.style.setProperty('display', 'none', 'important');
        });
    }
    hideToggle();
    new MutationObserver(hideToggle).observe(document.body, {childList:true, subtree:true});
})();
</script>
""", unsafe_allow_html=True)

# Brand logo and tagline displayed at the top of the sidebar.
st.sidebar.markdown("""
<div style="padding:1.6rem 1.2rem 1.4rem 1.2rem;margin-bottom:0.5rem">
    <svg viewBox="0 0 155 115" xmlns="http://www.w3.org/2000/svg" style="width:120px;height:auto;display:block;margin-bottom:10px">
      <text x="0"  y="38"  font-family="Space Grotesk,sans-serif" font-size="38" font-weight="800" letter-spacing="-0.04em"><tspan fill="#a5d6a7">H</tspan><tspan fill="rgba(255,255,255,0.92)">ou</tspan></text>
      <text x="30" y="76"  font-family="Space Grotesk,sans-serif" font-size="38" font-weight="800" letter-spacing="-0.04em"><tspan fill="#a5d6a7">S</tspan><tspan fill="rgba(255,255,255,0.92)">in</tspan></text>
      <text x="60" y="114" font-family="Space Grotesk,sans-serif" font-size="38" font-weight="800" letter-spacing="-0.04em" fill="#a5d6a7">G</text>
    </svg>
    <div style="height:1px;background:linear-gradient(90deg,rgba(76,175,80,0.35),transparent);margin:4px 0 10px 0"></div>
    <div style="font-family:'Inter',sans-serif;font-size:0.72rem;color:rgba(165,214,167,0.5);font-weight:400;line-height:1.6;letter-spacing:0.01em">Stop guessing.<br><span style="color:rgba(165,214,167,0.8);font-weight:500">Know your rent before you sign.</span></div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons for switching between pages.
if "page" not in st.session_state:
    st.session_state.page = "About"
if "favorites" not in st.session_state:
    st.session_state.favorites = set()

pages = ["About", "Map & Explorer", "Rent Estimator", "Analysis Details", "Favorites"]
_fav_count = len(st.session_state.favorites)
for _p in pages:
    _is_active = st.session_state.page == _p
    _btn_label = (f"Favorites  ({_fav_count})" if _fav_count > 0 else "Favorites") if _p == "Favorites" else _p
    _btn_style = (
        "background:linear-gradient(90deg,rgba(76,175,80,0.2),rgba(76,175,80,0.08));"
        "border:1px solid rgba(76,175,80,0.35);border-left:3px solid #4caf50;"
        "color:#ffffff;font-weight:700;box-shadow:0 2px 12px rgba(76,175,80,0.12);"
        if _is_active else
        "background:transparent;border:1px solid transparent;color:#5a9a6e;font-weight:500;"
    )
    st.sidebar.markdown(
        f"<style>#btn_{_p.replace(' ','_').replace('&','n')} button{{"
        f"display:flex!important;align-items:center!important;width:100%!important;"
        f"border-radius:12px!important;padding:11px 16px!important;"
        f"font-family:'Inter',sans-serif!important;font-size:0.88rem!important;"
        f"letter-spacing:0.01em!important;transition:all 0.18s ease!important;"
        f"cursor:pointer!important;text-align:left!important;justify-content:flex-start!important;"
        f"{_btn_style}}}</style>",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        if st.button(_btn_label, key=f"nav_{_p}"):
            st.session_state.page = _p
            st.rerun()

page = st.session_state.page

# Live model stats displayed at the bottom of the sidebar.
st.sidebar.markdown(f"""
<div class="sb-divider"></div>
<div class="sb-section-label">Live model stats</div>
<div class="sb-stat-grid">
  <div class="sb-stat">
    <div class="sb-stat-val">{len(df)}</div>
    <div class="sb-stat-lbl">Listings</div>
  </div>
  <div class="sb-stat">
    <div class="sb-stat-val">{r2:.0%}</div>
    <div class="sb-stat-lbl">Model accuracy</div>
  </div>
  <div class="sb-stat">
    <div class="sb-stat-val">HSG</div>
    <div class="sb-stat-lbl">Built for students</div>
  </div>
  <div class="sb-stat">
    <div class="sb-stat-val">Free</div>
    <div class="sb-stat-lbl">To use</div>
  </div>
</div>

""", unsafe_allow_html=True)

# Collect CV metrics, learning curve, and feature correlations for the Analysis page.
@st.cache_data
def _compute_ml_details(csv_version):
    """Re-run the pipeline to collect per-fold MAE/R², learning curve data,
    and feature–rent correlations for the Analysis Details page."""
    _df = load_data(csv_version=csv_version)
    _feature_cols = [
        "rooms", "living_space", "floor", "lat", "lng",
        "dist_transit_m", "dist_grocery_m", "dist_gym_m", "dist_hsg_m", "dist_center_m",
    ]
    _feature_names = [
        "Rooms", "Living space (m²)", "Floor", "Latitude", "Longitude",
        "Dist. transit", "Dist. grocery", "Dist. gym", "Dist. HSG", "Dist. city centre",
    ]
    _params = dict(
        n_estimators=800, max_depth=4, learning_rate=0.02, subsample=1.0,
        colsample_bytree=0.9, min_child_weight=2, reg_alpha=0.1,
        reg_lambda=1.0, gamma=0, random_state=42, verbosity=0,
    )
    X_all = _df[_feature_cols].values
    y_all = _df["rent_gross"].values
    cv    = KFold(n_splits=5, shuffle=True, random_state=42)

    # Rebuild the outlier mask using the same IQR rule as the main model.
    _p0    = cross_val_predict(XGBRegressor(**_params), X_all, y_all, cv=cv)
    _resid = np.abs(_p0 - y_all)
    _q25, _q75 = np.percentile(_resid, 25), np.percentile(_resid, 75)
    _omask = _resid > (_q75 + 2.0 * (_q75 - _q25))
    X_c, y_c = X_all[~_omask], y_all[~_omask]

    # Fit one model per fold and collect MAE, R², and the learning curve data.
    fold_rows   = []   # {Fold, Split, Index}
    fold_metrics = []  # {Fold, MAE, R²}
    lc_data     = []   # {Fold, n_estimators, val MAE}

    for fi, (tr, va) in enumerate(cv.split(X_c)):
        m = XGBRegressor(**_params, eval_metric="mae")
        m.fit(
            X_c[tr], y_c[tr],
            eval_set=[(X_c[tr], y_c[tr]), (X_c[va], y_c[va])],
            verbose=False,
        )
        res = m.evals_result()
        train_mae = res["validation_0"]["mae"]
        val_mae   = res["validation_1"]["mae"]

        # Sample every 25 trees to keep the chart data manageable.
        step = 25
        for ni in range(0, len(val_mae), step):
            lc_data.append({
                "Fold": f"Fold {fi+1}",
                "n_estimators": ni + 1,
                "Train MAE": round(train_mae[ni], 1),
                "Val MAE":   round(val_mae[ni],   1),
            })
        # Append the final tree if it was not already captured by the step loop.
        if (len(val_mae) - 1) % step != 0:
            lc_data.append({
                "Fold": f"Fold {fi+1}",
                "n_estimators": len(val_mae),
                "Train MAE": round(train_mae[-1], 1),
                "Val MAE":   round(val_mae[-1],   1),
            })

        p_va = m.predict(X_c[va])
        fold_metrics.append({
            "Fold":      f"Fold {fi+1}",
            "n train":   len(tr),
            "n val":     len(va),
            "MAE (CHF)": round(float(np.mean(np.abs(p_va - y_c[va]))), 1),
            "R²":        round(float(1 - np.sum((p_va - y_c[va])**2)
                                         / np.sum((y_c[va] - y_c[va].mean())**2)), 3),
        })

        # Track train/validation membership for each listing across folds.
        for i in tr:
            fold_rows.append({"Fold": f"Fold {fi+1}", "Index": i, "Split": "Train"})
        for i in va:
            fold_rows.append({"Fold": f"Fold {fi+1}", "Index": i, "Split": "Validation"})

    # Pearson correlation between each feature and rent on the clean set.
    corr_rows = []
    for i, name in enumerate(_feature_names):
        c = float(np.corrcoef(X_c[:, i], y_c)[0, 1])
        corr_rows.append({"Feature": name, "Correlation": round(c, 3),
                          "Direction": "positive" if c >= 0 else "negative"})

    return {
        "n_total":    len(y_all),
        "n_outliers": int(_omask.sum()),
        "n_clean":    len(y_c),
        "fold_metrics":  fold_metrics,
        "lc_data":       lc_data,
        "fold_rows":     fold_rows,
        "corr_rows":     corr_rows,
    }


@st.cache_data
def load_pois(poi_version=None):
    """Load POI data from pois.json; poi_version (file mtime) busts the cache on file change."""
    poi_path = os.path.join(os.path.dirname(__file__), "pois.json")
    with open(poi_path) as f:
        return json.load(f)


# PAGE 0 – ABOUT
if page == "About":
    # Hero banner with the app name, description, and key stats.
    st.markdown(f"""
    <style>
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0);    }}
    }}
    .hh-hero {{
        background: linear-gradient(135deg, #0a2e1a 0%, #0d3321 45%, #1a5c2a 100%);
        border-radius: 20px;
        padding: 3.5rem 3rem 3rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.6s ease both;
    }}
    .hh-hero::before {{
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 280px; height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(76,175,80,0.18) 0%, transparent 70%);
    }}
    .hh-hero::after {{
        content: '';
        position: absolute;
        bottom: -80px; left: 40%;
        width: 320px; height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(26,92,42,0.35) 0%, transparent 70%);
    }}
    .hh-badge {{
        display: inline-block;
        background: rgba(76,175,80,0.22);
        border: 1px solid rgba(76,175,80,0.45);
        color: #81c995;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-radius: 100px;
        padding: 4px 14px;
        margin-bottom: 1.2rem;
    }}
    .hh-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.6rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.04em;
        line-height: 1.0;
        margin-bottom: 1rem;
    }}
    .hh-title span {{
        background: linear-gradient(90deg, #4caf50, #81c995);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hh-sub {{
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: #a5d6a7;
        font-weight: 400;
        max-width: 520px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }}
    .hh-stats-row {{
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }}
    .hh-stat {{
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.85rem 1.4rem;
        min-width: 130px;
        backdrop-filter: blur(6px);
    }}
    .hh-stat-val {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }}
    .hh-stat-lbl {{
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #81c995;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 2px;
    }}

    /* Cards */
    .hh-cards {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-bottom: 2.2rem;
        animation: fadeUp 0.7s ease 0.1s both;
    }}
    .hh-card {{
        background: #ffffff;
        border: 1px solid #e0ede4;
        border-radius: 16px;
        padding: 1.6rem 1.5rem;
        box-shadow: 0 2px 12px rgba(13,51,33,0.06);
        transition: box-shadow 0.2s, transform 0.2s;
        cursor: default;
    }}
    .hh-card:hover {{
        box-shadow: 0 8px 28px rgba(13,51,33,0.13);
        transform: translateY(-3px);
    }}
    .hh-card-icon {{
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }}
    .hh-card-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #0d3321;
        margin-bottom: 0.45rem;
        letter-spacing: -0.01em;
    }}
    .hh-card-body {{
        font-family: 'Inter', sans-serif;
        font-size: 0.87rem;
        color: #5a7a65;
        line-height: 1.6;
    }}

    /* Steps */
    .hh-steps {{
        animation: fadeUp 0.7s ease 0.2s both;
        margin-bottom: 2.2rem;
    }}
    .hh-steps-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #0d3321;
        letter-spacing: -0.02em;
        margin-bottom: 1.2rem;
    }}
    .hh-steps-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }}
    .hh-step {{
        display: flex;
        gap: 1rem;
        align-items: flex-start;
        background: #f6faf7;
        border-radius: 14px;
        padding: 1.3rem 1.2rem;
        border: 1px solid #ddeee2;
    }}
    .hh-step-num {{
        flex-shrink: 0;
        width: 32px; height: 32px;
        background: #0d3321;
        color: #ffffff;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 800;
    }}
    .hh-step-text-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.92rem;
        font-weight: 700;
        color: #0d3321;
        margin-bottom: 0.3rem;
    }}
    .hh-step-text-body {{
        font-family: 'Inter', sans-serif;
        font-size: 0.83rem;
        color: #5a7a65;
        line-height: 1.55;
    }}

    /* Footer */
    .hh-footer {{
        background: linear-gradient(90deg, #f6faf7, #edf5ef);
        border: 1px solid #ddeee2;
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: fadeUp 0.7s ease 0.3s both;
    }}
    .hh-footer-icon {{
        font-size: 1.4rem;
    }}
    .hh-footer-text {{
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #4a7c5a;
    }}
    .hh-footer-text strong {{
        color: #0d3321;
        font-weight: 700;
    }}
    </style>

    <div class="hh-hero">
        <div class="hh-title">Find your<br><span>fair rent</span>.</div>
        <div class="hh-sub">
            Stop overpaying. Our ML model analyses 300+ real listings to tell you exactly
            whether any St. Gallen apartment is a bargain, fair, or overpriced in seconds.
        </div>
        <div class="hh-stats-row">
            <div class="hh-stat">
                <div class="hh-stat-val">{len(df)}</div>
                <div class="hh-stat-lbl">Listings</div>
            </div>
            <div class="hh-stat">
                <div class="hh-stat-val">R² {r2:.2f}</div>
                <div class="hh-stat-lbl">Model accuracy</div>
            </div>
            <div class="hh-stat">
                <div class="hh-stat-val">CHF {mae:.0f}</div>
                <div class="hh-stat-lbl">Avg. error / mo</div>
            </div>
            <div class="hh-stat">
                <div class="hh-stat-val">10</div>
                <div class="hh-stat-lbl">Features</div>
            </div>
        </div>
    </div>

    <div class="hh-cards">
        <div class="hh-card">
            <div class="hh-card-icon" style="background:#e8f5e9"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2e7d32" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg></div>
            <div class="hh-card-title">Interactive Map</div>
            <div class="hh-card-body">Every listing plotted on a live map, colour-coded from green (undervalued) to red (overpriced). Walk routes to HSG, transit, and shops included.</div>
        </div>
        <div class="hh-card">
            <div class="hh-card-icon" style="background:#e3f2fd"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1565c0" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="8" width="20" height="14" rx="2"/><path d="M12 8V5"/><circle cx="12" cy="3" r="2"/><line x1="8" y1="15" x2="8" y2="15" stroke-width="3"/><line x1="16" y1="15" x2="16" y2="15" stroke-width="3"/><path d="M7 19h4m2 0h4"/></svg></div>
            <div class="hh-card-title">XGBoost Pricing Model</div>
            <div class="hh-card-body">Trained on size, floor, location, and distance to 5 amenity types. Cross-validated on real data, not just a rule of thumb.</div>
        </div>
        <div class="hh-card">
            <div class="hh-card-icon" style="background:#fff3e0"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#e65100" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></div>
            <div class="hh-card-title">Instant Estimator</div>
            <div class="hh-card-body">Enter any St. Gallen address and get a fair-market rent estimate before you even contact the landlord.</div>
        </div>
    </div>

    <div class="hh-steps">
        <div class="hh-steps-title">How it works</div>
        <div class="hh-steps-grid">
            <div class="hh-step">
                <div class="hh-step-num">1</div>
                <div>
                    <div class="hh-step-text-title">Browse the map</div>
                    <div class="hh-step-text-body">Green dot = bargain. Red dot = overpriced. Filter by price, rooms, and walking distances in one click.</div>
                </div>
            </div>
            <div class="hh-step">
                <div class="hh-step-num">2</div>
                <div>
                    <div class="hh-step-text-title">Check the valuation gap</div>
                    <div class="hh-step-text-body">Our model predicts the fair rent from apartment features. The gap (%) tells you how much above or below market a listing really is.</div>
                </div>
            </div>
            <div class="hh-step">
                <div class="hh-step-num">3</div>
                <div>
                    <div class="hh-step-text-title">Estimate any address</div>
                    <div class="hh-step-text-body">Use the Rent Estimator to benchmark a specific flat, ideal before signing a lease or negotiating with a landlord.</div>
                </div>
            </div>
        </div>
    </div>

    <div class="hh-footer">
        <div class="hh-footer-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#4a7c5a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg></div>
        <div class="hh-footer-text">
            <strong>Ready to explore?</strong> Pick a page from the sidebar and start with <strong>Map & Explorer</strong> to see all listings at a glance.
        </div>
    </div>
    """, unsafe_allow_html=True)

# PAGE – MAP & EXPLORER
elif page == "Map & Explorer":
    import folium
    from streamlit_folium import st_folium


    # Load pre-computed POI locations for bus stops, groceries, gyms, and HSG.
    _poi_path = os.path.join(os.path.dirname(__file__), "pois.json")
    pois = load_pois(poi_version=os.path.getmtime(_poi_path))

    def gap_to_hex(gap):
        """Map valuation gap % to red→yellow→green hex (clamped ±25)."""
        gap = max(-25, min(25, gap))
        if gap >= 0:          # 0..+25: yellow → red
            t = gap / 25.0
            r, g, b = 220, int(200 * (1 - t)), 0
        else:                 # -25..0: green → yellow
            t = -gap / 25.0
            r, g, b = int(220 * (1 - t)), 180, 0
        return f"#{r:02x}{g:02x}{b:02x}"

    # Walking route helper functions using haversine distance and OSRM.
    def _hav_map(lat1, lon1, lat2, lon2):
        import math
        R = 6371000
        p1, p2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
             + math.cos(p1) * math.cos(p2)
             * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _nearest_poi(lat, lng, poi_list):
        """Closest POI to (lat, lng)."""
        return min(poi_list, key=lambda p: _hav_map(lat, lng, p["lat"], p["lng"]))

    def _osrm_route(lat1, lng1, lat2, lng2):
        """OSRM walking route as list of (lat,lng); straight line on failure."""
        import urllib.request, json as _json
        url = (f"http://router.project-osrm.org/route/v1/walking/"
               f"{lng1},{lat1};{lng2},{lat2}"
               f"?overview=full&geometries=geojson")
        try:
            with urllib.request.urlopen(url, timeout=6) as resp:
                data = _json.loads(resp.read())
            coords = data["routes"][0]["geometry"]["coordinates"]
            return [(c[1], c[0]) for c in coords]
        except Exception:
            return [(lat1, lng1), (lat2, lng2)]

    # Page header with map title and subtitle.
    st.markdown("""
    <style>
    .pg-header {
        background: linear-gradient(135deg, #0a2e1a 0%, #0d3321 55%, #1a5c2a 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .pg-header::after {
        content: '';
        position: absolute;
        right: -40px; top: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(76,175,80,0.15) 0%, transparent 65%);
    }
    .pg-header-icon {
        font-size: 2rem;
        line-height: 1;
    }
    .pg-header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .pg-header-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #81c995;
        margin-top: 3px;
    }
    .pg-section-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.4rem 0 0.7rem 0;
    }
    .pg-section-label-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #0d3321;
        letter-spacing: -0.01em;
    }
    .pg-section-label-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #d4e8d8, transparent);
    }
    .pg-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #c8e6c9, transparent);
        margin: 1.4rem 0;
        border: none;
    }
    </style>
    <div class="pg-header">
        <div class="pg-header-icon"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.9)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg></div>
        <div>
            <div class="pg-header-title">Map & Explorer</div>
            <div class="pg-header-sub">Red = overpriced &nbsp;·&nbsp; Yellow = fair &nbsp;·&nbsp; Green = undervalued &nbsp;·&nbsp; Click a dot for walk routes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter controls for listing type, per-room price, and room count.
    st.markdown('<div class="pg-section-label"><span class="pg-section-label-text">Filters</span><div class="pg-section-label-line"></div></div>', unsafe_allow_html=True)

    fc_a, fc_b, fc_c, fc_d = st.columns(4)
    label_filter = fc_a.multiselect(
        "Per-room price", ["expensive", "average", "affordable"],
        default=["expensive", "average", "affordable"],
    )
    type_filter = fc_b.multiselect(
        "Listing type", ["Studio", "Shared flat"],
        default=["Studio", "Shared flat"],
    )
    _pr_min = int(df["per_room_rent"].min())
    _pr_max = int(df["per_room_rent"].max())
    rent_range = fc_c.slider(
        "Rent/room (CHF/mo)", _pr_min, _pr_max, (_pr_min, _pr_max),
    )
    rooms_range = fc_d.slider(
        "Rooms", float(df.rooms.min()), float(df.rooms.max()),
        (float(df.rooms.min()), float(df.rooms.max())), step=0.5,
    )

    # Split layout: map on the left, walk-time sliders on the right.
    map_col, slider_col = st.columns([3, 1])

    # Render sliders before filtering so their values are available when building the query.
    _walk_max = int(df[["walk_hsg_min","walk_transit_min","walk_grocery_min",
                         "walk_gym_min","walk_center_min"]].max().max()) + 1
    with slider_col:
        st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.82rem;font-weight:700;color:#0d3321;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Max walk time</div>', unsafe_allow_html=True)
        max_hsg     = st.slider("HSG",     1, _walk_max, _walk_max, format="%d min", key="slider_hsg")
        max_transit = st.slider("Transit",  1, _walk_max, _walk_max, format="%d min", key="slider_transit")
        max_grocery = st.slider("Grocery",  1, _walk_max, _walk_max, format="%d min", key="slider_grocery")
        max_gym     = st.slider("Gym",      1, _walk_max, _walk_max, format="%d min", key="slider_gym")
        max_center  = st.slider("Centre",   1, _walk_max, _walk_max, format="%d min", key="slider_centre")
        st.markdown('<hr class="pg-divider"><div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.82rem;font-weight:700;color:#0d3321;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Layers</div>', unsafe_allow_html=True)
        show_listings = st.checkbox("Listings",  value=True, key="chk_listings")
        show_stops    = st.checkbox("Transit",    value=True, key="chk_transit")
        show_groc     = st.checkbox("Groceries",  value=True, key="chk_groceries")
        show_gyms     = st.checkbox("Gyms",       value=True, key="chk_gyms")
        show_hsg      = st.checkbox("HSG",         value=True, key="chk_hsg")

    filtered = df[
        (df["per_room_label"].isin(label_filter)) &
        (df["listing_type"].isin(type_filter)) &
        (df["per_room_rent"].between(*rent_range)) &
        (df["rooms"].between(*rooms_range)) &
        (df["walk_hsg_min"]     <= max_hsg) &
        (df["walk_transit_min"] <= max_transit) &
        (df["walk_grocery_min"] <= max_grocery) &
        (df["walk_gym_min"]     <= max_gym) &
        (df["walk_center_min"]  <= max_center)
    ].copy()

    # Pre-compute nearest POI coordinates for each listing for the JS routing layer.
    _HSG_LAT, _HSG_LNG = 47.431759683827714, 9.374557836074315

    def _nearest_coords(lat, lng, poi_list):
        best = min(poi_list, key=lambda p: (p["lat"] - lat) ** 2 + (p["lng"] - lng) ** 2)
        return [best["lat"], best["lng"]]

    _routes_data = {}
    for _, _r in filtered.iterrows():
        _key = f"{_r['lat']:.6f}_{_r['lng']:.6f}"
        _routes_data[_key] = {
            "transit": [[_r["lat"], _r["lng"]], _nearest_coords(_r["lat"], _r["lng"], pois["stops"])],
            "grocery": [[_r["lat"], _r["lng"]], _nearest_coords(_r["lat"], _r["lng"], pois["groceries"])],
            "gym":     [[_r["lat"], _r["lng"]], _nearest_coords(_r["lat"], _r["lng"], pois["gyms"])],
            "hsg":     [[_r["lat"], _r["lng"]], [_HSG_LAT, _HSG_LNG]],
        }
    _routes_json = json.dumps(_routes_data)

    # Add ESRI satellite tiles via TileLayer to avoid a folium ≥0.20 bug with the tiles= arg.
    m = folium.Map(
        location=[df["lat"].mean(), df["lng"].mean()],
        zoom_start=14,
        tiles=None,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, GeoEye, Earthstar Geographics",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)

    # White text shadow to keep emoji markers readable on dark satellite tiles.
    OUTLINE = "text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff"

    def poi_marker(p, emoji, font_size, fallback=""):
        """Emoji marker with tooltip + popup."""
        name = p.get("name", "").strip() or fallback
        sz = font_size + 10
        folium.Marker(
            location=[p["lat"], p["lng"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:{font_size}px;{OUTLINE};line-height:1">{emoji}</div>',
                icon_size=(sz, sz),
                icon_anchor=(sz // 2, sz // 2),
            ),
            tooltip=name or None,
            popup=folium.Popup(
                f'<div style="font-family:sans-serif;font-size:13px"><b>{emoji} {name}</b></div>',
                max_width=220,
            ) if name else None,
        ).add_to(m)

    # Add POI markers to the map, placed under the listing dots.
    if show_stops:
        for p in pois["stops"]:
            poi_marker(p, '🚎', 18)

    if show_groc:
        for p in pois["groceries"]:
            poi_marker(p, '🛍️', 18)

    if show_gyms:
        for p in pois["gyms"]:
            poi_marker(p, '🥊', 18)

    if show_hsg:
        for p in pois["hsg"]:
            poi_marker(p, '🎓', 22, "University of St. Gallen")

    # Add a coloured dot for each listing based on its per-room valuation gap.
    if show_listings:
        for _map_row_idx, row in filtered.iterrows():
            colour = gap_to_hex(row["per_room_gap"])
            ltype  = row["listing_type"]

            area_str = (f"{row['living_space']:.0f} m²"
                        if pd.notna(row.get("living_space")) and row["living_space"] > 0 else "?")

            _gap = row['per_room_gap']
            if _gap < 0:
                _gap_txt    = f"{_gap:.1f}% below market"
                _gap_clr    = "#4caf7d"
                _gap_bg     = "rgba(76,175,125,0.18)"
                _gap_border = "rgba(76,175,125,0.45)"
            else:
                _gap_txt    = f"+{_gap:.1f}% above market"
                _gap_clr    = "#ff6b6b"
                _gap_bg     = "rgba(255,107,107,0.15)"
                _gap_border = "rgba(255,107,107,0.4)"

            popup_html = f"""
            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;width:262px;background:#0d3321;border-radius:14px;overflow:hidden;color:#ffffff">
                <div style="padding:16px 16px 14px">
                    <div style="font-size:18px;font-weight:700;color:#ffffff;margin-bottom:3px;letter-spacing:-0.02em">{row['street']}</div>
                    <div style="font-size:13px;color:#81c995;margin-bottom:12px">{row['locality']} &middot; {int(row['postal_code'])}</div>
                    <div style="display:inline-block;border:1px solid #2d6a42;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:600;color:#a5d6a7;letter-spacing:0.06em">{ltype.upper()}</div>
                </div>
                <div style="height:1px;background:#1a4a2a;margin:0 16px"></div>
                <div style="padding:14px 16px 12px">
                    <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:7px">
                        <span style="font-size:32px;font-weight:700;color:#ffffff;letter-spacing:-0.03em">CHF {int(row['per_room_rent']):,}</span>
                        <span style="font-size:13px;color:#81c995;font-weight:400">/ room &middot; month</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px">
                        <span style="font-size:13px;color:#81c995">Predicted CHF {int(row['predicted_per_room_rent']):,}</span>
                        <span style="font-size:12px;font-weight:600;color:{_gap_clr};background:{_gap_bg};border:1px solid {_gap_border};border-radius:20px;padding:2px 9px;white-space:nowrap">{_gap_txt}</span>
                    </div>
                    <div style="font-size:12px;color:#4a7c5a">Full flat CHF {int(row['rent_gross']):,} / month</div>
                </div>
                <div style="height:1px;background:#1a4a2a;margin:0 16px"></div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;padding:14px 16px">
                    <div style="text-align:center">
                        <div style="font-size:20px;font-weight:700;color:#ffffff">{row['rooms']}</div>
                        <div style="font-size:10px;font-weight:500;color:#4a7c5a;text-transform:uppercase;letter-spacing:0.06em;margin-top:3px">Rooms</div>
                    </div>
                    <div style="text-align:center;border-left:1px solid #1a4a2a;border-right:1px solid #1a4a2a">
                        <div style="font-size:20px;font-weight:700;color:#ffffff">{area_str}</div>
                        <div style="font-size:10px;font-weight:500;color:#4a7c5a;text-transform:uppercase;letter-spacing:0.06em;margin-top:3px">Area</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size:20px;font-weight:700;color:#ffffff">{int(row['floor'])}</div>
                        <div style="font-size:10px;font-weight:500;color:#4a7c5a;text-transform:uppercase;letter-spacing:0.06em;margin-top:3px">Floor</div>
                    </div>
                </div>
                <div style="height:1px;background:#1a4a2a;margin:0 16px"></div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:14px 16px">
                    <div style="display:flex;align-items:center;gap:10px">
                        <div style="width:34px;height:34px;min-width:34px;background:#1a5c2a;border-radius:8px;display:flex;align-items:center;justify-content:center">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a5d6a7" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="3" width="15" height="13" rx="1"/><path d="M16 8h4l3 5v3h-7V8z"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/></svg>
                        </div>
                        <div><div style="font-size:14px;font-weight:600;color:#ffffff">{row['walk_transit_min']:.0f} min</div><div style="font-size:11px;color:#4a7c5a">Bus stop</div></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <div style="width:34px;height:34px;min-width:34px;background:#1a5c2a;border-radius:8px;display:flex;align-items:center;justify-content:center">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a5d6a7" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/><line x1="3" y1="6" x2="21" y2="6"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>
                        </div>
                        <div><div style="font-size:14px;font-weight:600;color:#ffffff">{row['walk_grocery_min']:.0f} min</div><div style="font-size:11px;color:#4a7c5a">Shopping</div></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <div style="width:34px;height:34px;min-width:34px;background:#1a5c2a;border-radius:8px;display:flex;align-items:center;justify-content:center">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a5d6a7" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                        </div>
                        <div><div style="font-size:14px;font-weight:600;color:#ffffff">{row['walk_gym_min']:.0f} min</div><div style="font-size:11px;color:#4a7c5a">Amenities</div></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <div style="width:34px;height:34px;min-width:34px;background:#1a5c2a;border-radius:8px;display:flex;align-items:center;justify-content:center">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a5d6a7" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
                        </div>
                        <div><div style="font-size:14px;font-weight:600;color:#ffffff">{row['walk_hsg_min']:.0f} min</div><div style="font-size:11px;color:#4a7c5a">University</div></div>
                    </div>
                </div>
                <div style="height:1px;background:#1a4a2a;margin:0 16px"></div>
                <div style="padding:12px 16px;display:flex;justify-content:space-between;align-items:center">
                    <a href="{row['homegate_url']}" target="_blank" style="font-size:13px;font-weight:500;color:#4caf7d;text-decoration:none">View on homegate.ch &#8599;</a>
                    {'<span style="font-size:12px;font-weight:600;color:#ff4b6e;display:inline-flex;align-items:center;gap:4px"><svg width="12" height="12" viewBox="0 0 24 24" fill="#ff4b6e" stroke="#ff4b6e" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg> Saved</span>' if _map_row_idx in st.session_state.favorites else ''}
                </div>
            </div>"""

            _rent_txt = f"{int(row['per_room_rent']):,}"
            _price_label = f"CHF {_rent_txt}"
            _icon_w = int(len(_price_label) * 8 + 32)  # 8px/char + dot + padding
            _icon_h = 24
            _badge_html = (
                f'<div style="'
                f'display:inline-flex;align-items:center;gap:5px;'
                f'background:#ffffff;'
                f'border-radius:20px;'
                f'padding:3px 8px 3px 6px;'
                f'box-shadow:0 2px 10px rgba(0,0,0,0.35),0 1px 3px rgba(0,0,0,0.2);'
                f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
                f'font-size:13px;'
                f'font-weight:700;'
                f'color:#0d3321;'
                f'white-space:nowrap;'
                f'letter-spacing:-0.02em;'
                f'cursor:pointer;'
                f'">'
                f'<span style="width:9px;height:9px;border-radius:50%;background:{colour};flex-shrink:0;display:inline-block;"></span>'
                f'{_price_label}'
                f'</div>'
            )
            folium.Marker(
                location=[row["lat"], row["lng"]],
                icon=folium.DivIcon(
                    html=_badge_html,
                    icon_size=(_icon_w, _icon_h),
                    icon_anchor=(_icon_w // 2, _icon_h // 2),
                ),
                popup=folium.Popup(popup_html, max_width=290),
                tooltip=f"{row['street']} · {ltype} · CHF {_rent_txt}/room · {row['per_room_gap']:+.1f}%",
            ).add_to(m)

    # Client-side walking route drawing using OSRM; clicking a dot draws routes without a rerun.
    from folium import MacroElement
    from jinja2 import Template

    # CSS to remove Folium's default popup border, shadow, and background.
    _css_macro = MacroElement()
    _css_macro._template = Template(
        "{% macro header(this, kwargs) %}"
        "<style>"
        ".leaflet-popup-content-wrapper{"
        "background:transparent!important;"
        "box-shadow:none!important;"
        "border:none!important;"
        "border-radius:0!important;"
        "padding:0!important;"
        "}"
        ".leaflet-popup-content{"
        "margin:0!important;"
        "padding:0!important;"
        "line-height:normal!important;"
        "}"
        ".leaflet-popup-tip-container{display:none!important}"
        "</style>"
        "{% endmacro %}"
    )
    _css_macro.add_to(m)

    _js_macro = MacroElement()
    _js_macro._template = Template(
        "{% macro script(this, kwargs) %}\n"
        "(function(){\n"
        "  var rd = " + _routes_json + ";\n"
        "  var al = [];\n"
        "  var colors = {transit:'#2980b9',grocery:'#e67e22',gym:'#8e44ad',hsg:'#27ae60'};\n"
        "  var mapObj = {{ this._parent.get_name() }};\n"
        "\n"
        "  function clearRoutes(){\n"
        "    al.forEach(function(l){mapObj.removeLayer(l);}); al=[];\n"
        "  }\n"
        "\n"
        "  function chaikin(pts,iters){\n"
        "    for(var i=0;i<iters;i++){\n"
        "      var out=[pts[0]];\n"
        "      for(var j=0;j<pts.length-1;j++){\n"
        "        var a=pts[j],b=pts[j+1];\n"
        "        out.push([a[0]*0.75+b[0]*0.25,a[1]*0.75+b[1]*0.25]);\n"
        "        out.push([a[0]*0.25+b[0]*0.75,a[1]*0.25+b[1]*0.75]);\n"
        "      }\n"
        "      out.push(pts[pts.length-1]); pts=out;\n"
        "    }\n"
        "    return pts;\n"
        "  }\n"
        "\n"
        "  function drawRoute(from, to, color){\n"
        "    var url = 'https://routing.openstreetmap.de/routed-foot/route/v1/foot/'\n"
        "      + from[1].toFixed(6)+','+from[0].toFixed(6)+';'\n"
        "      + to[1].toFixed(6)+','+to[0].toFixed(6)\n"
        "      + '?overview=full&geometries=geojson';\n"
        "    fetch(url)\n"
        "      .then(function(r){return r.json();})\n"
        "      .then(function(d){\n"
        "        if(!d.routes||!d.routes[0]) return;\n"
        "        var raw=d.routes[0].geometry.coordinates.map(function(c){return [c[1],c[0]];});\n"
        "        var latlngs=chaikin(raw,5);\n"
        "        var outline=L.polyline(latlngs,{color:'white',weight:7,opacity:0.8,smoothFactor:1}).addTo(mapObj);\n"
        "        var ln=L.polyline(latlngs,{color:color,weight:4,opacity:0.95,smoothFactor:1}).addTo(mapObj);\n"
        "        al.push(outline); al.push(ln);\n"
        "      })\n"
        "      .catch(function(){\n"
        "        var outline=L.polyline([from,to],{color:'white',weight:6,opacity:0.7,dashArray:'8 4',smoothFactor:4}).addTo(mapObj);\n"
        "        var ln=L.polyline([from,to],{color:color,weight:3,opacity:0.85,dashArray:'8 4',smoothFactor:4}).addTo(mapObj);\n"
        "        al.push(outline); al.push(ln);\n"
        "      });\n"
        "  }\n"
        "\n"
        "  function attachHandlers(){\n"
        "    mapObj.eachLayer(function(layer){\n"
        "      if(!(layer instanceof L.Marker) || !(layer.options.icon instanceof L.DivIcon)) return;\n"
        "      layer.on('click',function(e){\n"
        "        L.DomEvent.stopPropagation(e);\n"
        "        var ll=layer.getLatLng();\n"
        "        var key=ll.lat.toFixed(6)+'_'+ll.lng.toFixed(6);\n"
        "        clearRoutes();\n"
        "        var routes=rd[key]; if(!routes) return;\n"
        "        var ring=L.circleMarker(ll,{radius:13,color:'white',weight:2.5,fill:false,opacity:1}).addTo(mapObj);\n"
        "        al.push(ring);\n"
        "        Object.keys(routes).forEach(function(t){\n"
        "          drawRoute(routes[t][0],routes[t][1],colors[t]);\n"
        "        });\n"
        "      });\n"
        "    });\n"
        "    mapObj.on('click',function(){ clearRoutes(); });\n"
        "  }\n"
        "\n"
        "  setTimeout(attachHandlers, 250);\n"
        "})();\n"
        "{% endmacro %}"
    )
    _js_macro.add_to(m)

    with map_col:
        st_folium(m, use_container_width=True, height=650,
                  returned_objects=[],
                  key="main_folium_map")

    # Count cards for expensive, fair, affordable, studio, and shared-flat listings.
    pr_counts   = filtered["per_room_label"].value_counts()
    type_counts = filtered["listing_type"].value_counts()
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;
                margin:1.2rem 0;font-family:'Inter',sans-serif">
      <div style="background:#fde8e8;border:1px solid #f5c6c6;border-radius:12px;
                  padding:12px 16px">
        <div style="font-size:1.5rem;font-weight:800;color:#7a1c1c;
                    font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
            {pr_counts.get("expensive", 0)}</div>
        <div style="font-size:0.72rem;color:#b94040;text-transform:uppercase;
                    letter-spacing:0.07em;font-weight:600;margin-top:2px">Expensive/room</div>
      </div>
      <div style="background:#fef9e7;border:1px solid #f5e6a3;border-radius:12px;
                  padding:12px 16px">
        <div style="font-size:1.5rem;font-weight:800;color:#6b4c00;
                    font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
            {pr_counts.get("average", 0)}</div>
        <div style="font-size:0.72rem;color:#997a00;text-transform:uppercase;
                    letter-spacing:0.07em;font-weight:600;margin-top:2px">Fair/room</div>
      </div>
      <div style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:12px;
                  padding:12px 16px">
        <div style="font-size:1.5rem;font-weight:800;color:#0d3321;
                    font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
            {pr_counts.get("affordable", 0)}</div>
        <div style="font-size:0.72rem;color:#1a5c2a;text-transform:uppercase;
                    letter-spacing:0.07em;font-weight:600;margin-top:2px">Affordable/room</div>
      </div>
      <div style="background:#e3edf9;border:1px solid #aac4e8;border-radius:12px;
                  padding:12px 16px">
        <div style="font-size:1.5rem;font-weight:800;color:#1a3a6b;
                    font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
            {type_counts.get("Studio", 0)}</div>
        <div style="font-size:0.72rem;color:#2a5aa0;text-transform:uppercase;
                    letter-spacing:0.07em;font-weight:600;margin-top:2px">Studios</div>
      </div>
      <div style="background:#f3f0fb;border:1px solid #ccc0ef;border-radius:12px;
                  padding:12px 16px">
        <div style="font-size:1.5rem;font-weight:800;color:#3a1a6b;
                    font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
            {type_counts.get("Shared flat", 0)}</div>
        <div style="font-size:0.72rem;color:#5a3a9a;text-transform:uppercase;
                    letter-spacing:0.07em;font-weight:600;margin-top:2px">Shared flats</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Listing explorer with sorting, pagination, and save-to-favourites.
    st.markdown('<hr class="pg-divider">', unsafe_allow_html=True)

    # Transparent styling for the heart favourite buttons.
    st.markdown("""<style>
div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    padding: 2px 6px !important;
    min-height: 28px !important;
    font-size: 17px !important;
    line-height: 1 !important;
    color: #9ab8a0 !important;
}
div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"]:hover {
    border-color: #e05a6e !important;
    background: #fff0f3 !important;
    color: #e05a6e !important;
}
</style>""", unsafe_allow_html=True)

    # Sort column and direction controls for the explorer table.
    _SORT_OPTS = {
        'per_room_gap':            'Gap (best deal)',
        'per_room_rent':           'Rent / room',
        'predicted_per_room_rent': 'Predicted rent',
        'rent_gross':              'Full rent',
        'rooms':                   'Rooms',
        'walk_hsg_min':            'Walk to HSG',
        'street':                  'Address',
    }
    if '_expl_sort_col' not in st.session_state:
        st.session_state['_expl_sort_col'] = 'per_room_gap'
        st.session_state['_expl_sort_dir'] = True   # True = ascending

    _sc1, _sc2, _ = st.columns([2.5, 1, 7])
    with _sc1:
        _new_col = st.selectbox('Sort', options=list(_SORT_OPTS.keys()),
                                format_func=lambda k: _SORT_OPTS[k],
                                index=list(_SORT_OPTS.keys()).index(
                                    st.session_state['_expl_sort_col']),
                                key='_expl_sc', label_visibility='collapsed')
    with _sc2:
        _asc = st.toggle('↑', value=st.session_state['_expl_sort_dir'],
                         key='_expl_asc', help='Ascending')

    # Sync sort state from widget values to avoid an extra rerun.
    st.session_state['_expl_sort_col'] = _new_col
    st.session_state['_expl_sort_dir'] = _asc

    _disp = (filtered
             .sort_values(_new_col, ascending=_asc)
             .reset_index())

    # Pagination state, 25 listings per page.
    _N = 25
    _total   = len(_disp)
    _n_pages = max(1, (_total + _N - 1) // _N)
    if '_expl_page' not in st.session_state:
        st.session_state['_expl_page'] = 0
    # Reset to page 0 when the filter result count changes.
    if st.session_state.get('_expl_last_total') != _total:
        st.session_state['_expl_page'] = 0
        st.session_state['_expl_last_total'] = _total
    _page = min(st.session_state['_expl_page'], _n_pages - 1)
    _page_df = _disp.iloc[_page * _N : (_page + 1) * _N]

    # Header showing the total listing count and the current page range.
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0 2px">'
        f'<span style="font-size:1.25rem;font-weight:700;color:#0d3321;'
        f'font-family:Space Grotesk,sans-serif">Explorer</span>'
        f'<span style="background:#0d3321;color:#a5d6a7;font-size:12px;font-weight:600;'
        f'padding:3px 11px;border-radius:20px">{len(filtered)} listings</span></div>'
        f'<p style="font-size:12px;color:#4a7c5a;margin:0 0 8px">'
        f'{"No results" if _total == 0 else f"Showing {_page*_N+1}–{min((_page+1)*_N, _total)}"} · '
        f'click ♥ to save to Favourites</p>',
        unsafe_allow_html=True)

    # Helper that returns a coloured valuation gap badge as an HTML string.
    def _gap_badge(gap):
        if gap < -5:  bg, fg = '#d4e8d8', '#0d3321'
        elif gap > 5: bg, fg = '#fde8e8', '#7a1c1c'
        else:         bg, fg = '#fef3cd', '#6b4c00'
        sign = '+' if gap >= 0 else ''
        return (f'<span style="background:{bg};color:{fg};font-size:11px;font-weight:600;'
                f'padding:2px 8px;border-radius:20px;white-space:nowrap">'
                f'{sign}{gap:.1f}%</span>')

    # Column headers for the explorer table.
    _COL_W  = [0.35, 2.4, 0.7, 0.7, 0.75, 0.8, 0.7, 0.6]
    _LABELS = ['', 'Address', 'Rent/room', 'Predicted', 'Gap', 'Full rent', 'Size', 'Link']
    _hcols  = st.columns(_COL_W)
    for _hc, _lbl in zip(_hcols, _LABELS):
        _hc.markdown(
            f'<div style="font-size:11px;font-weight:600;color:#4a7c5a;'
            f'text-transform:uppercase;letter-spacing:.06em;padding:4px 0 2px">'
            f'{_lbl}</div>',
            unsafe_allow_html=True)
    st.markdown('<div style="border-top:2px solid #d4e8d8;margin-bottom:2px"></div>',
                unsafe_allow_html=True)

    # One row per listing on the current page.
    for _, row in _page_df.iterrows():
        _orig = row['index']
        _is_fav = _orig in st.session_state.favorites
        _area = (f"{row['living_space']:.0f} m²"
                 if pd.notna(row.get('living_space')) and row['living_space'] > 0 else 'n/a')
        _rc = st.columns(_COL_W)
        with _rc[0]:
            if st.button('♥' if _is_fav else '♡',
                         key=f'_h{_orig}',
                         help='Remove from Favourites' if _is_fav else 'Save to Favourites'):
                if _is_fav:
                    st.session_state.favorites.discard(_orig)
                else:
                    st.session_state.favorites.add(_orig)
                st.rerun()
        _rc[1].markdown(
            f'<div style="font-weight:600;color:#0d3321;font-size:13px;'
            f'line-height:1.3">{row["street"]}</div>'
            f'<div style="color:#6a9a78;font-size:12px">{row["locality"]}</div>',
            unsafe_allow_html=True)
        _rc[2].markdown(
            f'<span style="font-weight:700;color:#0d3321;font-size:13px">'
            f'CHF {int(row["per_room_rent"]):,}</span>',
            unsafe_allow_html=True)
        _rc[3].markdown(
            f'<span style="color:#4a7c5a;font-size:12px">'
            f'CHF {int(row["predicted_per_room_rent"]):,}</span>',
            unsafe_allow_html=True)
        _rc[4].markdown(_gap_badge(row['per_room_gap']), unsafe_allow_html=True)
        _rc[5].markdown(
            f'<span style="color:#4a7c5a;font-size:12px">'
            f'CHF {int(row["rent_gross"]):,}/mo</span>',
            unsafe_allow_html=True)
        _rc[6].markdown(
            f'<span style="color:#4a7c5a;font-size:12px">'
            f'{row["rooms"]:.0f} rm · {_area}</span>',
            unsafe_allow_html=True)
        _rc[7].markdown(
            f'<a href="{row["homegate_url"]}" target="_blank" '
            f'style="color:#1a5c2a;font-size:12px;font-weight:600;text-decoration:none">'
            f'View ↗</a>',
            unsafe_allow_html=True)
        st.markdown('<div style="border-top:1px solid #eef4ee"></div>',
                    unsafe_allow_html=True)

    # Previous and next buttons for page navigation.
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    _pc1, _pc2, _pc3 = st.columns([1, 2, 1])
    with _pc1:
        if st.button('← Prev', key='_expl_prev', disabled=(_page == 0)):
            st.session_state['_expl_page'] = _page - 1
            st.rerun()
    with _pc2:
        st.markdown(
            f'<p style="text-align:center;font-size:12px;color:#4a7c5a;margin-top:6px">'
            f'Page {_page+1} of {_n_pages}</p>',
            unsafe_allow_html=True)
    with _pc3:
        if st.button('Next →', key='_expl_next', disabled=(_page >= _n_pages - 1)):
            st.session_state['_expl_page'] = _page + 1
            st.rerun()


    # Cards showing the 5 listings with the lowest per-room valuation gap.
    st.markdown('<hr class="pg-divider"><div class="pg-section-label"><span class="pg-section-label-text">Top 5 best deals</span><div class="pg-section-label-line"></div></div>', unsafe_allow_html=True)
    top5 = filtered.nsmallest(5, "per_room_gap")
    cards = ""
    rank_colors = ["#f0c040", "#b0b8c8", "#cd7f32", "#4a7c5a", "#4a7c5a"]
    for rank, (_, r) in enumerate(top5.iterrows(), 1):
        area = (f"{r['living_space']:.0f} m²"
                if pd.notna(r.get("living_space")) and r["living_space"] > 0 else "n/a")
        medal = rank_colors[rank - 1]
        cards += (
            f'<div style="background:#ffffff;border:1px solid #e0ede4;border-radius:16px;'
            f'padding:18px;box-shadow:0 2px 10px rgba(13,51,33,0.06);'
            f'transition:box-shadow 0.2s">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
            f'<div style="width:30px;height:30px;border-radius:50%;background:{medal};'
            f'color:#fff;font-family:Space Grotesk,sans-serif;font-weight:800;font-size:13px;'
            f'display:flex;align-items:center;justify-content:center">#{rank}</div>'
            f'{_gap_badge(r["per_room_gap"])}'
            f'</div>'
            f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:800;'
            f'color:#0d3321;letter-spacing:-0.03em;margin-bottom:2px">'
            f'CHF {int(r["per_room_rent"]):,}'
            f'<span style="font-size:0.85rem;font-weight:500;color:#4a7c5a"> /room</span>'
            f'</div>'
            f'<div style="font-family:Inter,sans-serif;font-weight:600;font-size:13px;'
            f'color:#0d3321;margin-bottom:2px">{r["street"]}</div>'
            f'<div style="font-family:Inter,sans-serif;font-size:12px;color:#6a9a78;margin-bottom:10px">'
            f'{r["locality"]} · {r["listing_type"]} · {r["rooms"]:.0f} rm · {area} · Fl {int(r["floor"])}'
            f'</div>'
            f'<div style="font-size:11px;color:#81a890;margin-bottom:12px">'
            f'🎓 {r["walk_hsg_min"]:.0f} min &nbsp;·&nbsp; '
            f'🚎 {r["walk_transit_min"]:.0f} min &nbsp;·&nbsp; '
            f'🛍️ {r["walk_grocery_min"]:.0f} min'
            f'</div>'
            f'<a href="{r["homegate_url"]}" target="_blank" '
            f'style="display:inline-block;font-family:Inter,sans-serif;font-size:12px;'
            f'font-weight:600;color:#1a5c2a;background:#e8f5e9;border-radius:8px;'
            f'padding:5px 14px;text-decoration:none">View on homegate.ch ↗</a>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;'
        f'margin-top:4px;font-family:Inter,sans-serif">{cards}</div>',
        unsafe_allow_html=True,
    )

# PAGE – ANALYSIS DETAILS
elif page == "Analysis Details":
    import altair as alt

    st.markdown("""
    <div class="pg-header">
        <div class="pg-header-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#81c995" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
        </div>
        <div>
            <div class="pg-header-title">Analysis Details</div>
            <div class="pg-header-sub">311 listings · 82% model accuracy · Built for HSG students · Free to use</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Analysis Details page styles
    st.markdown("""
    <style>
    /* Re-inject shared header styles (for direct nav) */
    .pg-header {
        background: linear-gradient(135deg, #0a2e1a 0%, #0d3321 55%, #1a5c2a 100%);
        border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.4rem;
        display: flex; align-items: center; gap: 1.2rem;
        position: relative; overflow: hidden;
    }
    .pg-header::after {
        content: ''; position: absolute; right: -40px; top: -40px;
        width: 200px; height: 200px; border-radius: 50%;
        background: radial-gradient(circle, rgba(76,175,80,0.15) 0%, transparent 65%);
    }
    .pg-header-icon {
        width: 44px; height: 44px; flex-shrink: 0;
        background: rgba(255,255,255,0.08); border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
    }
    .pg-header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem; font-weight: 800; color: #ffffff;
        letter-spacing: -0.03em; line-height: 1.1;
    }
    .pg-header-sub { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #81c995; margin-top: 3px; }
    .ad-section {
        display: flex; align-items: baseline; gap: 10px;
        margin: 1.8rem 0 0.9rem; padding-bottom: 9px;
        border-bottom: 1.5px solid #e0ede4;
    }
    .ad-section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem; font-weight: 700; color: #0d3321; letter-spacing: -0.01em;
    }
    .ad-section-sub {
        font-family: 'Inter', sans-serif; font-size: 0.78rem; color: #6a9a78;
    }
    .ad-note {
        font-family: 'Inter', sans-serif; font-size: 0.83rem; color: #4a6e5a;
        padding: 9px 14px; background: #f3f8f4;
        border-left: 3px solid #81c995; border-radius: 0 8px 8px 0;
        margin-bottom: 1rem; line-height: 1.5;
    }
    .ad-kpi-grid { display: grid; gap: 10px; margin-bottom: 0.6rem; }
    .ad-kpi {
        background: #ffffff; border: 1px solid #e4ede8;
        border-radius: 12px; padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,40,15,0.05);
    }
    .ad-kpi-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem; font-weight: 800; color: #0a2217;
        letter-spacing: -0.03em; line-height: 1;
    }
    .ad-kpi-lbl {
        font-family: 'Inter', sans-serif; font-size: 0.67rem; color: #6a9a78;
        text-transform: uppercase; letter-spacing: 0.06em;
        font-weight: 600; margin-top: 6px;
    }
    .ad-kpi-accent { color: #e74c3c !important; }
    .ad-kpi-accent-amber { color: #b07a00 !important; }
    .ad-kpi-accent-blue  { color: #1a4a8a !important; }
    .ad-cap {
        font-family: 'Inter', sans-serif; font-size: 0.73rem;
        color: #8aaa96; margin: 4px 2px 8px;
    }
    /* Chart cards */
    [data-testid="stArrowVegaLiteChart"] {
        background: #ffffff; border: 1px solid #e4ede8;
        border-radius: 14px; padding: 1rem 0.6rem 0.5rem;
        box-shadow: 0 1px 4px rgba(0,40,15,0.05);
        margin-bottom: 0.5rem;
    }
    /* Tab pills */
    button[data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.83rem !important; font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Pipeline steps
    st.markdown('<div class="ad-section"><span class="ad-section-title">Pricing Model</span><span class="ad-section-sub">How the XGBoost model was built</span></div>', unsafe_allow_html=True)

    with st.spinner("Loading analysis…"):
        _mld = _compute_ml_details(_mtime)

    tab_pipe, tab_cv, tab_lc, tab_corr = st.tabs(
        ["Pipeline", "Cross-Validation", "Learning Curve", "Feature Correlations"]
    )

    # Tab showing the model pipeline as a numbered step list.
    with tab_pipe:
        st.markdown('<div class="ad-note">How the model goes from raw listings to a rent prediction for every apartment.</div>', unsafe_allow_html=True)

        _steps = [
            ("1", "#4a90d9", "Load data",
             f"{_mld['n_total']} listings · 10 features each"),
            ("2", "#7b68ee", "Detect outliers",
             "First-pass 5-fold CV → IQR rule on residuals"),
            ("3", "#e07b54", "Remove outliers",
             f"{_mld['n_outliers']} atypical listings removed → {_mld['n_clean']} remain"),
            ("4", "#50b86c", "5-fold CV on clean set",
             "Each listing predicted by a model that never saw it"),
            ("5", "#50b86c", "Train final XGBoost",
             f"800 trees · depth 4 · lr 0.02 · fitted on {_mld['n_clean']} listings"),
            ("6", "#4a90d9", "Predict all listings",
             f"All {_mld['n_total']} listings get a predicted rent (incl. removed outliers)"),
            ("7", "#f5a623", "Compute valuation gap",
             "(actual − predicted) / predicted × 100 % → flag ±5 % threshold"),
        ]

        html_steps = '<div style="font-family:Inter,sans-serif;padding:4px 0 8px">'
        for num, color, title, desc in _steps:
            html_steps += f"""
            <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:6px">
              <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0">
                <div style="width:28px;height:28px;border-radius:8px;background:{color};
                            color:#fff;font-weight:700;font-size:12px;display:flex;
                            align-items:center;justify-content:center;letter-spacing:-0.01em">{num}</div>
                {"<div style='width:2px;flex:1;min-height:14px;background:linear-gradient(" + color + ",transparent);margin-top:3px'></div>" if num != "7" else ""}
              </div>
              <div style="background:#fff;border:1px solid #e4ede8;border-radius:10px;
                          padding:10px 14px;flex:1;margin-bottom:4px;
                          box-shadow:0 1px 3px rgba(0,40,15,0.04)">
                <div style="font-weight:700;font-size:0.88rem;color:#0a2217;letter-spacing:-0.01em">{title}</div>
                <div style="color:#6a9a78;font-size:0.78rem;margin-top:2px;line-height:1.4">{desc}</div>
              </div>
            </div>"""
        html_steps += "</div>"
        st.html(html_steps)

        st.markdown(f"""
        <div class="ad-kpi-grid" style="grid-template-columns:repeat(4,1fr);margin-top:0.8rem">
          <div class="ad-kpi"><div class="ad-kpi-val">{_mld["n_total"]}</div><div class="ad-kpi-lbl">Total listings</div></div>
          <div class="ad-kpi"><div class="ad-kpi-val ad-kpi-accent">{_mld["n_outliers"]}</div><div class="ad-kpi-lbl">Outliers removed</div></div>
          <div class="ad-kpi"><div class="ad-kpi-val">{_mld["n_clean"]}</div><div class="ad-kpi-lbl">Training set</div></div>
          <div class="ad-kpi"><div class="ad-kpi-val ad-kpi-accent-blue">5</div><div class="ad-kpi-lbl">CV folds</div></div>
        </div>
        """, unsafe_allow_html=True)

    # Tab showing per-fold MAE and R² from cross-validation.
    with tab_cv:
        st.markdown('<div class="ad-note">Every listing is predicted exactly once, on a model that never trained on it. No data leakage.</div>', unsafe_allow_html=True)

        # Per-fold MAE and R² displayed as a row of cards.
        _fm_df = pd.DataFrame(_mld["fold_metrics"])
        _fm_cells = "".join(
            f'<div class="ad-kpi" style="flex:1;text-align:center">'
            f'<div class="ad-kpi-val" style="font-size:1.15rem">{row["Fold"]}</div>'
            f'<div style="font-family:Inter,sans-serif;font-size:0.82rem;color:#4a7c5a;margin-top:4px">CHF {row["MAE (CHF)"]:.0f}</div>'
            f'<div style="font-family:Inter,sans-serif;font-size:0.73rem;color:#6a9a78">R² {row["R²"]:.3f}</div>'
            f'</div>'
            for _, row in _fm_df.iterrows()
        )
        st.markdown(
            f'<div style="display:flex;gap:8px;margin-bottom:1rem">{_fm_cells}</div>',
            unsafe_allow_html=True,
        )

        mae_bar = (
            alt.Chart(_fm_df)
            .mark_bar()
            .encode(
                x=alt.X("Fold:N", title=""),
                y=alt.Y("MAE (CHF):Q", title="MAE (CHF/mo)", scale=alt.Scale(zero=False)),
                color=alt.Color("Fold:N", legend=None,
                                scale=alt.Scale(scheme="tableau10")),
                tooltip=["Fold:N",
                         alt.Tooltip("MAE (CHF):Q", title="MAE (CHF)", format=".1f"),
                         alt.Tooltip("R²:Q", format=".3f")],
            )
            .properties(height=220, title="MAE per Fold")
        )
        r2_bar = (
            alt.Chart(_fm_df)
            .mark_bar()
            .encode(
                x=alt.X("Fold:N", title=""),
                y=alt.Y("R²:Q", title="R²", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Fold:N", legend=None,
                                scale=alt.Scale(scheme="tableau10")),
                tooltip=["Fold:N",
                         alt.Tooltip("R²:Q", format=".3f"),
                         alt.Tooltip("MAE (CHF):Q", title="MAE (CHF)", format=".1f")],
            )
            .properties(height=220, title="R² per Fold")
        )
        st.altair_chart(alt.hconcat(mae_bar, r2_bar).resolve_scale(color="independent"),
                        width='stretch')


    # Tab showing training vs. validation MAE as tree count increases.
    with tab_lc:
        st.markdown('<div class="ad-note">Training vs. validation error as more trees are added. When the two lines converge, more trees stop helping.</div>', unsafe_allow_html=True)
        _lc_df = pd.DataFrame(_mld["lc_data"])

        # Average train and validation MAE across all folds.
        _lc_avg = (
            _lc_df.groupby("n_estimators")[["Train MAE", "Val MAE"]]
            .mean().reset_index()
        )
        _lc_long = _lc_avg.melt("n_estimators", var_name="Set", value_name="MAE")

        lc_chart = (
            alt.Chart(_lc_long)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("n_estimators:Q", title="Number of trees (n_estimators)"),
                y=alt.Y("MAE:Q",          title="MAE (CHF/mo)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Set:N",
                    scale=alt.Scale(domain=["Train MAE", "Val MAE"],
                                    range=["#4a90d9", "#e07b54"]),
                    legend=alt.Legend(title=""),
                ),
                tooltip=[
                    alt.Tooltip("n_estimators:Q", title="Trees"),
                    alt.Tooltip("Set:N",           title="Set"),
                    alt.Tooltip("MAE:Q",           title="MAE (CHF)", format=".1f"),
                ],
            )
            .properties(height=360)
        )
        # Individual fold lines shown faintly behind the average.
        lc_folds = (
            alt.Chart(_lc_df)
            .mark_line(strokeWidth=1, opacity=0.25, strokeDash=[4, 3])
            .encode(
                x=alt.X("n_estimators:Q"),
                y=alt.Y("Val MAE:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Fold:N", legend=None,
                                scale=alt.Scale(scheme="tableau10")),
            )
        )
        st.altair_chart((lc_folds + lc_chart).properties(height=360),
                        width='stretch')
        st.markdown('<div class="ad-cap">Solid lines = average across 5 folds · Faint dashed lines = individual folds</div>', unsafe_allow_html=True)

    # Tab showing Pearson correlation between each feature and rent.
    with tab_corr:
        st.markdown('<div class="ad-note">How strongly each input feature pushes rent up (green) or down (red).</div>', unsafe_allow_html=True)
        _corr_df = pd.DataFrame(_mld["corr_rows"]).sort_values("Correlation")
        corr_chart = (
            alt.Chart(_corr_df)
            .mark_bar()
            .encode(
                x=alt.X("Correlation:Q", title="Pearson Correlation with Rent",
                        scale=alt.Scale(domain=[-1, 1])),
                y=alt.Y("Feature:N", sort=alt.EncodingSortField("Correlation"),
                        title=""),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(domain=["positive", "negative"],
                                    range=["#2ecc71", "#e74c3c"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Feature:N",     title="Feature"),
                    alt.Tooltip("Correlation:Q", title="Correlation", format=".3f"),
                ],
            )
            .properties(height=320)
        )
        zero_line = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="black", strokeWidth=1)
            .encode(x="x:Q")
        )
        st.altair_chart(corr_chart + zero_line, width='stretch')
        st.markdown('<div class="ad-cap">Latitude/Longitude encode geographic position. Their high correlation reflects neighbourhood price effects.</div>', unsafe_allow_html=True)

    # Derived columns shared across all the charts below.
    _ad = df.dropna(subset=["living_space", "rent_gross", "predicted_rent"]).copy()
    _ad["residual"]    = _ad["rent_gross"] - _ad["predicted_rent"]
    _ad["abs_error"]   = _ad["residual"].abs()
    _ad["price_per_m2"] = (_ad["rent_gross"] / _ad["living_space"]).round(1)
    _ad["rooms_label"] = _ad["rooms"].apply(lambda r: f"{r:.1f} rooms")
    _mape = (_ad["abs_error"] / _ad["rent_gross"] * 100).mean()
    _within_10 = (_ad["abs_error"] / _ad["rent_gross"] < 0.10).mean() * 100
    _within_15 = (_ad["abs_error"] / _ad["rent_gross"] < 0.15).mean() * 100

    # Model performance KPIs from 5-fold cross-validation.
    st.markdown('<div class="ad-section"><span class="ad-section-title">Model Performance</span><span class="ad-section-sub">5-fold cross-validation · out-of-sample</span></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ad-kpi-grid" style="grid-template-columns:repeat(5,1fr)">
      <div class="ad-kpi"><div class="ad-kpi-val">{r2:.3f}</div><div class="ad-kpi-lbl">R² score</div></div>
      <div class="ad-kpi"><div class="ad-kpi-val ad-kpi-accent-blue">CHF {mae:.0f}</div><div class="ad-kpi-lbl">MAE / month</div></div>
      <div class="ad-kpi"><div class="ad-kpi-val ad-kpi-accent-amber">{_mape:.1f}%</div><div class="ad-kpi-lbl">MAPE</div></div>
      <div class="ad-kpi"><div class="ad-kpi-val">{_within_10:.0f}%</div><div class="ad-kpi-lbl">Within 10%</div></div>
      <div class="ad-kpi"><div class="ad-kpi-val">{_within_15:.0f}%</div><div class="ad-kpi-lbl">Within 15%</div></div>
    </div>
    <div class="ad-cap">All metrics from 5-fold cross-validation. Each listing predicted out-of-sample.</div>
    """, unsafe_allow_html=True)

    # XGBoost feature importance scores ranked by contribution to the model.
    st.markdown('<div class="ad-section"><span class="ad-section-title">Feature Importance</span><span class="ad-section-sub">What drives the rent predictions</span></div>', unsafe_allow_html=True)

    imp_df = (pd.DataFrame.from_dict(importances, orient="index", columns=["Importance"])
              .sort_values("Importance", ascending=True)
              .reset_index())
    imp_df.columns = ["Feature", "Importance"]
    imp_df["Importance %"] = (imp_df["Importance"] * 100).round(1)

    imp_chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X("Importance %:Q", title="Importance (%)"),
            y=alt.Y("Feature:N", sort="-x", title=""),
            color=alt.Color(
                "Importance %:Q",
                scale=alt.Scale(scheme="blues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Feature:N",       title="Feature"),
                alt.Tooltip("Importance %:Q",  title="Importance (%)", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    imp_text = imp_chart.mark_text(align="left", dx=4, fontSize=12).encode(
        text=alt.Text("Importance %:Q", format=".1f")
    )
    st.altair_chart(imp_chart + imp_text, width='stretch')

    st.markdown('<div class="ad-section"><span class="ad-section-title">Actual vs. Predicted Rent</span><span class="ad-section-sub">Above the diagonal = undervalued · below = overvalued</span></div>', unsafe_allow_html=True)

    # Scatter plot of actual vs. predicted rent, coloured by valuation label.
    _rent_min = int(_ad["predicted_rent"].min()) - 50
    _rent_max = int(max(_ad["rent_gross"].max(), _ad["predicted_rent"].max())) + 100

    avp_base = alt.Chart(_ad).encode(
        x=alt.X("predicted_rent:Q", title="Predicted Rent (CHF/mo)",
                scale=alt.Scale(domain=[_rent_min, _rent_max])),
        y=alt.Y("rent_gross:Q",     title="Actual Rent (CHF/mo)",
                scale=alt.Scale(domain=[_rent_min, _rent_max])),
    )
    avp_diagonal = (
        alt.Chart(pd.DataFrame({"x": [_rent_min, _rent_max], "y": [_rent_min, _rent_max]}))
        .mark_line(color="#aaaaaa", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )
    avp_points = avp_base.mark_point(filled=True, size=50, opacity=0.7).encode(
        color=alt.Color(
            "valuation_label:N",
            scale=alt.Scale(
                domain=["overvalued", "fair", "undervalued"],
                range=["#e74c3c", "#f39c12", "#2ecc71"]),
            legend=alt.Legend(title="Valuation"),
        ),
        tooltip=[
            alt.Tooltip("street:N",          title="Address"),
            alt.Tooltip("rent_gross:Q",       title="Actual (CHF)",    format=",d"),
            alt.Tooltip("predicted_rent:Q",   title="Predicted (CHF)", format=",d"),
            alt.Tooltip("valuation_gap:Q",    title="Gap (%)",         format=".1f"),
            alt.Tooltip("rooms:Q",            title="Rooms"),
            alt.Tooltip("living_space:Q",     title="Area (m²)"),
        ],
    )
    st.altair_chart((avp_diagonal + avp_points).properties(height=420), width='stretch')

    st.markdown('<div class="ad-section"><span class="ad-section-title">Prediction Error</span><span class="ad-section-sub">Residual = actual − predicted · centred near zero = well-calibrated</span></div>', unsafe_allow_html=True)

    # Histogram of prediction residuals (actual − predicted).
    _ad["residual_sign"] = _ad["residual"].apply(lambda r: "positive" if r > 0 else "negative")
    res_hist = (
        alt.Chart(_ad)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("residual:Q",
                    bin=alt.Bin(maxbins=40),
                    title="Residual (CHF/mo)  [actual − predicted]"),
            y=alt.Y("count()", title="Listings"),
            color=alt.Color(
                "residual_sign:N",
                scale=alt.Scale(
                    domain=["positive", "negative"],
                    range=["#2ecc71", "#e74c3c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("residual:Q",  bin=alt.Bin(maxbins=40), title="Residual range"),
                alt.Tooltip("count()",     title="Listings"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(res_hist, width='stretch')

    _bias = _ad["residual"].mean()
    st.markdown(f"""
    <div class="ad-kpi-grid" style="grid-template-columns:repeat(3,1fr)">
      <div class="ad-kpi"><div class="ad-kpi-val">CHF {_bias:+.0f}</div><div class="ad-kpi-lbl">Mean bias</div></div>
      <div class="ad-kpi"><div class="ad-kpi-val">CHF {_ad["residual"].std():.0f}</div><div class="ad-kpi-lbl">Std deviation</div></div>
      <div class="ad-kpi"><div class="ad-kpi-val ad-kpi-accent">CHF {_ad["abs_error"].max():.0f}</div><div class="ad-kpi-lbl">Max error</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ad-section"><span class="ad-section-title">Rent vs. Living Space</span><span class="ad-section-sub">Coloured by valuation · dashed line = trend</span></div>', unsafe_allow_html=True)

    # Scatter plot of rent vs. living area coloured by valuation label.
    scatter_df = _ad[["living_space", "rent_gross", "valuation_label",
                       "street", "predicted_rent", "valuation_gap", "rooms"]].copy()
    scatter = (
        alt.Chart(scatter_df)
        .mark_point(filled=True, size=55, opacity=0.75)
        .encode(
            x=alt.X("living_space:Q", title="Living Space (m²)"),
            y=alt.Y("rent_gross:Q",   title="Rent (CHF/mo)"),
            color=alt.Color(
                "valuation_label:N",
                scale=alt.Scale(
                    domain=["overvalued", "fair", "undervalued"],
                    range=["#e74c3c", "#f39c12", "#2ecc71"]),
                legend=alt.Legend(title="Valuation")),
            tooltip=[
                alt.Tooltip("street:N",          title="Address"),
                alt.Tooltip("rent_gross:Q",       title="Rent (CHF)",      format=",d"),
                alt.Tooltip("predicted_rent:Q",   title="Predicted (CHF)", format=",d"),
                alt.Tooltip("valuation_gap:Q",    title="Gap (%)",         format=".1f"),
                alt.Tooltip("living_space:Q",     title="Area (m²)"),
                alt.Tooltip("rooms:Q",            title="Rooms"),
            ],
        )
        .properties(height=380)
    )
    # Regression trend line overlaid on the scatter plot.
    trend = scatter.transform_regression(
        "living_space", "rent_gross"
    ).mark_line(color="#555555", strokeDash=[4, 3], strokeWidth=1.5)
    st.altair_chart((scatter + trend), width='stretch')

    st.markdown('<div class="ad-section"><span class="ad-section-title">Rent by Room Count</span><span class="ad-section-sub">Each dot is a listing · black tick = median</span></div>', unsafe_allow_html=True)

    # Strip plot of rent by room count with a median tick for each group.
    _room_counts = _ad["rooms"].value_counts()
    _room_keep   = _room_counts[_room_counts >= 5].index
    _rooms_df    = _ad[_ad["rooms"].isin(_room_keep)].copy()
    _rooms_df["rooms_str"] = _rooms_df["rooms"].apply(lambda r: f"{r:.1f}")

    rooms_strip = (
        alt.Chart(_rooms_df)
        .mark_point(filled=True, size=40, opacity=0.5)
        .encode(
            x=alt.X("rooms_str:N",   title="Rooms",
                    sort=[f"{r:.1f}" for r in sorted(_room_keep)]),
            y=alt.Y("rent_gross:Q",  title="Rent (CHF/mo)"),
            color=alt.Color(
                "valuation_label:N",
                scale=alt.Scale(
                    domain=["overvalued", "fair", "undervalued"],
                    range=["#e74c3c", "#f39c12", "#2ecc71"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("street:N",       title="Address"),
                alt.Tooltip("rent_gross:Q",   title="Rent (CHF)", format=",d"),
                alt.Tooltip("rooms_str:N",    title="Rooms"),
                alt.Tooltip("living_space:Q", title="Area (m²)"),
            ],
        )
    )
    rooms_median = (
        alt.Chart(_rooms_df)
        .mark_tick(color="black", thickness=2, size=30)
        .encode(
            x=alt.X("rooms_str:N",
                    sort=[f"{r:.1f}" for r in sorted(_room_keep)]),
            y=alt.Y("median(rent_gross):Q"),
            tooltip=[
                alt.Tooltip("rooms_str:N",             title="Rooms"),
                alt.Tooltip("median(rent_gross):Q",    title="Median rent (CHF)", format=",d"),
            ],
        )
    )
    st.altair_chart((rooms_strip + rooms_median).properties(height=340), width='stretch')
    st.markdown('<div class="ad-cap">Black tick = median rent · Only room counts with ≥ 5 listings shown</div>', unsafe_allow_html=True)

    st.markdown('<div class="ad-section"><span class="ad-section-title">Valuation Gap</span><span class="ad-section-sub">Beyond ±5% = flagged · dashed lines mark the thresholds</span></div>', unsafe_allow_html=True)

    # Histogram of valuation gaps across all listings.
    _gap_df = df[["valuation_gap", "valuation_label"]].copy()
    gap_hist = (
        alt.Chart(_gap_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("valuation_gap:Q",
                    bin=alt.Bin(step=5),
                    title="Valuation Gap (%)"),
            y=alt.Y("count()", title="Listings"),
            color=alt.Color(
                "valuation_label:N",
                scale=alt.Scale(
                    domain=["overvalued", "fair", "undervalued"],
                    range=["#e74c3c", "#f39c12", "#2ecc71"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("valuation_gap:Q", bin=alt.Bin(step=5), title="Gap range (%)"),
                alt.Tooltip("count()", title="Listings"),
            ],
        )
        .properties(height=260)
    )
    ref_lines = alt.Chart(pd.DataFrame({"x": [-5, 5]})).mark_rule(
        color="black", strokeDash=[5, 4], strokeWidth=1.5
    ).encode(x="x:Q")
    st.altair_chart((gap_hist + ref_lines), width='stretch')

    _ov = (df["valuation_label"] == "overvalued").sum()
    _fa = (df["valuation_label"] == "fair").sum()
    _un = (df["valuation_label"] == "undervalued").sum()
    st.markdown(f"""
    <div class="ad-kpi-grid" style="grid-template-columns:repeat(3,1fr)">
      <div class="ad-kpi">
        <div class="ad-kpi-val ad-kpi-accent">{_ov} <span style="font-size:0.9rem;font-weight:500;color:#999">({_ov/len(df)*100:.0f}%)</span></div>
        <div class="ad-kpi-lbl">Overvalued &gt; +5%</div>
      </div>
      <div class="ad-kpi">
        <div class="ad-kpi-val ad-kpi-accent-amber">{_fa} <span style="font-size:0.9rem;font-weight:500;color:#999">({_fa/len(df)*100:.0f}%)</span></div>
        <div class="ad-kpi-lbl">Fair −5% to +5%</div>
      </div>
      <div class="ad-kpi">
        <div class="ad-kpi-val">{_un} <span style="font-size:0.9rem;font-weight:500;color:#999">({_un/len(df)*100:.0f}%)</span></div>
        <div class="ad-kpi-lbl">Undervalued &lt; −5%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ad-section"><span class="ad-section-title">Price per m² by Neighbourhood</span><span class="ad-section-sub">Average gross rent per m² · postal codes with ≥ 3 listings</span></div>', unsafe_allow_html=True)

    # Bar chart of average price per m² grouped by postal code.
    _pc_df = (
        _ad.groupby("postal_code")
        .agg(
            avg_price_m2=("price_per_m2", "mean"),
            median_rent=("rent_gross",    "median"),
            count=("rent_gross",          "count"),
        )
        .reset_index()
        .sort_values("avg_price_m2", ascending=False)
    )
    _pc_df["postal_code"] = _pc_df["postal_code"].astype(str)
    _pc_df["avg_price_m2"] = _pc_df["avg_price_m2"].round(1)

    pc_chart = (
        alt.Chart(_pc_df[_pc_df["count"] >= 3])
        .mark_bar()
        .encode(
            x=alt.X("avg_price_m2:Q", title="Avg. price per m² (CHF/mo)"),
            y=alt.Y("postal_code:N",   sort="-x", title="Postal Code"),
            color=alt.Color(
                "avg_price_m2:Q",
                scale=alt.Scale(scheme="orangered"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("postal_code:N",   title="Postal Code"),
                alt.Tooltip("avg_price_m2:Q",  title="Avg CHF/m²",     format=".1f"),
                alt.Tooltip("median_rent:Q",   title="Median rent (CHF)", format=",d"),
                alt.Tooltip("count:Q",         title="Listings"),
            ],
        )
        .properties(height=max(200, len(_pc_df[_pc_df["count"] >= 3]) * 28))
    )
    pc_text = pc_chart.mark_text(align="left", dx=4, fontSize=11).encode(
        text=alt.Text("avg_price_m2:Q", format=".1f")
    )
    st.altair_chart(pc_chart + pc_text, width='stretch')
    st.markdown('<div class="ad-cap">Only postal codes with ≥ 3 listings shown</div>', unsafe_allow_html=True)

# PAGE – RENT ESTIMATOR
elif page == "Rent Estimator":
    import math
    import folium
    from streamlit_folium import st_folium

    st.markdown("""
    <style>
    /* Shared page-level styles (re-injected for direct nav) */
    .pg-header {
        background: linear-gradient(135deg, #0a2e1a 0%, #0d3321 55%, #1a5c2a 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .pg-header::after {
        content: '';
        position: absolute;
        right: -40px; top: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(76,175,80,0.15) 0%, transparent 65%);
    }
    .pg-section-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.4rem 0 0.7rem 0;
    }
    .pg-section-label-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #0d3321;
        letter-spacing: -0.01em;
    }
    .pg-section-label-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #d4e8d8, transparent);
    }
    .pg-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #c8e6c9, transparent);
        margin: 1.4rem 0;
        border: none;
    }

    /* Estimator header */
    .est-header {
        background: linear-gradient(135deg, #081f12 0%, #0d3321 45%, #1a5c2a 100%);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .est-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(76,175,80,0.18) 0%, transparent 65%);
    }
    .est-header::after {
        content: '';
        position: absolute;
        bottom: -70px; left: 30%;
        width: 280px; height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(26,92,42,0.3) 0%, transparent 65%);
    }
    .est-header-left {
        position: relative;
        z-index: 1;
    }
    .est-header-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        color: #4caf7d;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 6px;
    }
    .est-header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #fff;
        letter-spacing: -0.04em;
        line-height: 1.05;
        margin-bottom: 6px;
    }
    .est-header-title span {
        background: linear-gradient(90deg, #4caf50, #81c995);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .est-header-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #81c995;
        max-width: 340px;
    }

    /* Steps */
    .est-steps {
        display: flex;
        align-items: center;
        gap: 0;
        position: relative;
        z-index: 1;
        flex-shrink: 0;
    }
    .est-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        width: 110px;
    }
    .est-step-bubble {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(255,255,255,0.10);
        border: 1.5px solid rgba(76,175,80,0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 8px;
    }
    .est-step-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        color: #4caf7d;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 2px;
    }
    .est-step-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #c8e6c9;
        font-weight: 500;
        line-height: 1.3;
    }
    .est-step-arrow {
        color: rgba(76,175,80,0.5);
        font-size: 1.1rem;
        margin-bottom: 28px;
        flex-shrink: 0;
    }
    </style>

    <div class="est-header">
        <div class="est-header-left">
            <div class="est-header-eyebrow">HouSinG · St. Gallen 2026</div>
            <div class="est-header-title">Rent <span>Estimator</span></div>
            <div class="est-header-sub">Get an instant ML-powered fair-market rent estimate for any address in St. Gallen.</div>
        </div>
        <div class="est-steps">
            <div class="est-step">
                <div class="est-step-bubble"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg></div>
                <div class="est-step-num">Step 1</div>
                <div class="est-step-label">Pin your location on the map</div>
            </div>
            <div class="est-step-arrow">›</div>
            <div class="est-step">
                <div class="est-step-bubble"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg></div>
                <div class="est-step-num">Step 2</div>
                <div class="est-step-label">Set rooms, size & floor</div>
            </div>
            <div class="est-step-arrow">›</div>
            <div class="est-step">
                <div class="est-step-bubble"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg></div>
                <div class="est-step-num">Step 3</div>
                <div class="est-step-label">Get your fair-rent estimate</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load POI locations for the map markers.
    _poi_path_est = os.path.join(os.path.dirname(__file__), "pois.json")
    pois_est = load_pois(poi_version=os.path.getmtime(_poi_path_est))

    # Walking distance helpers using OSRM, with haversine × 1.25 as a fallback.
    def _hav(lat1, lng1, lat2, lng2):
        import math
        R = 6371000
        p1, p2 = math.radians(lat1), math.radians(lat2)
        a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
             + math.cos(p1) * math.cos(p2)
             * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _osrm_walk(lat1, lng1, lat2, lng2):
        """OSRM walking distance (m); fallback = haversine × 1.25."""
        import urllib.request, json as _json
        url = (f"https://routing.openstreetmap.de/routed-foot/route/v1/foot/"
               f"{lng1:.6f},{lat1:.6f};{lng2:.6f},{lat2:.6f}?overview=false")
        try:
            with urllib.request.urlopen(url, timeout=6) as resp:
                data = _json.loads(resp.read())
            if data.get("code") == "Ok" and data.get("routes"):
                return float(data["routes"][0]["distance"])
        except Exception:
            pass
        return _hav(lat1, lng1, lat2, lng2) * 1.25

    def _nearest_walk(lat, lng, poi_list):
        """Walking distance to closest POI (nearest by haversine first)."""
        best = min(poi_list, key=lambda p: _hav(lat, lng, p["lat"], p["lng"]))
        return _osrm_walk(lat, lng, best["lat"], best["lng"])

    HSG_LAT,    HSG_LNG    = 47.431759683827714, 9.374557836074315
    CENTER_LAT, CENTER_LNG = 47.4232, 9.3772

    # Retrieve the current pin location from session state.
    pin = st.session_state.get("estimator_pin", None)

    # Base map with ESRI satellite tiles and POI markers.
    ESRI    = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}")
    OUTLINE = ("text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,"
               "-1px 1px 0 #fff,1px 1px 0 #fff")

    def est_poi_marker(m, p, emoji, font_size, fallback=""):
        name = p.get("name", "").strip() or fallback
        sz   = font_size + 10
        folium.Marker(
            location=[p["lat"], p["lng"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:{font_size}px;{OUTLINE};line-height:1">{emoji}</div>',
                icon_size=(sz, sz), icon_anchor=(sz // 2, sz // 2),
            ),
            tooltip=name or None,
        ).add_to(m)

    est_map = folium.Map(
        location=[pin["lat"], pin["lng"]] if pin else [47.4245, 9.3767],
        zoom_start=14,
        tiles=ESRI, attr="Tiles © Esri",
    )

    for g in pois_est["stops"]:
        est_poi_marker(est_map, g, '🚎', 18)
    for g in pois_est["groceries"]:
        est_poi_marker(est_map, g, '🛍️', 18)
    for g in pois_est["gyms"]:
        est_poi_marker(est_map, g, '🥊', 18)
    for g in pois_est["hsg"]:
        est_poi_marker(est_map, g, '🎓', 22, "University of St. Gallen")
    est_poi_marker(est_map, {"lat": CENTER_LAT, "lng": CENTER_LNG, "name": "City centre"}, '🏙️', 18)

    # Add a red home marker at the pinned location if one has been placed.
    if pin:
        folium.Marker(
            [pin["lat"], pin["lng"]],
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
            tooltip="📍 Your apartment",
            z_index_offset=1000,
        ).add_to(est_map)

    map_out = st_folium(
        est_map,
        use_container_width=True,
        height=460,
        returned_objects=["last_clicked"],
    )

    # Update the pin on map click and rerun to refresh the prediction.
    if map_out and map_out.get("last_clicked"):
        clicked = map_out["last_clicked"]
        if pin is None or abs(clicked["lat"] - pin["lat"]) > 1e-7 or abs(clicked["lng"] - pin["lng"]) > 1e-7:
            st.session_state["estimator_pin"] = {"lat": clicked["lat"], "lng": clicked["lng"]}
            st.rerun()

    if not pin:
        st.markdown("""
        <style>
        @keyframes pinPulse {
            0%   { box-shadow: 0 0 0 0 rgba(76,175,80,0.45); }
            70%  { box-shadow: 0 0 0 18px rgba(76,175,80,0); }
            100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
        }
        .pin-prompt {
            display: flex;
            align-items: center;
            gap: 1.4rem;
            background: linear-gradient(135deg, #f0faf2, #e6f4ea);
            border: 1.5px dashed #4caf50;
            border-radius: 16px;
            padding: 1.8rem 2rem;
            margin-top: 1rem;
            animation: pinPulse 2s ease-in-out infinite;
        }
        .pin-prompt-icon {
            font-size: 2.6rem;
            line-height: 1;
        }
        .pin-prompt-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #0d3321;
            margin-bottom: 4px;
        }
        .pin-prompt-sub {
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            color: #4a7c5a;
        }
        </style>
        <div class="pin-prompt">
            <div class="pin-prompt-icon"><svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#1a5c2a" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg></div>
            <div>
                <div class="pin-prompt-title">Drop your pin to get started</div>
                <div class="pin-prompt-sub">Click anywhere on the map above to place your apartment location.
                The model will instantly compute walking times and estimate your fair-market rent.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Fetch walking distances from the pin, cached by coords to avoid re-fetching on slider changes.
    p_lat, p_lng = pin["lat"], pin["lng"]
    _dist_key = (round(p_lat, 5), round(p_lng, 5))
    if st.session_state.get("_est_dist_key") != _dist_key:
        with st.spinner("Computing walking distances…"):
            st.session_state["_est_dist_key"]     = _dist_key
            st.session_state["_est_dist_transit"] = _nearest_walk(p_lat, p_lng, pois_est["stops"])
            st.session_state["_est_dist_grocery"] = _nearest_walk(p_lat, p_lng, pois_est["groceries"])
            st.session_state["_est_dist_gym"]     = _nearest_walk(p_lat, p_lng, pois_est["gyms"])
            st.session_state["_est_dist_hsg"]     = _osrm_walk(p_lat, p_lng, HSG_LAT, HSG_LNG)
            st.session_state["_est_dist_center"]  = _osrm_walk(p_lat, p_lng, CENTER_LAT, CENTER_LNG)
    dist_transit = st.session_state["_est_dist_transit"]
    dist_grocery = st.session_state["_est_dist_grocery"]
    dist_gym     = st.session_state["_est_dist_gym"]
    dist_hsg     = st.session_state["_est_dist_hsg"]
    dist_center  = st.session_state["_est_dist_center"]

    # Find the postal code of the nearest listing to the pin using vectorised haversine.
    dlat = np.radians(df["lat"].values - p_lat)
    dlng = np.radians(df["lng"].values - p_lng)
    a_   = (np.sin(dlat / 2) ** 2
            + np.cos(math.radians(p_lat)) * np.cos(np.radians(df["lat"].values))
            * np.sin(dlng / 2) ** 2)
    dists_to_listings = 6371000 * 2 * np.arctan2(np.sqrt(a_), np.sqrt(1 - a_))
    nearest_postal = int(df.loc[dists_to_listings.argmin(), "postal_code"])

    st.markdown('<hr class="pg-divider">', unsafe_allow_html=True)

    # Two-column layout: apartment specs on the left, prediction on the right.
    left_col, right_col = st.columns([5, 4], gap="large")

    with left_col:
        # Walking distance cards for transit, grocery, gym, HSG, and city centre.
        st.markdown(f"""
        <div class="pg-section-label">
            <span class="pg-section-label-text">Walking distances</span>
            <div class="pg-section-label-line"></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
                    margin-bottom:1.2rem;font-family:'Inter',sans-serif">
          <div style="background:#e3edf9;border:1px solid #aac4e8;border-radius:12px;
                      padding:12px 10px;text-align:center">
            <div style="line-height:1;margin-bottom:2px"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1a3a6b" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="14" rx="2"/><path d="M3 10h18"/><path d="M7 20v-2m10 2v-2"/><circle cx="7" cy="16" r="1" fill="#1a3a6b"/><circle cx="17" cy="16" r="1" fill="#1a3a6b"/></svg></div>
            <div style="font-size:1.2rem;font-weight:800;color:#1a3a6b;
                        font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
                {dist_transit / WALK_SPEED:.0f}<span style="font-size:0.7rem;font-weight:500"> min</span></div>
            <div style="font-size:0.65rem;color:#4a6fa0;text-transform:uppercase;
                        letter-spacing:0.07em;font-weight:600">Transit</div>
          </div>
          <div style="background:#fff8e1;border:1px solid #ffe082;border-radius:12px;
                      padding:12px 10px;text-align:center">
            <div style="line-height:1;margin-bottom:2px"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#997a00" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 1.99 1.61h9.72a2 2 0 0 0 1.99-1.61L23 6H6"/></svg></div>
            <div style="font-size:1.2rem;font-weight:800;color:#6b4c00;
                        font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
                {dist_grocery / WALK_SPEED:.0f}<span style="font-size:0.7rem;font-weight:500"> min</span></div>
            <div style="font-size:0.65rem;color:#997a00;text-transform:uppercase;
                        letter-spacing:0.07em;font-weight:600">Grocery</div>
          </div>
          <div style="background:#fce4ec;border:1px solid #f48fb1;border-radius:12px;
                      padding:12px 10px;text-align:center">
            <div style="line-height:1;margin-bottom:2px"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a0305a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="12" x2="18" y2="12"/><rect x="2" y="9" width="4" height="6" rx="1"/><rect x="18" y="9" width="4" height="6" rx="1"/><line x1="6" y1="11" x2="6" y2="13"/><line x1="18" y1="11" x2="18" y2="13"/></svg></div>
            <div style="font-size:1.2rem;font-weight:800;color:#7a1c3a;
                        font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
                {dist_gym / WALK_SPEED:.0f}<span style="font-size:0.7rem;font-weight:500"> min</span></div>
            <div style="font-size:0.65rem;color:#a0305a;text-transform:uppercase;
                        letter-spacing:0.07em;font-weight:600">Gym</div>
          </div>
          <div style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:12px;
                      padding:12px 10px;text-align:center">
            <div style="line-height:1;margin-bottom:2px"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0d3321" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg></div>
            <div style="font-size:1.2rem;font-weight:800;color:#0d3321;
                        font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
                {dist_hsg / WALK_SPEED:.0f}<span style="font-size:0.7rem;font-weight:500"> min</span></div>
            <div style="font-size:0.65rem;color:#1a5c2a;text-transform:uppercase;
                        letter-spacing:0.07em;font-weight:600">HSG</div>
          </div>
          <div style="background:#f3e5f5;border:1px solid #ce93d8;border-radius:12px;
                      padding:12px 10px;text-align:center">
            <div style="line-height:1;margin-bottom:2px"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4a1060" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg></div>
            <div style="font-size:1.2rem;font-weight:800;color:#4a1060;
                        font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
                {dist_center / WALK_SPEED:.0f}<span style="font-size:0.7rem;font-weight:500"> min</span></div>
            <div style="font-size:0.65rem;color:#7b1fa2;text-transform:uppercase;
                        letter-spacing:0.07em;font-weight:600">Centre</div>
          </div>
          <div style="background:#f6faf7;border:1px solid #c8e6c9;border-radius:12px;
                      padding:12px 10px;text-align:center;display:flex;flex-direction:column;
                      align-items:center;justify-content:center">
            <div style="font-size:0.65rem;color:#4a7c5a;text-transform:uppercase;
                        letter-spacing:0.07em;font-weight:600;margin-bottom:2px">Postal code</div>
            <div style="font-size:1.1rem;font-weight:800;color:#0d3321;
                        font-family:'Space Grotesk',sans-serif">{nearest_postal}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Sliders for apartment rooms, living area, and floor.
        st.markdown('<div class="pg-section-label"><span class="pg-section-label-text">Apartment specs</span><div class="pg-section-label-line"></div></div>', unsafe_allow_html=True)
        rooms        = st.slider("Rooms",             1.0, 7.0, 3.0, step=0.5)
        living_space = st.slider("Living space (m²)",  20,  200,  75)
        floor        = st.slider("Floor",               0,   15,   2)

        if st.button("📍 Move pin", width='stretch'):
            st.session_state.pop("estimator_pin", None)
            st.rerun()

    # Rent prediction output and nearby similar listings.
    with right_col:
        X_input   = np.array([[rooms, living_space, floor, p_lat, p_lng,
                                dist_transit, dist_grocery, dist_gym, dist_hsg,
                                dist_center]])
        predicted = model.predict(X_input)[0]

        # Compare the estimated rent to the median in the same postal code.
        _pc_listings = df[df["postal_code"].astype(int) == nearest_postal]["rent_gross"]
        _mkt_median  = _pc_listings.median() if len(_pc_listings) else predicted
        _vs_market   = ((predicted - _mkt_median) / _mkt_median * 100)
        _vs_label    = ("above" if _vs_market > 0 else "below")
        _vs_color    = ("#7a1c1c" if _vs_market > 0 else "#0d3321")
        _vs_bg       = ("#fde8e8" if _vs_market > 0 else "#e8f5e9")
        _vs_border   = ("#f5c6c6" if _vs_market > 0 else "#a5d6a7")

        st.markdown(f"""
        <style>
        @keyframes slideIn {{
            from {{ opacity:0; transform:translateY(12px); }}
            to   {{ opacity:1; transform:translateY(0); }}
        }}
        .est-result {{
            background: linear-gradient(135deg, #081f12 0%, #0d3321 50%, #1a5c2a 100%);
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
            animation: slideIn 0.4s ease both;
        }}
        .est-result::before {{
            content: '';
            position: absolute;
            top: -40px; right: -40px;
            width: 180px; height: 180px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(76,175,80,0.22) 0%, transparent 70%);
        }}
        .est-result::after {{
            content: '';
            position: absolute;
            bottom: -50px; left: 20%;
            width: 200px; height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(26,92,42,0.3) 0%, transparent 70%);
        }}
        </style>
        <div class="est-result">
            <div style="font-size:0.68rem;color:#81c995;text-transform:uppercase;
                        letter-spacing:0.13em;font-weight:600;margin-bottom:0.4rem;
                        font-family:'Inter',sans-serif;position:relative;z-index:1">
                Fair market rent estimate</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:2.8rem;font-weight:800;
                        color:#fff;letter-spacing:-0.04em;line-height:1;margin-bottom:0.3rem;
                        position:relative;z-index:1">
                CHF {predicted:,.0f}
                <span style="font-size:0.9rem;font-weight:500;color:#a5d6a7">/mo</span>
            </div>
            <div style="display:flex;gap:0.8rem;margin-top:0.9rem;flex-wrap:wrap;
                        position:relative;z-index:1">
                <div style="background:rgba(255,255,255,0.09);border:1px solid rgba(255,255,255,0.16);
                            border-radius:10px;padding:8px 14px">
                    <div style="font-size:0.62rem;color:#81c995;text-transform:uppercase;
                                letter-spacing:0.08em;font-weight:600;font-family:'Inter',sans-serif">Margin</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#fff;
                                font-family:'Space Grotesk',sans-serif">± CHF {mae:.0f}</div>
                </div>
                <div style="background:rgba(255,255,255,0.09);border:1px solid rgba(255,255,255,0.16);
                            border-radius:10px;padding:8px 14px">
                    <div style="font-size:0.62rem;color:#81c995;text-transform:uppercase;
                                letter-spacing:0.08em;font-weight:600;font-family:'Inter',sans-serif">Likely range</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#fff;
                                font-family:'Space Grotesk',sans-serif">
                        CHF {max(0,predicted-mae):,.0f}–{predicted+mae:,.0f}</div>
                </div>
            </div>
            <div style="margin-top:0.9rem;background:{_vs_bg};border:1px solid {_vs_border};
                        border-radius:10px;padding:8px 14px;display:inline-block;
                        position:relative;z-index:1">
                <span style="font-size:0.82rem;font-weight:600;color:{_vs_color};
                             font-family:'Inter',sans-serif">
                    {abs(_vs_market):.1f}% {_vs_label} the {nearest_postal} median
                    (CHF {_mkt_median:,.0f}/mo)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Cards showing similar listings in the same postal code and room range.
        st.markdown('<div class="pg-section-label"><span class="pg-section-label-text">Similar listings nearby</span><div class="pg-section-label-line"></div></div>', unsafe_allow_html=True)

        similar = df[
            (df["rooms"].between(rooms - 0.5, rooms + 0.5)) &
            (df["postal_code"].astype(int) == nearest_postal)
        ].copy().sort_values("valuation_gap")

        if len(similar):
            cards_html = ""
            for _, r in similar.iterrows():
                area = f"{r['living_space']:.0f} m²" if pd.notna(r.get("living_space")) and r["living_space"] > 0 else "n/a"
                gap  = r["valuation_gap"]
                if gap < -5:
                    gbg, gfg = "#e8f5e9", "#0d3321"
                elif gap > 5:
                    gbg, gfg = "#fde8e8", "#7a1c1c"
                else:
                    gbg, gfg = "#fef9e7", "#6b4c00"
                sign = "+" if gap >= 0 else ""
                cards_html += f"""
                <div style="background:#fff;border:1px solid #e0ede4;border-radius:12px;
                            padding:12px 14px;margin-bottom:8px;font-family:'Inter',sans-serif;
                            box-shadow:0 1px 6px rgba(13,51,33,0.05)">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;
                                margin-bottom:6px">
                        <div style="font-weight:600;font-size:13px;color:#0d3321;
                                    flex:1;margin-right:8px">{r["street"]}</div>
                        <span style="background:{gbg};color:{gfg};font-size:11px;font-weight:700;
                                     padding:2px 9px;border-radius:20px;white-space:nowrap;flex-shrink:0">
                            {sign}{gap:.1f}%</span>
                    </div>
                    <div style="display:flex;gap:1rem;font-size:12px;color:#5a7a65">
                        <span style="font-weight:700;color:#0d3321;font-family:'Space Grotesk',sans-serif">
                            CHF {int(r["rent_gross"]):,}/mo</span>
                        <span>·</span>
                        <span>{r["rooms"]:.0f} rooms · {area} · Fl {int(r["floor"])}</span>
                    </div>
                </div>"""
            st.markdown(
                f'<div style="max-height:340px;overflow-y:auto;padding-right:2px">{cards_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"""
            <div style="background:#f6faf7;border:1px dashed #c8e6c9;border-radius:12px;
                        padding:1.2rem;text-align:center;font-family:'Inter',sans-serif;
                        color:#4a7c5a;font-size:0.88rem">
                No listings found in {nearest_postal} with ~{rooms:.0f} rooms.<br>
                <span style="font-size:0.8rem;color:#81a890">Try adjusting the rooms slider.</span>
            </div>
            """, unsafe_allow_html=True)

# PAGE – FAVORITES & COMPARISON
elif page == "Favorites":

    # Page header and CSS for the Favorites page.
    st.markdown("""
    <style>
    .pg-header {
        background: linear-gradient(135deg, #0a2e1a 0%, #0d3321 55%, #1a5c2a 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .pg-header::after {
        content: '';
        position: absolute;
        right: -40px; top: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(76,175,80,0.15) 0%, transparent 65%);
    }
    .pg-header-icon  {
        width: 44px; height: 44px; flex-shrink: 0;
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
    }
    .pg-header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem; font-weight: 800; color: #ffffff;
        letter-spacing: -0.03em; line-height: 1.1;
    }
    .pg-header-sub { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #81c995; margin-top: 3px; }
    .pg-divider { border: none; border-top: 1px solid #e0ede4; margin: 1.4rem 0; }
    .pg-section-label { display: flex; align-items: center; gap: 10px; margin: 1.4rem 0 0.7rem 0; }
    .pg-section-label-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem; font-weight: 700; color: #0d3321; letter-spacing: -0.01em;
    }
    .pg-section-label-line { flex: 1; height: 1px; background: linear-gradient(90deg, #c8e6c9, transparent); }
    </style>
    """, unsafe_allow_html=True)

    _fav_count = len(st.session_state.favorites)
    st.markdown(f"""
    <div class="pg-header">
        <div class="pg-header-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#81c995" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
        </div>
        <div>
            <div class="pg-header-title">Favorites</div>
            <div class="pg-header-sub">{_fav_count} listing{"s" if _fav_count != 1 else ""} saved · Compare side by side</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if _fav_count == 0:
        st.markdown("""
        <div style="background:#f6faf7;border:1px dashed #c8e6c9;border-radius:16px;
                    padding:2.5rem 2rem;text-align:center;font-family:'Inter',sans-serif;margin-top:1rem">
            <div style="margin-bottom:14px;display:flex;justify-content:center">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#a5d6a7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
            </div>
            <div style="font-size:1.1rem;font-weight:700;color:#0d3321;margin-bottom:6px;
                        font-family:'Space Grotesk',sans-serif">No saved listings yet</div>
            <div style="font-size:0.88rem;color:#4a7c5a;max-width:360px;margin:0 auto">
                Head to <b>Map &amp; Explorer</b>, click any listing on the map,
                and tap <b>Save to Favorites</b> in the popup to save it here.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("← Go to Map & Explorer"):
            st.session_state.page = "Map & Explorer"
            st.rerun()

    else:
        # Pull the favourited listing rows from the dataframe.
        _fav_indices = list(st.session_state.favorites)
        _fav_listings = df.loc[df.index.isin(_fav_indices)].copy()

        # Remove button for each individual saved listing.
        st.markdown('<div class="pg-section-label"><span class="pg-section-label-text">Saved listings</span><div class="pg-section-label-line"></div></div>', unsafe_allow_html=True)

        _remove_cols = st.columns(min(len(_fav_listings), 4))
        for _ci, (_idx, _r) in enumerate(_fav_listings.iterrows()):
            with _remove_cols[_ci % 4]:
                _area = f"{_r['living_space']:.0f} m²" if pd.notna(_r.get("living_space")) and _r["living_space"] > 0 else "n/a"
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #d4e8d8;border-radius:12px;'
                    f'padding:10px 12px;font-family:Inter,sans-serif;font-size:12px;color:#0d3321;'
                    f'margin-bottom:6px">'
                    f'<div style="font-weight:700;margin-bottom:2px;white-space:nowrap;overflow:hidden;'
                    f'text-overflow:ellipsis" title="{_r["street"]}">{_r["street"]}</div>'
                    f'<div style="color:#4a7c5a">{_r["locality"]} · CHF {int(_r["rent_gross"]):,}/mo</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("✕ Remove", key=f"rm_fav_{_idx}", width='stretch'):
                    st.session_state.favorites.discard(_idx)
                    st.rerun()

        if st.button("Clear all favorites", key="clear_all_fav"):
            st.session_state.favorites = set()
            st.rerun()

        # Side-by-side comparison table for all saved listings.
        st.markdown('<hr class="pg-divider">', unsafe_allow_html=True)
        st.markdown('<div class="pg-section-label"><span class="pg-section-label-text">Side-by-side comparison</span><div class="pg-section-label-line"></div></div>', unsafe_allow_html=True)

        _th_style = (
            "padding:12px 14px;font-family:'Space Grotesk',sans-serif;font-size:0.78rem;"
            "font-weight:700;letter-spacing:0.04em;text-transform:uppercase;color:#a5d6a7;"
            "background:#0d3321;border-right:1px solid #1a4a2a;white-space:nowrap"
        )
        _th_first = (
            "padding:12px 14px;font-family:'Space Grotesk',sans-serif;font-size:0.78rem;"
            "font-weight:700;letter-spacing:0.04em;text-transform:uppercase;color:#a5d6a7;"
            "background:#081f12;border-right:1px solid #1a4a2a;min-width:130px"
        )
        _td_attr = (
            "padding:10px 14px;font-family:'Inter',sans-serif;font-size:0.82rem;"
            "font-weight:600;color:#1e1e1e;background:#f9fcf9;border-right:1px solid #e0ede4;"
            "border-bottom:1px solid #e8f0e9;vertical-align:top"
        )
        _td_label = (
            "padding:10px 14px;font-family:'Space Grotesk',sans-serif;font-size:0.8rem;"
            "font-weight:600;color:#0d3321;background:#eaf4eb;border-right:1px solid #d4e8d8;"
            "border-bottom:1px solid #d4e8d8;white-space:nowrap;vertical-align:middle"
        )
        _td_attr_alt = (
            "padding:10px 14px;font-family:'Inter',sans-serif;font-size:0.82rem;"
            "font-weight:600;color:#1e1e1e;background:#ffffff;border-right:1px solid #e0ede4;"
            "border-bottom:1px solid #e8f0e9;vertical-align:top"
        )

        # Table header row with listing address and postal code.
        _header_cells = f'<th style="{_th_first}">Attribute</th>'
        for _idx, _r in _fav_listings.iterrows():
            _header_cells += (
                f'<th style="{_th_style}">'
                f'<div style="font-size:0.85rem;font-weight:800;color:#ffffff;'
                f'margin-bottom:2px;letter-spacing:-0.01em">{_r["street"]}</div>'
                f'<div style="font-size:0.72rem;font-weight:400;color:#81c995;'
                f'text-transform:none;letter-spacing:0">{_r["locality"]} · {int(_r["postal_code"])}</div>'
                f'</th>'
            )

        def _cmp_row(label, values_html, alt=False):
            _td = _td_attr_alt if alt else _td_attr
            cells = f'<td style="{_td_label}">{label}</td>'
            for v in values_html:
                cells += f'<td style="{_td}">{v}</td>'
            return f'<tr>{cells}</tr>'

        def _gap_html_fav(gap):
            if gap < -5:
                c, bg = "#0d5c22", "#d4edda"
            elif gap > 5:
                c, bg = "#7a1c1c", "#fde8e8"
            else:
                c, bg = "#6b4c00", "#fff3cd"
            sign = "+" if gap >= 0 else ""
            return (f'<span style="background:{bg};color:{c};font-size:11px;font-weight:700;'
                    f'padding:2px 9px;border-radius:20px;white-space:nowrap">{sign}{gap:.1f}%</span>')

        _rows = ""
        _rows += _cmp_row("Full rent / month",
            [f'<span style="font-size:1rem;font-weight:800;color:#0d3321">CHF {int(r["rent_gross"]):,}</span>'
             for _, r in _fav_listings.iterrows()])
        _rows += _cmp_row("Rent per room",
            [f'CHF {int(r["per_room_rent"]):,}' for _, r in _fav_listings.iterrows()], alt=True)
        _rows += _cmp_row("Predicted / room",
            [f'CHF {int(r["predicted_per_room_rent"]):,}' for _, r in _fav_listings.iterrows()])
        _rows += _cmp_row("vs. Market",
            [_gap_html_fav(r["per_room_gap"]) for _, r in _fav_listings.iterrows()], alt=True)
        _rows += _cmp_row("Type",
            [r["listing_type"] for _, r in _fav_listings.iterrows()])
        _rows += _cmp_row("Rooms",
            [f'{r["rooms"]:.0f}' for _, r in _fav_listings.iterrows()], alt=True)
        _rows += _cmp_row("Living area",
            [f'{r["living_space"]:.0f} m²' if pd.notna(r.get("living_space")) and r["living_space"] > 0 else "n/a"
             for _, r in _fav_listings.iterrows()])
        _rows += _cmp_row("Floor",
            [f'{int(r["floor"])}' for _, r in _fav_listings.iterrows()], alt=True)
        _rows += _cmp_row("Walk to HSG",
            [f'🎓 {r["walk_hsg_min"]:.0f} min' for _, r in _fav_listings.iterrows()])
        _rows += _cmp_row("Walk to transit",
            [f'🚎 {r["walk_transit_min"]:.0f} min' for _, r in _fav_listings.iterrows()], alt=True)
        _rows += _cmp_row("Walk to grocery",
            [f'🛍️ {r["walk_grocery_min"]:.0f} min' for _, r in _fav_listings.iterrows()])
        _rows += _cmp_row("Walk to gym",
            [f'🥊 {r["walk_gym_min"]:.0f} min' for _, r in _fav_listings.iterrows()], alt=True)
        _rows += _cmp_row("Listing",
            [f'<a href="{r["homegate_url"]}" target="_blank" '
             f'style="color:#1a5c2a;font-weight:600;font-size:12px;text-decoration:none">'
             f'View ↗</a>'
             for _, r in _fav_listings.iterrows()])

        st.markdown(
            f'<div style="border:1px solid #c8e6c9;border-radius:12px;overflow:hidden;'
            f'font-family:Inter,sans-serif;margin-bottom:1.5rem">'
            f'<div style="overflow-x:auto">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>{_header_cells}</tr></thead>'
            f'<tbody>{_rows}</tbody>'
            f'</table></div></div>',
            unsafe_allow_html=True,
        )

        # Highlight the saved listing with the lowest per-room valuation gap.
        _best = _fav_listings.loc[_fav_listings["per_room_gap"].idxmin()]
        _best_area = (f"{_best['living_space']:.0f} m²"
                      if pd.notna(_best.get("living_space")) and _best["living_space"] > 0 else "n/a")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#081f12,#0d3321,#1a5c2a);
                    border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;
                    font-family:'Inter',sans-serif;display:flex;align-items:center;gap:16px">
            <div style="font-size:2rem">🏆</div>
            <div>
                <div style="font-size:0.72rem;color:#81c995;text-transform:uppercase;
                            letter-spacing:0.08em;font-weight:600;margin-bottom:3px">Best deal in your favorites</div>
                <div style="font-size:1.1rem;font-weight:800;color:#ffffff;
                            font-family:'Space Grotesk',sans-serif;letter-spacing:-0.02em">
                    {_best["street"]} · CHF {int(_best["rent_gross"]):,}/mo
                </div>
                <div style="font-size:0.82rem;color:#a5d6a7;margin-top:2px">
                    {_best["locality"]} · {_best["listing_type"]} · {_best["rooms"]:.0f} rooms · {_best_area} ·
                    <b>{_best["per_room_gap"]:+.1f}% vs. market</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

