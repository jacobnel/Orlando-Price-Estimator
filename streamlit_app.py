"""
Zonda Price Estimator - single-file Streamlit app.

Everything (data loading, model training, caching, UI) lives in this one file
so it is easy to deploy. Run locally with:  streamlit run streamlit_app.py
"""

import os
import re
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ===========================================================================
# CONFIG
# ===========================================================================
DATA_PATH = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTFT2h5moD2HKINhL80EdKBZcIAdt9QRWOeODn5AtpbVfcPcQvPhG3_OQGp8q7hBY8l-1RosjHqUrMx/pub?output=csv"

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FEATURES = [
    "Unit Size", "Lot Size", "Bedrooms", "Bathrooms",
    "Sale Year", "Sale Month",
    "Beds_x_Sqft", "Baths_x_Sqft", "Beds_x_Baths", "Lot_per_Sqft",
]
CAT_FEATS = ["City", "Seller", "Zip Code"]
GLOBAL_KEY = "__global__"


# ===========================================================================
# DATA LOADING / CLEANING
# ===========================================================================
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, dtype={"Zip Code": str}, low_memory=False)
    df.columns = [c.replace("\r", "").strip() for c in df.columns]

    required = [
        "Unit Size", "Lot Size", "Sale Price", "Bathrooms", "Bedrooms",
        "City", "Sale Date", "Seller", "Zip Code", "Product Style",
    ]
    df = df[required].copy()

    for col in ["City", "Seller", "Zip Code", "Product Style"]:
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )

    for col in ["Unit Size", "Lot Size", "Sale Price", "Bathrooms", "Bedrooms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Sale Date"] = pd.to_datetime(df["Sale Date"], errors="coerce")

    df = df.dropna(
        subset=["Sale Price", "Sale Date", "Product Style", "City", "Zip Code"]
    ).copy()

    for col in ["Unit Size", "Lot Size", "Bathrooms", "Bedrooms"]:
        df[col] = df.groupby("Product Style")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())

    df = df[(df["Sale Price"] > 50000) & (df["Sale Price"] < 5000000)].copy()
    df = df[(df["Unit Size"] > 400) & (df["Unit Size"] < 10000)].copy()
    df = df[(df["Bathrooms"] > 0) & (df["Bedrooms"] > 0)].copy()

    df["Sale Year"] = df["Sale Date"].dt.year
    df["Sale Month"] = df["Sale Date"].dt.month
    df["Beds_x_Sqft"] = df["Bedrooms"] * df["Unit Size"]
    df["Baths_x_Sqft"] = df["Bathrooms"] * df["Unit Size"]
    df["Beds_x_Baths"] = df["Bedrooms"] * df["Bathrooms"]
    df["Lot_per_Sqft"] = df["Lot Size"] / df["Unit Size"].replace(0, np.nan)
    df["Lot_per_Sqft"] = df["Lot_per_Sqft"].replace([np.inf, -np.inf], np.nan)
    df["Lot_per_Sqft"] = df["Lot_per_Sqft"].fillna(df["Lot_per_Sqft"].median())

    cleaned = []
    for style in df["Product Style"].dropna().unique():
        sub = df[df["Product Style"] == style].copy()
        sub["price_per_sqft"] = sub["Sale Price"] / sub["Unit Size"]
        q1 = sub["price_per_sqft"].quantile(0.25)
        q3 = sub["price_per_sqft"].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        sub = sub[(sub["price_per_sqft"] >= low) & (sub["price_per_sqft"] <= high)].copy()
        cleaned.append(sub)
    df = pd.concat(cleaned, ignore_index=True)

    for col in ["City", "Seller", "Zip Code"]:
        vc = df[col].value_counts()
        rare = vc[vc <= 10].index
        df[col] = df[col].apply(lambda x: "other" if x in rare else x)

    return df.reset_index(drop=True)


# ===========================================================================
# TRAINING
# ===========================================================================
def train_models(df):
    style_models, style_rmses, style_medians = {}, {}, {}

    for style in sorted(df["Product Style"].dropna().unique()):
        df_sub = df[df["Product Style"] == style].copy()
        if len(df_sub) < 50:
            continue

        X = df_sub[FEATURES + CAT_FEATS]
        y = df_sub["Sale Price"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_pool = Pool(X_train, y_train, cat_features=CAT_FEATS)
        val_pool = Pool(X_val, y_val, cat_features=CAT_FEATS)

        model = CatBoostRegressor(
            iterations=900, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            loss_function="RMSE", eval_metric="RMSE", random_seed=42,
            early_stopping_rounds=40, verbose=False,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))
        style_models[style] = model
        style_rmses[style] = rmse
        style_medians[style] = {
            "lot_size": float(df_sub["Lot Size"].median()),
            "beds": float(df_sub["Bedrooms"].median()),
            "baths": float(df_sub["Bathrooms"].median()),
        }

    X = df[FEATURES + CAT_FEATS]
    y = df["Sale Price"]
    global_pool = Pool(X, y, cat_features=CAT_FEATS)
    global_model = CatBoostRegressor(
        iterations=700, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="RMSE", eval_metric="RMSE", random_seed=42, verbose=False,
    )
    global_model.fit(global_pool)

    style_models[GLOBAL_KEY] = global_model
    style_medians[GLOBAL_KEY] = {
        "lot_size": float(df["Lot Size"].median()),
        "beds": float(df["Bedrooms"].median()),
        "baths": float(df["Bathrooms"].median()),
    }
    return style_models, style_rmses, style_medians


def build_options(df, style_models):
    def uniq(col):
        return sorted(df[col].dropna().unique().tolist())
    styles = sorted([s for s in style_models.keys() if s != GLOBAL_KEY])
    return {"cities": uniq("City"), "sellers": uniq("Seller"),
            "zips": uniq("Zip Code"), "styles": styles}


# ===========================================================================
# CACHING (save / load trained models to disk)
# ===========================================================================
def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or "blank"


def save_artifacts(style_models, style_rmses, style_medians, options, directory=MODELS_DIR):
    os.makedirs(directory, exist_ok=True)
    manifest = {"models": {}, "rmses": style_rmses,
                "medians": style_medians, "options": options}
    for style, model in style_models.items():
        fname = f"model_{_safe_name(style)}.cbm"
        model.save_model(os.path.join(directory, fname))
        manifest["models"][style] = fname
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_artifacts(directory=MODELS_DIR):
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    style_models = {}
    for style, fname in manifest["models"].items():
        m = CatBoostRegressor()
        m.load_model(os.path.join(directory, fname))
        style_models[style] = m
    return style_models, manifest["rmses"], manifest["medians"], manifest["options"]


def get_or_train(directory=MODELS_DIR, force=False):
    if not force:
        cached = load_artifacts(directory)
        if cached is not None:
            return cached
    df = load_data()
    style_models, style_rmses, style_medians = train_models(df)
    options = build_options(df, style_models)
    try:
        save_artifacts(style_models, style_rmses, style_medians, options, directory)
    except Exception:
        pass  # read-only filesystem is fine; we just won't cache
    return style_models, style_rmses, style_medians, options


# ===========================================================================
# PREDICTION
# ===========================================================================
def prepare_row(city, seller, zipcode, sqft, lot_size, beds, baths, year, month):
    row = pd.DataFrame({
        "Unit Size": [sqft], "Lot Size": [lot_size],
        "Bedrooms": [beds], "Bathrooms": [baths],
        "Sale Year": [year], "Sale Month": [month],
        "Beds_x_Sqft": [beds * sqft], "Baths_x_Sqft": [baths * sqft],
        "Beds_x_Baths": [beds * baths],
        "Lot_per_Sqft": [lot_size / sqft if sqft else 0],
        "City": [city], "Seller": [seller], "Zip Code": [zipcode],
    })
    return row[FEATURES + CAT_FEATS]


def predict_price(style_models, style_medians, city, seller, style, zipcode,
                  sqft, beds=None, baths=None, lot_size=None, year=None, month=None):
    model = style_models.get(style, style_models[GLOBAL_KEY])
    med = style_medians.get(style, style_medians[GLOBAL_KEY])
    if beds is None:
        beds = med["beds"]
    if baths is None:
        baths = med["baths"]
    if lot_size is None:
        lot_size = med["lot_size"]
    if year is None:
        year = pd.Timestamp.today().year
    if month is None:
        month = pd.Timestamp.today().month
    city = city.strip() if isinstance(city, str) else city
    seller = seller.strip() if isinstance(seller, str) else seller
    zipcode = str(zipcode).strip()
    row = prepare_row(city, seller, zipcode, sqft, lot_size, beds, baths, year, month)
    return float(model.predict(row)[0])


# ===========================================================================
# STREAMLIT UI
# ===========================================================================
st.set_page_config(page_title="Zonda Price Estimator", page_icon="🏠", layout="centered")


@st.cache_resource(show_spinner="Loading / training models (one-time, ~1-2 min)…")
def load_models():
    return get_or_train()


style_models, style_rmses, style_medians, options = load_models()

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;1,9..144,500&display=swap');
      .zonda-eyebrow{font-family:monospace;font-size:.78rem;letter-spacing:.28em;
        text-transform:uppercase;color:#9c3b1b;}
      .zonda-title{font-family:'Fraunces',serif;font-size:2.6rem;line-height:1.05;
        margin:.1rem 0 .3rem;color:#20190f;}
      .zonda-title em{font-style:italic;color:#9c3b1b;}
      .zonda-lede{color:#6a5f4d;max-width:46ch;margin-bottom:.5rem;}
      .price-card{background:#20190f;border-radius:6px;padding:30px 28px;margin-top:6px;}
      .price-tag{font-family:monospace;font-size:.72rem;letter-spacing:.24em;
        text-transform:uppercase;color:#d99c7d;}
      .price-value{font-family:'Fraunces',serif;font-size:3rem;line-height:1.05;
        color:#f3ede1;margin:.25rem 0;}
      .price-sub{color:#b6ab97;font-size:.9rem;}
      .price-meta{color:#b6ab97;font-size:.82rem;margin-top:1rem;
        border-top:1px solid rgba(255,255,255,.14);padding-top:.8rem;}
      .price-meta b{color:#f3ede1;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="zonda-eyebrow">— Comparable Sales Model</div>',
            unsafe_allow_html=True)
st.markdown('<div class="zonda-title">Estimate a home\'s <em>sale price.</em></div>',
            unsafe_allow_html=True)
st.markdown('<div class="zonda-lede">Fill in the property details. The model uses '
            'historical sales, segmented by product style, to estimate a likely '
            'price.</div>', unsafe_allow_html=True)

st.divider()

st.subheader("Location & Builder")
c1, c2 = st.columns(2)
city = c1.selectbox("City", [""] + options["cities"])
zipcode = c2.selectbox("Zip Code", [""] + options["zips"])
seller = st.selectbox("Seller / Builder", [""] + options["sellers"])

st.subheader("Product")
style = st.selectbox("Product Style", options["styles"])

st.subheader("Specifications")
sqft = st.number_input("Unit Size (sqft)  •  required",
                       min_value=0.0, value=None, step=50.0, placeholder="e.g. 2400")
s1, s2 = st.columns(2)
beds = s1.number_input("Bedrooms", min_value=0.0, value=None, step=1.0,
                       placeholder="median", help="Leave blank to use the median.")
baths = s2.number_input("Bathrooms", min_value=0.0, value=None, step=0.5,
                        placeholder="median", help="Leave blank to use the median.")
lot_size = st.number_input("Lot Size", min_value=0.0, value=None, step=100.0,
                           placeholder="median", help="Leave blank to use the median.")
y1, y2 = st.columns(2)
year = y1.number_input("Sale Year", min_value=1990, max_value=2100, value=None,
                       step=1, placeholder="current")
month = y2.number_input("Sale Month (1-12)", min_value=1, max_value=12, value=None,
                        step=1, placeholder="current")

st.write("")
go = st.button("Estimate Price", type="primary", use_container_width=True)

if go:
    if not sqft or sqft <= 0:
        st.error("Please enter the Unit Size (sqft).")
    else:
        price = predict_price(
            style_models=style_models, style_medians=style_medians,
            city=city or "other", seller=seller or "other", style=style,
            zipcode=zipcode or "other", sqft=float(sqft),
            beds=float(beds) if beds else None,
            baths=float(baths) if baths else None,
            lot_size=float(lot_size) if lot_size else None,
            year=int(year) if year else None,
            month=int(month) if month else None,
        )
        rmse = style_rmses.get(style)
        meta = ""
        if rmse:
            meta = (f'<div class="price-meta">Typical model error for this style is '
                    f'about <b>±${rmse:,.0f}</b>. Treat the figure as a midpoint, '
                    f'not a precise value.</div>')
        st.markdown(
            f"""
            <div class="price-card">
              <div class="price-tag">Estimated Sale Price</div>
              <div class="price-value">${price:,.0f}</div>
              <div class="price-sub">{style} in {city or '—'}</div>
              {meta}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("This is a statistical estimate from a machine learning model — not an "
           "appraisal or offer. Accuracy depends on how closely the inputs match the "
           "training data.")
