"""Sydney Housing Price Predictor (SIT720 8.1D).

Run from the project root:
    source .venv/bin/activate
    python -m streamlit run app/app.py

The model and metrics files are created by notebooks/housing_price_prediction.ipynb.
"""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# The saved pipeline uses the custom classes in src/features.py, so the project root must be importable.
sys.path.insert(0, str(ROOT))
from src.features import MODEL_INPUTS, TARGET  # noqa: E402

MODEL_PATH = ROOT / "models" / "gradient_boosting_model.joblib"
DATA_PATH = ROOT / "data" / "sydney_housing_final.csv"
METRICS_PATH = ROOT / "report" / "final_metrics.json"

ACCESS_COLS = ["distance_to_nearest_train_station_km", "distance_to_nearest_bus_route_m", "regular_bus_routes_within_500m"]
MESSAGE = "Model estimate — not a professional property valuation."

NAVY = "#1d3557"
TEAL = "#2a7f8e"
AMBER = "#c8841a"
TYPE_COLOURS = {"House": NAVY, "Townhouse": "#7aa6b8", "Unit": TEAL}
PRICE_TICKS = [400_000, 700_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000]
PRICE_RANGE = [330_000, 26_000_000]
# Vega-Lite expression that shows axis labels as $650k or $1.2M
AUD_AXIS = "datum.value >= 1e6 ? '$' + format(datum.value / 1e6, '.1f') + 'M' : '$' + format(datum.value / 1e3, '.0f') + 'k'"

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  .result-card {border-radius: 12px; padding: 1.2rem 1.4rem; background: #f7fafc; border: 1px solid #dbe3ea;
                border-left: 5px solid #c8841a; margin-bottom: 1rem;}
  .result-card .label {color: #64748b; font-size: .9rem;}
  .result-card .value {color: #1d3557; font-size: 2.4rem; font-weight: 700; line-height: 1.2;}
  .result-card .sub {color: #64748b; font-size: .88rem;}
  div[data-testid="stMetric"] {border: 1px solid #e3e9ef; border-radius: 10px; padding: .6rem .8rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sales():
    return pd.read_csv(DATA_PATH)[["suburb", "property_type", TARGET]]


def aud(x):
    return f"${x:,.0f}"


def aud_short(x):
    if x >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x / 1e3:.0f}k"


def prepare_inputs(table):
    # Keep exactly the columns the model was trained on. Unknown values stay NaN,
    # so the pipeline's own imputer fills them in.
    table = pd.DataFrame(table, columns=MODEL_INPUTS)
    for col in MODEL_INPUTS:
        if col not in ("suburb", "property_type"):
            table[col] = table[col].astype(float)
    return table


if not MODEL_PATH.exists():
    st.error("The trained model is not available yet. Please run the project notebook first.")
    st.stop()

bundle = load_model()
model = bundle["model"]
defaults = bundle["suburb_access_defaults"]
ranges = bundle["ranges"]
sales = load_sales()
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None

st.title("Sydney Housing Price Predictor")
st.markdown("Machine-learning decision support for Blacktown, Parramatta and Mosman")
st.caption(f"**Model:** {bundle['model_name']}  ·  **Training data:** {bundle['trained_on_rows']} sold properties")

left, right = st.columns([0.4, 0.6], gap="large")

# ---------------------------------------------------------------- inputs
with left:
    with st.container(border=True):
        st.subheader("Property details")
        c1, c2 = st.columns(2)
        suburb = c1.selectbox("Suburb", ["Blacktown", "Parramatta", "Mosman"], key="suburb")
        ptype = c2.selectbox("Property type", ["House", "Unit", "Townhouse"], key="ptype")
        is_unit = ptype == "Unit"

        with st.form("property_form", border=False):
            # Input limits come from the training data so users stay inside the range the model has seen.
            c1, c2, c3 = st.columns(3)
            bedrooms = c1.number_input("Bedrooms", int(ranges["bedrooms"][0]), int(ranges["bedrooms"][1]), 2 if is_unit else 3)
            bathrooms = c2.number_input("Bathrooms", int(ranges["bathrooms"][0]), int(ranges["bathrooms"][1]), 1 if is_unit else 2)
            car = c3.number_input("Car spaces", int(ranges["car_spaces"][0]), int(ranges["car_spaces"][1]), 1)
            car_unknown = st.checkbox("Car spaces unknown")

            if is_unit:
                st.caption("Land size is not used for units (strata units have no individual land parcel).")
                land = np.nan
                land_unknown = True
            else:
                land_min, land_max = ranges["land_size_sqm"]
                land = st.number_input("Land size (m²)", float(np.floor(land_min)), float(np.ceil(land_max)),
                                       550.0 if ptype == "House" else 200.0, step=10.0,
                                       help=f"Training range {land_min:,.0f}–{land_max:,.0f} m².")
                land_unknown = st.checkbox("Land size unknown")

            building_min, building_max = ranges["building_size_sqm"]
            building = st.number_input("Building / floor area (m²)", float(np.floor(building_min)), float(np.ceil(building_max)),
                                       80.0 if is_unit else 120.0, step=5.0,
                                       help=f"Training range {building_min:,.0f}–{building_max:,.0f} m².")
            building_unknown = st.checkbox("Building / floor area unknown", value=True)

            suburb_defaults = defaults[suburb]
            use_defaults = st.checkbox(f"Use typical {suburb} accessibility values", value=True,
                                       help="Medians of the sampled properties in this suburb, not values for a specific address.")
            with st.expander("Accessibility values (used when the box above is unticked)"):
                station = st.number_input("Distance to nearest train station (km)", 0.0, 10.0,
                                          float(suburb_defaults["distance_to_nearest_train_station_km"]), step=0.05, key=f"st_{suburb}")
                bus = st.number_input("Distance to nearest bus route (m)", 0.0, 1000.0,
                                      float(suburb_defaults["distance_to_nearest_bus_route_m"]), step=5.0, key=f"bus_{suburb}")
                routes = st.number_input("Regular bus routes within 500 m", 0, 60,
                                         int(suburb_defaults["regular_bus_routes_within_500m"]), key=f"rt_{suburb}")

            submitted = st.form_submit_button("Estimate Property Value", type="primary", width="stretch")

    if submitted:
        values = {
            "suburb": suburb,
            "property_type": ptype,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "car_spaces": np.nan if car_unknown else car,
            "land_size_sqm": np.nan if land_unknown else land,
            "building_size_sqm": np.nan if building_unknown else building,
        }
        if use_defaults:
            for col in ACCESS_COLS:
                values[col] = float(suburb_defaults[col])
        else:
            values["distance_to_nearest_train_station_km"] = station
            values["distance_to_nearest_bus_route_m"] = bus
            values["regular_bus_routes_within_500m"] = routes

        prediction = float(model.predict(prepare_inputs([values]))[0])
        st.session_state["result"] = {"values": values, "pred": prediction}

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
with right:
    if result is None:
        st.info("Enter the property details and select **Estimate Property Value**.")
        show_suburb, show_type, pred = suburb, ptype, None
    else:
        values = result["values"]
        pred = result["pred"]
        show_suburb, show_type = values["suburb"], values["property_type"]
        st.markdown(f"""
        <div class="result-card">
          <div class="label">Estimated sale price · {show_type} in {show_suburb}</div>
          <div class="value">{aud(round(pred, -3))}</div>
          <div class="sub">{MESSAGE}</div>
        </div>""", unsafe_allow_html=True)

        notes = []
        similar_sales = bundle["segment_counts"].get((show_suburb, show_type), 0)
        if similar_sales < 5:
            notes.append(f"Only {similar_sales} similar sale(s) were in the training data.")
        if show_suburb == "Mosman" and show_type == "House":
            notes.append("Mosman houses were the hardest segment to predict (large errors in testing).")
        size_missing = pd.isna(values["building_size_sqm"]) or (show_type != "Unit" and pd.isna(values["land_size_sqm"]))
        if size_missing:
            notes.append("Some size information is unknown, so typical values for this property type were used.")
        if notes:
            st.warning(" ".join(notes))

    suburb_sales = sales[sales["suburb"] == show_suburb]
    type_sales = suburb_sales[suburb_sales["property_type"] == show_type]
    suburb_median = suburb_sales[TARGET].median()

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{show_suburb} median", aud_short(suburb_median))
    m2.metric(f"{show_type} median", aud_short(type_sales[TARGET].median()) if len(type_sales) > 0 else "–",
              help=f"Based on {len(type_sales)} sale(s) in the training data.")
    m3.metric("Estimate vs suburb median", "–" if pred is None else f"{pred / suburb_median - 1:+.0%}")

    tab1, tab2, tab3 = st.tabs([f"{show_suburb} price distribution", "Median by suburb and type", f"{show_suburb} price range by type"])

    with tab1:
        # Log-spaced bins because prices range from about $400k to $23M.
        edges = np.logspace(np.log10(PRICE_RANGE[0]), np.log10(PRICE_RANGE[1]), 25)
        counts, _ = np.histogram(suburb_sales[TARGET], bins=edges)
        hist = pd.DataFrame({"low": edges[:-1], "high": edges[1:], "count": counts})
        chart = alt.Chart(hist).mark_rect(color=TEAL, opacity=0.85, stroke="white").encode(
            x=alt.X("low:Q", scale=alt.Scale(type="log", domain=PRICE_RANGE), title="Sale price (AUD, log scale)",
                    axis=alt.Axis(labelExpr=AUD_AXIS, values=PRICE_TICKS, grid=False)),
            x2="high:Q",
            y=alt.Y("count:Q", title="Number of sales", axis=alt.Axis(tickMinStep=1)),
            y2=alt.datum(0))
        if pred is not None:
            chart = chart + alt.Chart(pd.DataFrame({"p": [pred]})).mark_rule(color=AMBER, strokeWidth=3).encode(x="p:Q")
        st.altair_chart(chart.properties(height=280, title=f"Sale prices of the {len(suburb_sales)} {show_suburb} properties in the training data"),
                        width="stretch")
        if pred is not None:
            st.caption("The amber line marks this estimate.")

    with tab2:
        medians = sales.groupby(["suburb", "property_type"])[TARGET].median().reset_index(name="median")
        chart = alt.Chart(medians).mark_bar().encode(
            x=alt.X("suburb:N", title="Suburb", sort=["Blacktown", "Parramatta", "Mosman"], axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("property_type:N", sort=["House", "Townhouse", "Unit"]),
            y=alt.Y("median:Q", title="Median sale price (AUD)", axis=alt.Axis(labelExpr=AUD_AXIS)),
            color=alt.Color("property_type:N", title="Property type",
                            scale=alt.Scale(domain=list(TYPE_COLOURS), range=list(TYPE_COLOURS.values()))),
            tooltip=["suburb", "property_type", alt.Tooltip("median:Q", format="$,.0f")])
        st.altair_chart(chart.properties(height=300, title="Median sale price by suburb and property type"), width="stretch")
        st.caption("Townhouse medians are based on only 1–2 sales per suburb.")

    with tab3:
        bands = suburb_sales.groupby("property_type")[TARGET].describe().reset_index()
        bands = bands.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
        bands = bands[bands["count"] >= 3]  # too few sales for a meaningful range

        low = suburb_sales[TARGET].min() * 0.85
        high = suburb_sales[TARGET].max() * 1.15
        if pred is not None:
            low = min(low, pred * 0.85)
            high = max(high, pred * 1.15)
        ticks = [t for t in [300_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000] if low <= t <= high]

        y = alt.Y("property_type:N", title="Property type", sort=["House", "Townhouse", "Unit"])
        x = alt.X("q1:Q", scale=alt.Scale(type="log", domain=[low, high]), title="Sale price (AUD, log scale)",
                  axis=alt.Axis(labelExpr=AUD_AXIS, values=ticks))
        bars = alt.Chart(bands).mark_bar(height=28, color="#b8cfdc").encode(x=x, x2="q3:Q", y=y)
        median_ticks = alt.Chart(bands).mark_tick(color=NAVY, thickness=3, size=36).encode(x="median:Q", y=y)
        chart = bars + median_ticks
        if pred is not None and show_type in bands["property_type"].values:
            point = pd.DataFrame({"property_type": [show_type], "p": [pred]})
            chart = chart + alt.Chart(point).mark_point(shape="diamond", size=200, filled=True, color=AMBER).encode(x="p:Q", y=y)
        st.altair_chart(chart.properties(height=alt.Step(64), title=f"{show_suburb}: middle 50% of sale prices by property type"),
                        width="stretch")
        caption = "Bars show the 25th–75th percentile and the navy tick is the median. Types with fewer than 3 sales are not shown."
        if pred is not None:
            caption += " The amber diamond is this estimate."
        st.caption(caption)

# ---------------------------------------------------------------- performance and limitations
st.divider()
c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Model performance")
    if metrics is None:
        st.write("Performance figures are not available yet. Please run the project notebook first.")
    else:
        cv = metrics["cv_primary_split"]
        repeated = metrics["cv_repeated_splits"]
        holdout = metrics["final_holdout"]
        table = pd.DataFrame(
            {"MAE": [cv["CV MAE"], repeated["mean CV MAE (6 splits)"], holdout["MAE"]],
             "RMSE": [cv["CV RMSE"], repeated["mean CV RMSE"], holdout["RMSE"]],
             "R²": [cv["CV R²"], repeated["mean CV R²"], holdout["R2"]],
             "MdAPE": [cv["CV MdAPE %"], repeated["mean CV MdAPE %"], holdout["MdAPE"]]},
            index=["5-fold CV (95 properties)", "Mean of 6 repeated CV splits", "Final holdout (10 properties)"])
        st.dataframe(table.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R²": "{:.3f}", "MdAPE": "{:.1f}%"}), width="stretch")
        st.caption(f"Gradient Boosting was selected for the final prototype using the repeated-cross-validation mean-MAE criterion. "
                   f"However, model rankings varied across metrics, and the model showed substantial overfitting risk "
                   f"(training R² {cv['Train R²']:.3f} vs CV R² {cv['CV R²']:.3f}). The holdout has only 10 properties and no Mosman houses.")
with c2:
    st.subheader("Limitations")
    st.markdown("""
- Trained on only **105 sales from three suburbs**; other suburbs are not supported.
- **Luxury Mosman houses** are very hard to predict; errors of several million dollars occurred.
- **Building area is missing** for most houses, and views, condition and renovations are not recorded.
- Sales are mostly from 2026, so later market changes are not reflected.
- 30 of the 105 sales could not be matched on the individual listing page.
- This is an **educational prototype, not a professional valuation**.
""")

with st.expander("Estimate several properties from a CSV file"):
    st.write("Upload a CSV with these columns (car spaces, land and building area may be blank):")
    st.code(", ".join(MODEL_INPUTS))
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        uploaded = pd.read_csv(upload)
        missing_cols = [c for c in MODEL_INPUTS if c not in uploaded.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        elif not uploaded["suburb"].isin(defaults).all():
            st.error(f"Only these suburbs are supported: {sorted(defaults)}")
        elif uploaded[ACCESS_COLS].isna().any().any():
            st.error("The accessibility columns cannot be blank.")
        else:
            rows = prepare_inputs(uploaded)
            uploaded["estimated_price"] = model.predict(rows).round(-3)
            st.dataframe(uploaded, width="stretch")
            st.download_button("Download estimates", uploaded.to_csv(index=False), "estimates.csv", "text/csv")
