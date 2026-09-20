from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

st.set_page_config(page_title="Montréal BIXI Demand Lab", page_icon="🚲", layout="wide")
st.markdown(
    """
<style>
.block-container{padding-top:2rem;max-width:1250px}.hero{padding:1.4rem 1.6rem;border-radius:18px;
background:linear-gradient(120deg,#073b4c,#118ab2);color:white;margin-bottom:1rem}
[data-testid="stMetric"]{background:#f6f8fa;padding:14px;border-radius:12px}
</style>
<div class="hero"><h1>Montréal BIXI Demand Lab</h1>
<p>When will Montréal ride? Exploring how time and weather shape hourly bike demand.</p></div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    hourly = pd.read_parquet(DATA / "hourly_demand.parquet")
    predictions = pd.read_parquet(DATA / "predictions.parquet") if (DATA / "predictions.parquet").exists() else None
    features = pd.read_parquet(DATA / "hourly_features.parquet")
    return hourly, predictions, features


if not (DATA / "hourly_features.parquet").exists():
    st.error("No processed data found. Run `make demo` or the real-data pipeline first.")
    st.stop()

hourly, predictions, features = load_data()
hourly["datetime"] = pd.to_datetime(hourly["datetime"])
features["datetime"] = pd.to_datetime(features["datetime"])

with st.sidebar:
    st.header("Filters")
    start, end = hourly.datetime.min().date(), hourly.datetime.max().date()
    selected = st.date_input("Date range", (start, end), min_value=start, max_value=end)
    days = st.multiselect(
        "Days",
        list(range(7)),
        default=list(range(7)),
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
    )

if isinstance(selected, tuple) and len(selected) == 2:
    mask = hourly.datetime.dt.date.between(selected[0], selected[1])
else:
    mask = pd.Series(True, index=hourly.index)
filtered = hourly[mask & hourly.datetime.dt.dayofweek.isin(days)].copy()

total = int(filtered.demand.sum())
peak = filtered.loc[filtered.demand.idxmax()] if len(filtered) else None
c1, c2, c3 = st.columns(3)
c1.metric("Trips", f"{total:,}")
c2.metric("Average per hour", f"{filtered.demand.mean():,.0f}" if len(filtered) else "—")
c3.metric(
    "Peak hour",
    peak.datetime.strftime("%a %H:%M") if peak is not None else "—",
    f"{int(peak.demand):,} trips" if peak is not None else None,
)

tabs = st.tabs(["Demand overview", "Stations", "Weather", "Forecast performance", "Methods"])
with tabs[0]:
    st.subheader("Hourly demand")
    st.plotly_chart(px.line(filtered, x="datetime", y="demand", labels={"demand": "Trips"}), use_container_width=True)
    left, right = st.columns(2)
    by_hour = features.groupby("hour", as_index=False).demand.mean()
    left.plotly_chart(
        px.area(by_hour, x="hour", y="demand", labels={"demand": "Average trips"}), use_container_width=True
    )
    heat = features.pivot_table(index="day_of_week", columns="hour", values="demand", aggfunc="mean")
    right.plotly_chart(
        px.imshow(
            heat, aspect="auto", color_continuous_scale="Teal", labels={"x": "Hour", "y": "Day", "color": "Trips"}
        ),
        use_container_width=True,
    )

with tabs[1]:
    st.subheader("Station activity")
    station_path = DATA / "stations.parquet"
    station_hourly_path = DATA / "station_hourly.parquet"
    if station_path.exists() and station_hourly_path.exists():
        stations = pd.read_parquet(station_path)
        station_hourly = pd.read_parquet(station_hourly_path)
        totals = station_hourly.groupby("station_id", as_index=False).departures.sum()
        station_map = stations.merge(totals, on="station_id", how="inner")
        station_map["marker_size"] = station_map.departures.clip(lower=1)
        fig = px.scatter_map(
            station_map,
            lat="latitude",
            lon="longitude",
            size="marker_size",
            color="departures",
            hover_name="name",
            hover_data={"borough": True, "departures": ":,", "marker_size": False},
            color_continuous_scale="Tealgrn",
            size_max=22,
            zoom=10,
            map_style="carto-positron",
        )
        st.plotly_chart(fig, use_container_width=True)
        top_stations = station_map.nlargest(15, "departures")[["name", "borough", "departures"]]
        st.dataframe(top_stations, hide_index=True, use_container_width=True)
    else:
        st.info("Station-level files are created by the official-data pipeline.")

with tabs[2]:
    st.subheader("Weather sensitivity")
    col1, col2 = st.columns(2)
    sample = features.sample(min(3000, len(features)), random_state=42)
    col1.plotly_chart(
        px.scatter(
            sample, x="temp_c", y="demand", color="precip_mm", opacity=0.45, trendline="lowess" if False else None
        ),
        use_container_width=True,
    )
    bins = pd.cut(
        features.precip_mm, bins=[-0.1, 0, 1, 3, 10, float("inf")], labels=["None", "≤1", "1–3", "3–10", ">10"]
    )
    rain = features.assign(rain_bin=bins).groupby("rain_bin", observed=True, as_index=False).demand.mean()
    col2.plotly_chart(
        px.bar(rain, x="rain_bin", y="demand", labels={"rain_bin": "Rain (mm)", "demand": "Average trips"}),
        use_container_width=True,
    )

with tabs[3]:
    if predictions is None:
        st.info("Run `make train` to create out-of-sample predictions.")
    else:
        predictions["datetime"] = pd.to_datetime(predictions["datetime"])
        metrics = json.loads((MODELS / "metrics.json").read_text())
        st.caption(f"Winning model: {metrics['winner'].replace('_', ' ').title()} · chronological 80/20 split")
        chart = predictions.tail(24 * 14)
        fig = go.Figure()
        fig.add_scatter(x=chart.datetime, y=chart.demand, name="Actual", line={"color": "#073b4c"})
        fig.add_scatter(x=chart.datetime, y=chart.prediction, name="Predicted", line={"color": "#ef476f"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            pd.DataFrame(metrics["models"]).T.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "WAPE": "{:.1%}"}),
            use_container_width=True,
        )

with tabs[4]:
    st.markdown("""
**Target:** total BIXI departures per hour.  
**Data:** 14,249,363 official 2025 BIXI trip records plus ECCC hourly weather.  
**Predictors:** calendar cycles, Québec holidays, weather, and strictly lagged demand.  
**Validation:** the final 20% of observations are held out chronologically.  
**Baseline:** demand at the same hour one week earlier.  
**Limitations:** completed rides are not unmet demand. Outages, rebalancing, events, and
forecast-weather error can affect results.
""")
