import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

CSV = "Banglore_traffic_Enriched.csv"
REQUIRED = {"Date", "Area Name", "Road/Intersection Name", "Traffic Volume", "Average Speed",
            "Congestion Level", "Incident Reports", "Weather Conditions",
            "Roadwork and Construction Activity", "Road Latitude", "Road Longitude",
            "Year", "Day Name", "Is Weekend", "Year-Month", "Environmental Impact"}

if not os.path.exists(CSV):
    np.random.seed(42)
    n = 1500
    df_sample = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="H"),
        "Area Name": np.random.choice(["Indiranagar", "Whitefield", "Koramangala", "M.G. Road", "Jayanagar", "Hebbal", "Yeshwanthpur", "Electronic City"], n),
        "Road/Intersection Name": [f"Intersection {i%60}" for i in range(n)],
        "Traffic Volume": np.random.randint(5000, 50000, n),
        "Average Speed": np.random.uniform(15, 60, n),
        "Congestion Level": np.random.uniform(20, 100, n),
        "Incident Reports": np.random.randint(0, 5, n),
        "Weather Conditions": np.random.choice(["Clear", "Rainy", "Foggy", "Cloudy"], n),
        "Roadwork and Construction Activity": np.random.choice(["None", "Minor", "Major"], n),
        "Road Latitude": np.random.uniform(12.85, 13.05, n),
        "Road Longitude": np.random.uniform(77.52, 77.73, n),
        "Year": np.random.choice([2023, 2024], n),
        "Day Name": np.random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
        "Is Weekend": np.random.choice([False, True], n, p=[0.7, 0.3]),
        "Year-Month": np.random.choice(["2023-01", "2023-02", "2024-01", "2024-02"], n),
        "Environmental Impact": np.random.uniform(50, 150, n)
    })
    df_sample.to_csv(CSV, index=False)

_ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/%s/MapServer/tile/{z}/{y}/{x}"
TILES = {
    "OpenStreetMap": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", "&copy; OpenStreetMap contributors"),
    "Light gray": (_ESRI % "World_Light_Gray_Base", "Tiles &copy; Esri"),
    "Dark gray": (_ESRI % "World_Dark_Gray_Base", "Tiles &copy; Esri"),
}
GREEN, AMBER, RED = "#2f6f5e", "#d9a521", "#d9534f"

st.set_page_config(page_title="Bengaluru Traffic", page_icon="🚦", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1500px;}
  [data-testid="stMetric"] {background:#fff; border:1px solid #e7e5df; border-radius:12px; padding:14px 18px;}
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {color:#1f2937 !important;}
  [data-testid="stMetricLabel"] p {font-size:.78rem; letter-spacing:.04em; text-transform:uppercase; color:#6b7280;}
  .card-title {font-weight:700; font-size:1rem; margin:0;}
  .card-sub {color:#8a8a85; font-size:.8rem; margin:0 0 .6rem 0;}
  footer, #MainMenu {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🚦 Bengaluru Traffic")
        st.caption("Sign in to open the dashboard")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", width="stretch", type="primary"):
                if u == "admin" and p == "blr123":
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
        st.caption("Credentials: username **admin** · password **blr123**")
    st.stop()

@st.cache_data(show_spinner="Loading data…")
def load():
    df = pd.read_csv(CSV)
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace("/", "_").str.replace(" ", "_").str.replace("-", "_"))
    df["date"] = pd.to_datetime(df["date"])
    return df

raw = load()

st.sidebar.markdown("### 🚦 BENGALURU / TRAFFIC")
page = st.sidebar.radio(
    "Menu",
    ["📊 Overview", "🗺️ Geospatial Map", "📈 Trends", "⛅ Weather & Roadworks", "🔎 Data Explorer"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
areas = sorted(raw["area_name"].unique())
sel_areas = st.sidebar.multiselect("Area", areas, default=areas)
years = sorted(raw["year"].unique())
sel_years = st.sidebar.multiselect("Year", years, default=years)
st.sidebar.markdown("---")
st.sidebar.caption("Logged in as **admin**")
if st.sidebar.button("Logout", width="stretch"):
    st.session_state.auth = False
    st.rerun()

df = raw[raw["area_name"].isin(sel_areas) & raw["year"].isin(sel_years)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

@st.cache_data
def road_stats(d):
    g = (d.groupby(["area_name", "road_intersection_name", "road_latitude", "road_longitude"])
           .agg(congestion=("congestion_level", "mean"), speed=("average_speed", "mean"),
                volume=("traffic_volume", "mean"), incidents=("incident_reports", "sum"))
           .reset_index())
    return g

def colour(v, lo, hi):
    return RED if v >= hi else AMBER if v >= lo else GREEN

@st.cache_resource(show_spinner=False)
def build_map(stats_key, tiles, heat, size_by_volume, height_marker=True):
    stats = stats_key
    m = folium.Map(location=[stats.road_latitude.mean(), stats.road_longitude.mean()],
                   zoom_start=12, tiles=None, control_scale=True)
    url, attr = TILES[tiles]
    folium.TileLayer(tiles=url, attr=attr, name=tiles).add_to(m)
    lo, hi = stats.congestion.quantile([.33, .66])
    if heat:
        HeatMap(stats[["road_latitude", "road_longitude", "congestion"]].values.tolist(),
                radius=35, blur=28).add_to(m)
    for r in stats.itertuples():
        rad = 6 + (r.volume / stats.volume.max()) * 14 if size_by_volume else 9
        c = colour(r.congestion, lo, hi)
        folium.CircleMarker(
            [r.road_latitude, r.road_longitude], radius=rad, color=c, weight=2,
            fill=True, fill_color=c, fill_opacity=.75,
            tooltip=folium.Tooltip(
                f"<b>{r.road_intersection_name}</b><br>{r.area_name}<br>"
                f"Congestion: {r.congestion:.1f}%<br>Speed: {r.speed:.1f} km/h<br>"
                f"Avg volume: {r.volume:,.0f}<br>Incidents: {r.incidents:,}"),
        ).add_to(m)
    return m

LEGEND = (f"<span style='color:{RED}'>●</span> Higher congestion &nbsp; "
          f"<span style='color:{AMBER}'>●</span> Moderate &nbsp; "
          f"<span style='color:{GREEN}'>●</span> Lower")

stats = road_stats(df)

if page == "📊 Overview":
    st.title("Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg congestion", f"{df.congestion_level.mean():.1f}%")
    k2.metric("Avg speed", f"{df.average_speed.mean():.1f} km/h")
    k3.metric("Total traffic volume", f"{df.traffic_volume.sum()/1e6:.1f}M")
    k4.metric("Incident reports", f"{df.incident_reports.sum():,}")

    st.markdown("<p class='card-title'>Geospatial snapshot</p>"
                f"<p class='card-sub'>{len(stats)} road locations plotted from the traffic CSV</p>", unsafe_allow_html=True)
    m = build_map(stats, "OpenStreetMap", False, True)
    st_folium(m, height=430, use_container_width=True, returned_objects=[], key="snap")
    st.markdown(LEGEND, unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<p class='card-title'>Average speed by month</p>", unsafe_allow_html=True)
        ms = df.groupby("year_month", as_index=False)["average_speed"].mean()
        fig = px.area(ms, x="year_month", y="average_speed", color_discrete_sequence=[GREEN])
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="km/h")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("<p class='card-title'>Area comparison</p>", unsafe_allow_html=True)
        ac = (df.groupby("area_name")
                .agg(mean_congestion=("congestion_level", "mean"),
                     avg_speed=("average_speed", "mean"),
                     incidents=("incident_reports", "sum"))
                .sort_values("mean_congestion", ascending=False).round(1))
        st.dataframe(ac, width="stretch", height=330)

elif page == "🗺️ Geospatial Map":
    st.title("Geospatial Map")
    a, b, c, d = st.columns(4)
    metric = a.selectbox("Colour by", ["Congestion", "Average speed"])
    tiles = b.selectbox("Base map", ["OpenStreetMap", "Light gray", "Dark gray"])
    heat = c.toggle("Heatmap layer", value=False)
    size = d.toggle("Size by traffic volume", value=True)

    s = stats.copy()
    if metric == "Average speed":
        s["congestion"] = 100 - s["speed"]
    m = build_map(s, tiles, heat, size)
    st_folium(m, height=640, use_content_width=True, returned_objects=[], key="full")
    st.markdown(LEGEND, unsafe_allow_html=True)

elif page == "📈 Trends":
    st.title("Trends")
    metric = st.selectbox("Metric", ["congestion_level", "average_speed", "traffic_volume",
                                     "incident_reports", "environmental_impact"])
    mm = df.groupby(["year_month", "area_name"], as_index=False)[metric].mean()
    st.plotly_chart(px.line(mm, x="year_month", y=metric, color="area_name").update_layout(height=400, xaxis_title=None), width="stretch")
    c1, c2 = st.columns(2)
    dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dd = df.groupby("day_name", as_index=False)[metric].mean()
    dd["day_name"] = pd.Categorical(dd["day_name"], dn, ordered=True)
    c1.plotly_chart(px.bar(dd.sort_values("day_name"), x="day_name", y=metric, title="By day of week", color_discrete_sequence=[GREEN]), width="stretch")
    wk = df.groupby("is_weekend", as_index=False)[metric].mean()
    c2.plotly_chart(px.bar(wk, x="is_weekend", y=metric, title="Weekday vs weekend", color_discrete_sequence=[AMBER]), width="stretch")

elif page == "⛅ Weather & Roadworks":
    st.title("Weather & Roadworks")
    c1, c2 = st.columns(2)
    w = df.groupby("weather_conditions", as_index=False).agg(congestion=("congestion_level", "mean"), speed=("average_speed", "mean"))
    c1.plotly_chart(px.bar(w, x="weather_conditions", y="congestion", title="Mean congestion by weather", color_discrete_sequence=[RED]), width="stretch")
    r = df.groupby("roadwork_and_construction_activity", as_index=False).agg(congestion=("congestion_level", "mean"), speed=("average_speed", "mean"))
    c2.plotly_chart(px.bar(r, x="roadwork_and_construction_activity", y="speed", title="Avg speed: roadworks yes/no", color_discrete_sequence=[GREEN]), width="stretch")

else:
    st.title("Data Explorer")
    num = df.select_dtypes("number").columns.tolist()
    a, b = st.columns(2)
    x = a.selectbox("X", num, index=num.index("traffic_volume"))
    y = b.selectbox("Y", num, index=num.index("congestion_level"))
    samp = df.sample(min(len(df), 3000), random_state=1)
    st.plotly_chart(px.scatter(samp, x=x, y=y, color="area_name", opacity=.6).update_layout(height=450), width="stretch")
    st.dataframe(df.head(500), width="stretch")
    st.download_button("Download filtered CSV", df.to_csv(index=False).encode(), "traffic_filtered.csv", "text/csv")
