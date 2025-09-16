import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors
import io
import requests

st.set_page_config(page_title="Reactor Applications – Phase Space", layout="wide")

# --------- Fixed Google Sheet ---------
SHEET_ID = "1KeB-INjb93b77xqG4CiHU5tqDdoa60GdI5ZHve6tsbk"
SHEET_NAME = "AllData"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

st.title("Reactor Applications in Phase Space")
st.caption("Flux and Energy axes are shown in log₁₀ with 10^x tick labels; Temperature in °C.")
st.markdown(f"**Data source:** [Google sheets](https://docs.google.com/spreadsheets/d/1KeB-INjb93b77xqG4CiHU5tqDdoa60GdI5ZHve6tsbk/edit?gid=0#gid=0)")

# Manual refresh button
if st.button("🔄 Reload data"):
    st.session_state["_refresh"] = True

@st.cache_data(ttl=0)
def load_csv() -> pd.DataFrame:
    # Read as strings so we can normalize EU formats reliably
    df = pd.read_csv(CSV_URL, dtype=str, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    # Required columns for the phase-space plot
    required = ["name","flux_min","flux_max","energy_min","energy_max","temp_min","temp_max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in sheet: {missing}")

    # Normalize numeric strings like "1,00E+10" / "2,00E−02" → float
    def to_num(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s = s.replace("\u2212", "-").replace("−", "-")  # Unicode minus → hyphen
        s = s.replace("\xa0", "")                       # NBSP
        s = s.replace(",", ".")                         # decimal comma → dot
        try:
            return float(s)
        except ValueError:
            return np.nan

    num_cols = ["flux_min","flux_max","energy_min","energy_max","temp_min","temp_max"]
    for c in num_cols:
        df[c] = df[c].apply(to_num)

    # Keep only valid, fully-parsed rows
    df = df.dropna(subset=["name"] + num_cols)
    df["name"] = df["name"].astype(str)
    return df

if st.session_state.get("_refresh", False):
    st.cache_data.clear()
    st.session_state["_refresh"] = False

# ---- Load & validate ----
try:
    df = load_csv()
except Exception as e:
    st.error("Could not fetch the Google Sheet via CSV link.")
    st.markdown(
        "- Check the sheet sharing: **Anyone with the link → Viewer**.\n"
        f"- Try opening this CSV URL in your browser: [{CSV_URL}]({CSV_URL})"
    )
    with st.expander("Error details"):
        st.code(repr(e))
    st.stop()

required_cols = ["name", "flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your sheet is missing required columns: {missing}")
    st.stop()

# Coerce numeric columns; drop incomplete rows
for c in ["flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=required_cols)
df["name"] = df["name"].astype(str)

# ---- Sidebar controls (taller default to avoid clipping) ----
with st.sidebar:
    st.header("Plot settings")
    height = st.slider("Plot height (px)", 700, 1600, 1050, 50)
    x_log_ticks = st.multiselect("Flux tick exponents", [6, 8, 10, 12, 14, 16, 18],
                                 default=[6, 8, 10, 12, 14, 16, 18])
    y_log_ticks = st.multiselect("Energy tick exponents", [-3, -1, 0, 2, 4, 6, 8],
                                 default=[-3, -1, 0, 2, 4, 6, 8])
    xlim = st.slider("Flux exponent range", 0, 20, (6, 18))
    ylim = st.slider("Energy exponent range", -5, 10, (-3, 8))
    zlim = st.slider("Temperature range (°C)", 0, 1500, (0, 1300))

    st.divider()
    st.subheader("Filter")
    names = ["(All)"] + sorted(df["name"].unique().tolist())
    name_pick = st.selectbox("Application", names)

# Apply filter
plot_df = df if name_pick == "(All)" else df[df["name"] == name_pick]

# ---- Colors (unique & deterministic per name) ----
palette = (qcolors.Dark24 + qcolors.Light24 + qcolors.Alphabet + qcolors.Set3 + qcolors.D3)
name_list = sorted(df["name"].unique().tolist())
color_map = {n: palette[i % len(palette)] for i, n in enumerate(name_list)}

# ---- Geometry helpers ----
def cuboid_vertices(flux_range, energy_range, temp_range):
    x0, x1 = np.log10(flux_range[0]), np.log10(flux_range[1])
    y0, y1 = np.log10(energy_range[0]), np.log10(energy_range[1])
    z0, z1 = temp_range
    V = np.array([
        [x0, y0, z0],[x1, y0, z0],[x1, y1, z0],[x0, y1, z0],
        [x0, y0, z1],[x1, y0, z1],[x1, y1, z1],[x0, y1, z1],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return V, edges, faces

def add_wireframe(fig, name, color, flux_range, energy_range, temp_range):
    V, E, _ = cuboid_vertices(flux_range, energy_range, temp_range)
    first = True
    for (i, j) in E:
        fig.add_trace(go.Scatter3d(
            x=[V[i,0], V[j,0]],
            y=[V[i,1], V[j,1]],
            z=[V[i,2], V[j,2]],
            mode="lines",
            line=dict(width=6, color=color),
            name=name if first else None,
            showlegend=first,
            hoverinfo="skip"  # we'll show hover on the invisible mesh
        ))
        first = False

def add_hover_mesh(fig, name, color, flux_range, energy_range, temp_range, opacity=0.01):
    V, _, F = cuboid_vertices(flux_range, energy_range, temp_range)
    i, j, k = zip(*F)
    # Ultra-low opacity mesh so hovering anywhere on the box shows its name
    fig.add_trace(go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, showlegend=False,
        hoverinfo="text", text=name
    ))

def add_filled_box(fig, name, color, flux_range, energy_range, temp_range, opacity=0.15):
    V, _, F = cuboid_vertices(flux_range, energy_range, temp_range)
    i, j, k = zip(*F)
    fig.add_trace(go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, showlegend=True,
        hoverinfo="skip"
    ))

# ---- Build plot ----
fig = go.Figure()



# Applications: wireframe + invisible mesh for hover
for _, r in plot_df.iterrows():
    name = str(r["name"])
    color = color_map[name]
    flux_range = (float(r["flux_min"]), float(r["flux_max"]))
    energy_range = (float(r["energy_min"]), float(r["energy_max"]))
    temp_range = (float(r["temp_min"]), float(r["temp_max"]))
    add_wireframe(fig, name, color, flux_range, energy_range, temp_range)
    add_hover_mesh(fig, name, color, flux_range, energy_range, temp_range, opacity=0.01)

# Axis ticks/labels like MPL
def ticktext(vals):
    return [f"10^{int(v) if v==int(v) else v}" for v in vals]

fig.update_layout(
    scene=dict(
        xaxis=dict(
            title="Neutron flux [1/(cm²·s)]",
            tickvals=x_log_ticks,
            ticktext=ticktext(x_log_ticks),
            range=[xlim[0], xlim[1]],
        ),
        yaxis=dict(
            title="Neutron energy [eV]",
            tickvals=y_log_ticks,
            ticktext=ticktext(y_log_ticks),
            range=[ylim[0], ylim[1]],
            autorange="reversed",  # invert like your Matplotlib plot
        ),
        zaxis=dict(
            title="Temperature [°C]",
            range=[zlim[0], zlim[1]],
        ),
        aspectmode="cube",
        # Give the 3D scene more vertical room inside the figure
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
    ),
    legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=30, b=0),
    height=height,
)

st.plotly_chart(fig, width='stretch')

with st.expander("Show raw data"):
    st.dataframe(df, width='stretch')








