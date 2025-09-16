import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors

st.set_page_config(page_title="Reactor Applications – Phase Space", layout="wide")

# --------- Fixed Google Sheet (CSV export) ----------
SHEET_ID = "1KeB-INjb93b77xqG4CiHU5tqDdoa60GdI5ZHve6tsbk"
GID = "0"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?gid={GID}#gid={GID}"

st.title("Reactor Applications in Phase Space")
st.caption("Flux and Energy axes are shown in log₁₀ with 10^x tick labels; Temperature in °C.")
st.markdown(f"**Data source:** [Google Sheet]({SHEET_URL})")

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

# Reload button to clear cache if sheet changes
if st.button("Reload data from Google Sheet"):
    load_data.clear()

df = load_data(CSV_URL)

required_cols = ["name", "flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your sheet is missing required columns: {missing}")
    st.stop()

# Ensure numeric
for c in ["flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["name", "flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"])

# --------- Controls ----------
with st.sidebar:
    st.header("Plot settings")
    # Taller plot to avoid getting cut off + user control
    height = st.slider("Plot height (px)", 600, 1400, 950, 50)

    # Axis ticks and limits (kept from your MPL defaults)
    x_log_ticks = st.multiselect("Flux tick exponents", [6, 8, 10, 12, 14, 16, 18], default=[6, 8, 10, 12, 14, 16, 18])
    y_log_ticks = st.multiselect("Energy tick exponents", [-3, -1, 0, 2, 4, 6, 8], default=[-3, -1, 0, 2, 4, 6, 8])
    xlim = st.slider("Flux exponent range", 0, 20, (6, 18))
    ylim = st.slider("Energy exponent range", -5, 10, (-3, 8))
    zlim = st.slider("Temperature range (°C)", 0, 1500, (0, 1300))

    st.divider()
    st.subheader("Filter")
    names = ["(All)"] + sorted(df["name"].astype(str).unique().tolist())
    name_pick = st.selectbox("Application", names)

# Apply filter
plot_df = df if name_pick == "(All)" else df[df["name"] == name_pick]

# --------- Color assignment (unique, deterministic) ----------
# Large qualitative palette to cover many items
palette = (
    qcolors.Dark24
    + qcolors.Light24
    + qcolors.Alphabet
    + qcolors.Set3
    + qcolors.D3
)
name_list = sorted(df["name"].astype(str).unique().tolist())
color_map = {n: palette[i % len(palette)] for i, n in enumerate(name_list)}

# --------- Geometry helpers ----------
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
            hoverinfo="skip"  # hover handled by invisible mesh (next function)
        ))
        first = False

def add_hover_mesh(fig, name, color, flux_range, energy_range, temp_range, opacity=0.01):
    V, _, F = cuboid_vertices(flux_range, energy_range, temp_range)
    i, j, k = zip(*F)
    # Invisible (nearly) volume so hovering anywhere shows the application name
    fig.add_trace(go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=False,
        hoverinfo="text",
        text=name
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

# --------- Build plot ----------
fig = go.Figure()

# Reactor capability overlay (same as before)
reactor_flux_range = (1e6, 5e14)
reactor_energy_range = (0.025, 1e7)
reactor_temp_range = (20, 1300)
add_filled_box(
    fig, "Reactor capability (5–10 MW)", "gray",
    reactor_flux_range, reactor_energy_range, reactor_temp_range, opacity=0.15
)

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
        xaxis=dict(title="Neutron flux [1/(cm²·s)]", tickvals=x_log_ticks, ticktext=ticktext(x_log_ticks), range=[xlim[0], xlim[1]]),
        yaxis=dict(title="Neutron energy [eV]", tickvals=y_log_ticks, ticktext=ticktext(y_log_ticks), range=[ylim[0], ylim[1]], autorange="reversed"),
        zaxis=dict(title="Temperature [°C]", range=[zlim[0], zlim[1]]),
        aspectmode="cube",
    ),
    legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=30, b=0),
    height=height,
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Show raw data"):
    st.dataframe(df, use_container_width=True)
