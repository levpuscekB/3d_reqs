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
st.caption("MPRR applications in 3D phase space of Temperature [°C], Neutron Flux [1/cm^2/s] and Neutron energy [eV]. ")
st.markdown(f"**Data source:** [Google sheets](https://docs.google.com/spreadsheets/d/1KeB-INjb93b77xqG4CiHU5tqDdoa60GdI5ZHve6tsbk/edit?gid=0#gid=0)")
st.markdown(f"**Visibility of applications can be changed by clicking on the entries in the legend.**")
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
    height = st.slider("Plot height (px) (change based on your screen size)", 700, 1600, 1050, 50)

    st.divider()


# Apply filter
plot_df = df 

# ---- Colors (unique & deterministic per name) ----
palette = (qcolors.Dark24 + qcolors.Light24 + qcolors.Alphabet + qcolors.Set3 + qcolors.D3)
name_list = sorted(df["name"].unique().tolist())
color_map = {n: palette[i % len(palette)] for i, n in enumerate(name_list)}

# ---- Geometry helpers ----
def cuboid_vertices(flux_range, energy_range, temp_range):
    # USE RAW values (no np.log10 here)
    x0, x1 = flux_range
    y0, y1 = energy_range
    z0, z1 = temp_range
    V = np.array([
        [x0, y0, z0],[x1, y0, z0],[x1, y1, z0],[x0, y1, z0],
        [x0, y0, z1],[x1, y0, z1],[x1, y1, z1],[x0, y1, z1],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),
             (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return V, edges, faces


def add_wireframe(fig, name, color, flux_range, energy_range, temp_range):
    V, E, _ = cuboid_vertices(flux_range, energy_range, temp_range)

    # Merge all edges into ONE trace using None separators
    xs, ys, zs = [], [], []
    for (i, j) in E:
        xs += [V[i,0], V[j,0], None]
        ys += [V[i,1], V[j,1], None]
        zs += [V[i,2], V[j,2], None]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(width=6),
        name=name,
        showlegend=True,
        hoverinfo="skip",
        marker=dict(color=color),
        legendgroup=name,          # <-- group with mesh
    ))

def add_hover_mesh(fig, name, color, flux_range, energy_range, temp_range, opacity=0.01):
    V, _, F = cuboid_vertices(flux_range, energy_range, temp_range)
    i, j, k = zip(*F)
    fig.add_trace(go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name,
        showlegend=False,          # no extra legend entry
        hoverinfo="text", text=name,
        legendgroup=name,          # <-- same group as wireframe
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
            type="log",
            title="Neutron flux [1/(cm²·s)]",
            autorange=True,          # or set numeric range=[1e6, 1e18]
            dtick=1,                 # 1 decade per major tick: 10^n
            showexponent="all",
            exponentformat="power",  # shows 10^n format
        ),
        yaxis=dict(
            type="log",
            title="Neutron energy [eV]",
            autorange="reversed",    # invert like before
            dtick=1,
            showexponent="all",
            exponentformat="power",
        ),
        zaxis=dict(
            title="Temperature [°C]",
            autorange=True,          # or range=[0, 1300]
        ),
        aspectmode="cube",
    ),
    legend=dict(
        x=1.02, y=1,
        bgcolor="rgba(255,255,255,0.7)",
        groupclick="togglegroup"   # <-- clicking the label toggles the whole group
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    height=height,
)

st.plotly_chart(fig, width='stretch')

with st.expander("Show raw data"):
    st.dataframe(df, width='stretch')
















