import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Reactor Applications – Phase Space", layout="wide")

# ----------------------------
# Sidebar: Data source options
# ----------------------------
st.sidebar.header("Data source")
source_mode = st.sidebar.radio(
    "Choose how to read your Google Sheet",
    ["CSV (published to web)", "Service Account (st.secrets)"],
    help="CSV is simplest. Service Account works without publishing the sheet."
)

@st.cache_data(show_spinner=False)
def load_from_csv(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    return df

@st.cache_data(show_spinner=False)
def load_from_service_account(spreadsheet_id: str, worksheet_name: str) -> pd.DataFrame:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    # Expecting st.secrets["gcp_service_account"] to contain a full service account JSON.
    # Example to put in .streamlit/secrets.toml:
    # [gcp_service_account]
    # type = "service_account"
    # project_id = "YOUR_PROJECT_ID"
    # private_key_id = "..."
    # private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    # client_email = "YOUR_SA@YOUR_PROJECT_ID.iam.gserviceaccount.com"
    # client_id = "..."
    # auth_uri = "https://accounts.google.com/o/oauth2/auth"
    # token_uri = "https://oauth2.googleapis.com/token"
    # auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    # client_x509_cert_url = "..."
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Missing [gcp_service_account] in st.secrets")

    scope = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        dict(st.secrets["gcp_service_account"]), scopes=scope
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df

# ----------------------------
# Load data
# ----------------------------
required_cols = ["name", "color", "flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"]

df = None
load_error = None

if source_mode == "CSV (published to web)":
    csv_url = st.sidebar.text_input("Paste CSV URL", placeholder="https://docs.google.com/spreadsheets/d/.../pub?output=csv")
    if csv_url:
        try:
            df = load_from_csv(csv_url)
        except Exception as e:
            load_error = f"Failed to read CSV: {e}"
else:
    # Service account path
    ss_id = st.sidebar.text_input("Spreadsheet ID", placeholder="the long ID in the sheet URL")
    ws_name = st.sidebar.text_input("Worksheet name", value="Sheet1")
    if ss_id and ws_name:
        try:
            df = load_from_service_account(ss_id, ws_name)
        except Exception as e:
            load_error = f"Failed to read via Service Account: {e}"

# ----------------------------
# Fallback sample (optional)
# ----------------------------
use_sample = st.sidebar.toggle("Load minimal sample data", value=False, help="Useful while wiring up Google Sheets.")
if use_sample:
    df = pd.DataFrame([
        ["Neutron activation analysis (NAA)", "blue", 1e10, 1e12, 0.02, 1e4, 20, 25],
        ["Prompt gamma NAA", "blue", 1e6, 1e8, 1e-3, 0.5, 20, 25],
        ["Geochronology - Argon", "blue", 1e15, 1e18, 1e6, 2e7, 20, 200],
        ["Geochronology - Fission tracks", "blue", 1e12, 1e13, 0.02, 0.5, 20, 120],
        ["Si doping", "blue", 1e11, 1e13, 0.02, 0.5, 200, 300],
        ["Gem colorization", "blue", 1e11, 1e12, 1e6, 2e7, 100, 150],
        ["Scattering experiments", "blue", 1e13, 1e14, 1e-3, 2e7, 20, 25],
        ["Material irradiation", "blue", 1e12, 1e14, 1e6, 2e7, 25, 800],
        ["Positron source", "blue", 1e12, 1e14, 0.02, 0.5, 200, 400],
        ["Test & calibration", "blue", 1e11, 1e14, 0.02, 2e7, 200, 300],
        ["Fuel irradiation", "blue", 1e13, 1e15, 0.02, 0.5, 300, 1200],
        ["Education & training", "green", 1e11, 1e13, 1e-3, 2e7, 20, 300],
        ["Isotope production", "red", 1e12, 1e14, 0.02, 0.5, 150, 300],
        ["Boron neutron capture therapy (BNCT)", "red", 1e9, 1e10, 0.02, 1e4, 20, 25],
    ], columns=required_cols)

# Validate / normalize
if df is None:
    if load_error:
        st.error(load_error)
    st.info("Provide a data source on the left (or toggle sample data).")
    st.stop()

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your sheet is missing required columns: {missing}")
    st.stop()

# Coerce numeric columns
for c in ["flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["flux_min", "flux_max", "energy_min", "energy_max", "temp_min", "temp_max"])

# ----------------------------
# Controls
# ----------------------------
st.title("Reactor Applications in Phase Space")
st.caption("Axes: Flux and Energy are on log₁₀ scales (shown as 10^x tick labels). Temperature in °C.")

with st.sidebar.expander("Filters", expanded=False):
    names = ["(All)"] + sorted(df["name"].unique().tolist())
    name_pick = st.selectbox("Application", names)
    colors = ["(All)"] + sorted(df["color"].dropna().astype(str).unique().tolist())
    color_pick = st.selectbox("Color", colors)

    # Plot ranges & ticks (same defaults as your MPL code)
    x_log_ticks = st.multiselect("Flux tick exponents", [6, 8, 10, 12, 14, 16, 18], default=[6, 8, 10, 12, 14, 16, 18])
    y_log_ticks = st.multiselect("Energy tick exponents", [-3, -1, 0, 2, 4, 6, 8], default=[-3, -1, 0, 2, 4, 6, 8])
    xlim = st.slider("Flux exponent range", 0, 20, (6, 18))
    ylim = st.slider("Energy exponent range", -5, 10, (-3, 8))
    zlim = st.slider("Temperature range (°C)", 0, 1500, (0, 1300))

with st.sidebar.expander("Reactor capability overlay", expanded=True):
    show_reactor = st.checkbox("Show reactor capability (5–10 MW)", value=True)
    reactor_flux_range = (1e6, 5e14)
    reactor_energy_range = (0.025, 1e7)
    reactor_temp_range = (20, 1300)
    reactor_alpha = st.slider("Reactor opacity", 0.0, 0.6, 0.15, 0.05)

# Apply filters
plot_df = df.copy()
if name_pick != "(All)":
    plot_df = plot_df[plot_df["name"] == name_pick]
if color_pick != "(All)":
    plot_df = plot_df[plot_df["color"].astype(str) == color_pick]

# ----------------------------
# Geometry helpers
# ----------------------------
def cuboid_vertices(flux_range, energy_range, temp_range):
    # transform X,Y to log10
    x0, x1 = np.log10(flux_range[0]), np.log10(flux_range[1])
    y0, y1 = np.log10(energy_range[0]), np.log10(energy_range[1])
    z0, z1 = temp_range
    # 8 vertices
    V = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])
    # 12 edges as pairs of indices
    edge_pairs = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7)   # verticals
    ]
    return V, edge_pairs

def add_wireframe(fig, name, color, flux_range, energy_range, temp_range, show_legend=True):
    V, E = cuboid_vertices(flux_range, energy_range, temp_range)
    # To avoid 12 legend entries, add one trace with legend, rest without
    first = True
    for (i, j) in E:
        fig.add_trace(go.Scatter3d(
            x=[V[i,0], V[j,0]],
            y=[V[i,1], V[j,1]],
            z=[V[i,2], V[j,2]],
            mode="lines",
            line=dict(width=6, color=color),
            name=name if first and show_legend else None,
            showlegend=first and show_legend,
            hoverinfo="skip",
        ))
        first = False

def add_filled_box(fig, name, color, flux_range, energy_range, temp_range, opacity=0.15):
    # Build a Mesh3d (12 triangles) for the faces
    V, _ = cuboid_vertices(flux_range, energy_range, temp_range)
    # Triangulate faces using vertex indices
    faces = [
        (0,1,2),(0,2,3),   # bottom
        (4,5,6),(4,6,7),   # top
        (0,1,5),(0,5,4),   # side
        (1,2,6),(1,6,5),
        (2,3,7),(2,7,6),
        (3,0,4),(3,4,7),
    ]
    i, j, k = zip(*faces)
    fig.add_trace(go.Mesh3d(
        x=V[:,0], y=V[:,1], z=V[:,2],
        i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, showlegend=True
    ))

# ----------------------------
# Build plot
# ----------------------------
fig = go.Figure()

# Reactor capability (semi-transparent)
if show_reactor:
    add_filled_box(
        fig,
        "Reactor capability (5–10 MW)",
        color="gray",
        flux_range=reactor_flux_range,
        energy_range=reactor_energy_range,
        temp_range=reactor_temp_range,
        opacity=reactor_alpha,
    )

# Each application as a wireframe
for _, r in plot_df.iterrows():
    add_wireframe(
        fig,
        name=str(r["name"]),
        color=str(r["color"]),
        flux_range=(float(r["flux_min"]), float(r["flux_max"])),
        energy_range=(float(r["energy_min"]), float(r["energy_max"])),
        temp_range=(float(r["temp_min"]), float(r["temp_max"])),
        show_legend=True
    )

# Axis ticks/labels like the MPL version
def ticktext(vals):
    return [f"10^{int(v) if v==int(v) else v}" for v in vals]

fig.update_layout(
    scene=dict(
        xaxis=dict(
            title=r"Neutron flux [1/(cm²·s)]",
            tickvals=x_log_ticks,
            ticktext=ticktext(x_log_ticks),
            range=[xlim[0], xlim[1]],
        ),
        yaxis=dict(
            title="Neutron energy [eV]",
            tickvals=y_log_ticks,
            ticktext=ticktext(y_log_ticks),
            range=[ylim[0], ylim[1]],
            autorange="reversed"  # invert like your Matplotlib plot
        ),
        zaxis=dict(
            title=r"Temperature [°C]",
            range=[zlim[0], zlim[1]],
        ),
        aspectmode="cube",
    ),
    legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=30, b=0),
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Data preview
# ----------------------------
with st.expander("Show raw data"):
    st.dataframe(df, use_container_width=True)
