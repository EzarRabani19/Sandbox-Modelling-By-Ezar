import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(
    page_title="GeoStruct 3D — Structural Geology Sandbox",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("GeoStruct 3D: Kinematic Structural Geology Sandbox")
st.caption("Simulasi 3D Blok Stratigrafi, Kinematika Sesar (Normal, Reverse, Strike-Slip, Oblique), Fault-Drag Folding, & Erosi.")

# --- SIDEBAR: KONTROL PARAMETER GEOLOGI ---
st.sidebar.header("1. Preset Skenario Tektonik")
preset = st.sidebar.selectbox(
    "Pilih Skenario Cepat:",
    [
        "Custom (Atur Manual)",
        "0. Undeformed Strata (Lapisan Datar)",
        "1. Normal Fault (Rift / Ekstensional)",
        "2. Thrust Fault + Drag Fold + Erosi",
        "3. Strike-Slip Fault (Dextral)",
        "4. Oblique-Slip Fault"
    ]
)

# Nilai default berdasarkan preset
defaults = {
    "strike": 0, "dip": 60, "dip_slip": 0.0, "strike_slip": 0.0,
    "fold_amp": 1.5, "drag_width": 4.5, "erosion": False, "erosion_lvl": 4.5
}

if preset == "1. Normal Fault (Rift / Ekstensional)":
    defaults.update({"strike": 0, "dip": 60, "dip_slip": -5.0, "strike_slip": 0.0, "fold_amp": 1.6, "erosion": False})
elif preset == "2. Thrust Fault + Drag Fold + Erosi":
    defaults.update({"strike": 0, "dip": 35, "dip_slip": 5.5, "strike_slip": 0.0, "fold_amp": 2.4, "erosion": True, "erosion_lvl": 4.2})
elif preset == "3. Strike-Slip Fault (Dextral)":
    defaults.update({"strike": 15, "dip": 82, "dip_slip": 0.0, "strike_slip": 7.0, "fold_amp": 0.8, "erosion": False})
elif preset == "4. Oblique-Slip Fault":
    defaults.update({"strike": 25, "dip": 65, "dip_slip": 3.5, "strike_slip": 6.0, "fold_amp": 1.4, "erosion": False})

st.sidebar.header("2. Geometri & Kinematika Sesar")
strike = st.sidebar.slider("Strike (° - Right Hand Rule)", 0, 360, defaults["strike"], 1)
dip = st.sidebar.slider("Dip Angle (°)", 15, 88, defaults["dip"], 1)
dip_slip = st.sidebar.slider("Dip-Slip (m) [- Normal | + Reverse]", -10.0, 10.0, defaults["dip_slip"], 0.2)
strike_slip = st.sidebar.slider("Strike-Slip (m) [- Sinistral | + Dextral]", -10.0, 10.0, defaults["strike_slip"], 0.2)

st.sidebar.header("3. Lipatan & Proses Permukaan")
fold_amp = st.sidebar.slider("Intensitas Fault-Drag Fold", 0.0, 4.0, defaults["fold_amp"], 0.1)
drag_width = st.sidebar.slider("Lebar Zona Shear/Drag (m)", 1.0, 10.0, defaults["drag_width"], 0.5)
enable_erosion = st.sidebar.checkbox("Aktifkan Erosi Permukaan", value=defaults["erosion"])
erosion_lvl = st.sidebar.slider("Level Erosi (m)", -2.0, 10.0, defaults["erosion_lvl"], 0.2, disabled=not enable_erosion)

st.sidebar.header("4. Tampilan Visual")
show_fault_plane = st.sidebar.checkbox("Tampilkan Bidang Sesar", value=True)
grid_res = st.sidebar.select_slider("Resolusi Mesh 3D", options=[30, 45, 60, 80], value=50)

# --- ENGINE KALKULASI VEKTOR GEOLOGI ---
def compute_fault_vectors(strike_deg, dip_deg):
    s_rad = np.radians(strike_deg)
    d_rad = np.radians(dip_deg)

    # Vektor Strike (Sistem koordinat: X = East, Y = North, Z = Elevation)
    strike_vec = np.array([np.sin(s_rad), np.cos(s_rad), 0.0])
    strike_vec /= np.linalg.norm(strike_vec)

    # Arah Dip (90 derajat searah jarum jam dari Strike sesuai Right-Hand Rule)
    dip_dir_horiz = np.array([np.cos(s_rad), -np.sin(s_rad), 0.0])
    dip_vec = np.array([
        dip_dir_horiz[0] * np.cos(d_rad),
        dip_dir_horiz[1] * np.cos(d_rad),
        -np.sin(d_rad)
    ])
    dip_vec /= np.linalg.norm(dip_vec)

    # Normal bidang sesar mengarah ke Hanging Wall (ke atas/arah dip)
    normal = np.cross(strike_vec, dip_vec)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    return strike_vec, dip_vec, normal

strike_vec, dip_vec, normal_vec = compute_fault_vectors(strike, dip)

# Perhitungan Metrik Struktur
dip_rad = np.radians(dip)
heave = abs(dip_slip * np.cos(dip_rad))
throw = abs(dip_slip * np.sin(dip_rad))
net_slip = np.sqrt(dip_slip**2 + strike_slip**2)
rake_deg = np.degrees(np.arctan2(dip_slip, strike_slip)) if net_slip > 0.01 else 0.0

if abs(dip_slip) <= 0.2 and abs(strike_slip) <= 0.2:
    regime = "Undeformed"
elif abs(dip_slip) > abs(strike_slip) * 1.5:
    regime = "Extensional (Normal Fault)" if dip_slip < 0 else "Compressional (Reverse/Thrust)"
elif abs(strike_slip) > abs(dip_slip) * 1.5:
    regime = "Dextral Strike-Slip" if strike_slip > 0 else "Sinistral Strike-Slip"
else:
    regime = "Oblique-Slip Fault"

# Tampilkan Metrik di Atas
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rezim Struktur", regime)
m2.metric("Net Slip", f"{net_slip:.2f} m")
m3.metric("Throw (Vertikal)", f"{throw:.2f} m")
m4.metric("Heave (Horizontal)", f"{heave:.2f} m")
m5.metric("Rake / Pitch", f"{rake_deg:.1f}°")

# --- PEMODELAN 3D STRATIGRAFI & DEFORMASI ---
layers = [
    {"name": "Alluvium / Topsoil", "z_top": 5.0, "z_bot": 3.5, "color": "#C05621"},
    {"name": "Limestone (Reservoir)", "z_top": 3.5, "z_bot": 1.5, "color": "#3182CE"},
    {"name": "Shale (Caprock)", "z_top": 1.5, "z_bot": 0.0, "color": "#2F855A"},
    {"name": "Sandstone", "z_top": 0.0, "z_bot": -2.0, "color": "#D69E2E"},
    {"name": "Basement (Granite)", "z_top": -2.0, "z_bot": -5.0, "color": "#4A5568"},
]

block_x, block_y = 15.0, 12.0
x_lin = np.linspace(-block_x, block_x, grid_res)
y_lin = np.linspace(-block_y, block_y, grid_res)
X_base, Y_base = np.meshgrid(x_lin, y_lin)

def deform_surface(X, Y, z_init, side_mask):
    """Menerapkan kinematika sesar + fault-drag folding + erosi pada grid permukaan."""
    pts = np.stack([X, Y, np.full_like(X, z_init)], axis=-1)
    signed_dist = np.tensordot(pts, normal_vec, axes=([2], [0]))

    # Masking sisi blok (Footwall vs Hanging Wall) agar patahan terputus tajam
    valid = (signed_dist <= 0.0) if side_mask == "footwall" else (signed_dist > 0.0)
    
    X_out = np.where(valid, X, np.nan)
    Y_out = np.where(valid, Y, np.nan)
    Z_out = np.where(valid, z_init, np.nan)

    # Efek Fault-Drag Folding (melengkung di sekitar bidang sesar)
    drag_factor = np.exp(-(signed_dist**2) / (2 * drag_width**2))
    drag_sign = np.where(signed_dist >= 0, -1.0, 1.0)
    Z_out = Z_out + drag_sign * drag_factor * (dip_slip * 0.25) * fold_amp

    # Pergeseran blok Hanging Wall sepanjang vektor Dip-Slip & Strike-Slip
    if side_mask == "hanging_wall":
        disp = (-dip_slip * dip_vec) + (strike_slip * strike_vec)
        X_out = X_out + disp[0]
        Y_out = Y_out + disp[1]
        Z_out = Z_out + disp[2]

    # Simulasi Erosi Permukaan
    if enable_erosion:
        Z_out = np.where(Z_out > erosion_lvl, erosion_lvl, Z_out)

    return X_out, Y_out, Z_out

# --- RENDER VISUALISASI 3D & STEREONET ---
col_3d, col_stereo = st.columns([2.3, 1.0])

with col_3d:
    fig3d = go.Figure()

    # Render setiap lapisan pada Footwall & Hanging Wall
    for idx, layer in enumerate(layers):
        for side in ["footwall", "hanging_wall"]:
            Xt, Yt, Zt = deform_surface(X_base, Y_base, layer["z_top"], side)
            fig3d.add_trace(go.Surface(
                x=Xt, y=Yt, z=Zt,
                colorscale=[[0, layer["color"]], [1, layer["color"]]],
                showscale=False,
                name=f"{layer['name']} ({side})",
                showlegend=(side == "footwall"),
                opacity=0.95,
                contours=dict(
                    x=dict(show=True, color="rgba(0,0,0,0.25)", width=1),
                    y=dict(show=True, color="rgba(0,0,0,0.25)", width=1)
                )
            ))

            # Render bagian dasar lapisan terbawah (Basement) agar blok tertutup rapi
            if idx == len(layers) - 1:
                Xb, Yb, Zb = deform_surface(X_base, Y_base, layer["z_bot"], side)
                fig3d.add_trace(go.Surface(
                    x=Xb, y=Yb, z=Zb,
                    colorscale=[[0, layer["color"]], [1, layer["color"]]],
                    showscale=False,
                    showlegend=False,
                    opacity=0.95
                ))

    # Render Bidang Sesar (Fault Plane)
    if show_fault_plane:
        u = np.linspace(-16, 16, 10)
        v = np.linspace(-9, 9, 10)
        U, V = np.meshgrid(u, v)
        Xf = U * strike_vec[0] + V * dip_vec[0]
        Yf = U * strike_vec[1] + V * dip_vec[1]
        Zf = U * strike_vec[2] + V * dip_vec[2]

        fig3d.add_trace(go.Surface(
            x=Xf, y=Yf, z=Zf,
            colorscale=[[0, "#FF4D4D"], [1, "#FF4D4D"]],
            showscale=False,
            opacity=0.35,
            name="Fault Plane",
            showlegend=True
        ))

    fig3d.update_layout(
        title="Model Blok Geologi Struktur 3D",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.1))
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig3d, use_container_width=True)

with col_stereo:
    # Proyeksi Equal-Area (Schmidt Net) Bidang Sesar & Slickenside Lineation
    st.subheader("Proyeksi Stereonet (Schmidt Net)")
    
    # Lingkaran luar stereonet
    theta = np.linspace(0, 2 * np.pi, 200)
    fig_stereo = go.Figure()
    fig_stereo.add_trace(go.Scatter(
        x=np.sin(theta), y=np.cos(theta),
        mode="lines", line=dict(color="#888", width=2),
        name="Primitive Circle", showlegend=False
    ))

    # Hitung Great Circle bidang sesar (Equal-Area Schmidt projection)
    alpha = np.linspace(-np.pi/2, np.pi/2, 100)
    s_rad = np.radians(strike)
    d_rad = np.radians(dip)
    
    # Vektor titik pada great circle di 3D (lower hemisphere)
    gx, gy = [], []
    for a in alpha:
        pt = np.cos(a) * strike_vec + np.sin(a) * dip_vec
        if pt[2] > 0:
            pt = -pt
        # Schmidt Equal-Area formula: R = sqrt(1 - |z|)
        horiz_norm = np.sqrt(pt[0]**2 + pt[1]**2)
        if horiz_norm > 1e-6:
            R = np.sqrt(1.0 + pt[2]) # karena pt[2] <= 0
            gx.append(R * (pt[0] / horiz_norm))
            gy.append(R * (pt[1] / horiz_norm))

    fig_stereo.add_trace(go.Scatter(
        x=gx, y=gy, mode="lines",
        line=dict(color="#FF4D4D", width=3),
        name=f"Fault ({strike:03d}°/{dip:02d}°)"
    ))

    # Plot Pole to Fault Plane
    pole = -normal_vec if normal_vec[2] > 0 else normal_vec
    p_horiz = np.sqrt(pole[0]**2 + pole[1]**2)
    if p_horiz > 1e-6:
        Rp = np.sqrt(1.0 + pole[2])
        px, py = Rp * (pole[0] / p_horiz), Rp * (pole[1] / p_horiz)
    else:
        px, py = 0.0, 0.0

    fig_stereo.add_trace(go.Scatter(
        x=[px], y=[py], mode="markers",
        marker=dict(color="#3182CE", size=10, symbol="diamond"),
        name="Pole to Fault"
    ))

    # Plot Net-Slip Vector (Slickenside)
    if net_slip > 0.05:
        slip_3d = (-dip_slip * dip_vec) + (strike_slip * strike_vec)
        slip_3d /= np.linalg.norm(slip_3d)
        if slip_3d[2] > 0:
            slip_3d = -slip_3d
        s_horiz = np.sqrt(slip_3d[0]**2 + slip_3d[1]**2)
        Rs = np.sqrt(1.0 + slip_3d[2])
        sx, sy = Rs * (slip_3d[0] / s_horiz), Rs * (slip_3d[1] / s_horiz)

        fig_stereo.add_trace(go.Scatter(
            x=[sx], y=[sy], mode="markers",
            marker=dict(color="#F6AD55", size=11, symbol="star"),
            name=f"Slip Vector (Rake {rake_deg:.0f}°)"
        ))

    fig_stereo.update_layout(
        xaxis=dict(range=[-1.15, 1.15], visible=False),
        yaxis=dict(range=[-1.15, 1.15], visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=20, b=10),
        height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.02)
    )
    st.plotly_chart(fig_stereo, use_container_width=True)

    # Ringkasan & Ekspor Data CSV
    summary_df = pd.DataFrame({
        "Parameter": ["Strike (°)", "Dip (°)", "Dip-Slip (m)", "Strike-Slip (m)", "Net Slip (m)", "Throw (m)", "Heave (m)", "Rake (°)", "Regime"],
        "Nilai": [strike, dip, dip_slip, strike_slip, round(net_slip, 2), round(throw, 2), round(heave, 2), round(rake_deg, 1), regime]
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    csv_data = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Parameter Sesar (CSV)",
        data=csv_data,
        file_name=f"geostruct_3d_S{strike}_D{dip}.csv",
        mime="text/csv",
        use_container_width=True
    )
