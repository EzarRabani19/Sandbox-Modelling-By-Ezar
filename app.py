import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(
    page_title="GeoStruct 3D — High-Res Structural Geology Sandbox",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("GeoStruct 3D: High-Resolution Structural Geology Sandbox")
st.caption("Pemodelan Blok Geologi 3D Solid (-100 m s.d. 100 m), Reologi Deformasi (Brittle, Brittle-Ductile, Ductile), & Stereonet.")

# --- SIDEBAR: KONTROL PARAMETER GEOLOGI ---
st.sidebar.header("1. Reologi & Jenis Deformasi")
deform_type = st.sidebar.radio(
    "Pilih Jenis Deformasi:",
    [
        "Brittle (Getas — Patahan Tajam)",
        "Brittle-Ductile (Transisi — Patahan + Drag Fold)",
        "Ductile (Liat — Lipatan Kontinu / Shear Zone)"
    ],
    index=1
)

st.sidebar.header("2. Preset Skenario Tektonik")
preset = st.sidebar.selectbox(
    "Pilih Skenario Cepat:",
    [
        "Custom (Atur Manual)",
        "0. Undeformed Strata (Lapisan Datar)",
        "1. Normal Fault (Rift / Ekstensional)",
        "2. Thrust Fault + Fold + Erosi",
        "3. Strike-Slip Fault (Dextral)",
        "4. Oblique-Slip Fault"
    ]
)

defaults = {
    "strike": 0, "dip": 60, "dip_slip": 0.0, "strike_slip": 0.0,
    "fold_amp": 1.5, "drag_width": 25.0, "erosion": False, "erosion_lvl": 25.0
}

if preset == "1. Normal Fault (Rift / Ekstensional)":
    defaults.update({"strike": 0, "dip": 60, "dip_slip": -30.0, "strike_slip": 0.0, "fold_amp": 1.6, "erosion": False})
elif preset == "2. Thrust Fault + Fold + Erosi":
    defaults.update({"strike": 0, "dip": 35, "dip_slip": 35.0, "strike_slip": 0.0, "fold_amp": 2.4, "erosion": True, "erosion_lvl": 24.0})
elif preset == "3. Strike-Slip Fault (Dextral)":
    defaults.update({"strike": 15, "dip": 82, "dip_slip": 0.0, "strike_slip": 45.0, "fold_amp": 1.2, "erosion": False})
elif preset == "4. Oblique-Slip Fault":
    defaults.update({"strike": 25, "dip": 65, "dip_slip": 22.0, "strike_slip": 40.0, "fold_amp": 1.5, "erosion": False})

st.sidebar.header("3. Geometri & Kinematika Struktur")
strike = st.sidebar.slider("Strike (° - Right Hand Rule)", 0, 360, defaults["strike"], 1)
dip = st.sidebar.slider("Dip Angle (°)", 15, 88, defaults["dip"], 1)
dip_slip = st.sidebar.slider("Dip-Slip / Throw Displacement (m)", -60.0, 60.0, defaults["dip_slip"], 0.5)
strike_slip = st.sidebar.slider("Strike-Slip Displacement (m)", -60.0, 60.0, defaults["strike_slip"], 0.5)

st.sidebar.header("4. Parameter Lipatan & Erosi")
is_brittle_only = deform_type.startswith("Brittle (")
fold_amp = st.sidebar.slider(
    "Intensitas Lipatan (Fold Amplitude)",
    0.0, 4.0,
    0.0 if is_brittle_only else defaults["fold_amp"],
    0.1,
    disabled=is_brittle_only
)
drag_width = st.sidebar.slider(
    "Lebar Zona Deformasi / Shear Zone (m)",
    5.0, 70.0,
    defaults["drag_width"],
    1.0,
    disabled=is_brittle_only
)
enable_erosion = st.sidebar.checkbox("Aktifkan Erosi Permukaan", value=defaults["erosion"])
erosion_lvl = st.sidebar.slider("Level Erosi (m)", -10.0, 60.0, defaults["erosion_lvl"], 0.5, disabled=not enable_erosion)

st.sidebar.header("5. Resolusi & Mode Visualisasi")
grid_res = st.sidebar.select_slider(
    "Resolusi Mesh 3D (Grid Nodes)",
    options=[60, 90, 120, 150, 180],
    value=120
)
color_mode = st.sidebar.radio(
    "Mode Pewarnaan Blok:",
    ["Lithology (Formasi Batuan)", "Shear Strain Intensity (Regangan)"]
)
show_fault_plane = st.sidebar.checkbox("Tampilkan Bidang Sesar / Axial Plane", value=True)
show_grid_lines = st.sidebar.checkbox("Tampilkan Garis Kontur Mesh", value=False)

# --- ENGINE KALKULASI VEKTOR GEOLOGI ---
def compute_fault_vectors(strike_deg, dip_deg):
    s_rad = np.radians(strike_deg)
    d_rad = np.radians(dip_deg)

    strike_vec = np.array([np.sin(s_rad), np.cos(s_rad), 0.0])
    strike_vec /= np.linalg.norm(strike_vec)

    dip_dir_horiz = np.array([np.cos(s_rad), -np.sin(s_rad), 0.0])
    dip_vec = np.array([
        dip_dir_horiz[0] * np.cos(d_rad),
        dip_dir_horiz[1] * np.cos(d_rad),
        -np.sin(d_rad)
    ])
    dip_vec /= np.linalg.norm(dip_vec)

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

if abs(dip_slip) <= 0.5 and abs(strike_slip) <= 0.5:
    regime = "Undeformed"
elif deform_type.startswith("Ductile"):
    if abs(dip_slip) > abs(strike_slip) * 1.5:
        regime = "Ductile Monocline / Extensional Rollover" if dip_slip < 0 else "Ductile Fault-Propagation Fold"
    else:
        regime = "Ductile Wrench / Shear Zone"
else:
    if abs(dip_slip) > abs(strike_slip) * 1.5:
        regime = "Extensional (Normal Fault)" if dip_slip < 0 else "Compressional (Reverse/Thrust)"
    elif abs(strike_slip) > abs(dip_slip) * 1.5:
        regime = "Dextral Strike-Slip" if strike_slip > 0 else "Sinistral Strike-Slip"
    else:
        regime = "Oblique-Slip Fault"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rezim Struktur", regime)
m2.metric("Net Slip / Offset", f"{net_slip:.2f} m")
m3.metric("Throw (Vertikal)", f"{throw:.2f} m")
m4.metric("Heave (Horizontal)", f"{heave:.2f} m")
m5.metric("Rake / Pitch", f"{rake_deg:.1f}°")

# --- DEFINISI STRATIGRAFI DETAIL (7 LAPISAN) ---
layers = [
    {"name": "Alluvium / Topsoil", "z_top": 30.0, "z_bot": 23.0, "color": "#C05621"},
    {"name": "Upper Sandstone", "z_top": 23.0, "z_bot": 15.0, "color": "#ECC94B"},
    {"name": "Shale Caprock", "z_top": 15.0, "z_bot": 8.0, "color": "#2F855A"},
    {"name": "Reef Limestone (Reservoir)", "z_top": 8.0, "z_bot": -2.0, "color": "#3182CE"},
    {"name": "Siltstone / Mudstone", "z_top": -2.0, "z_bot": -11.0, "color": "#38B2AC"},
    {"name": "Lower Conglomerate", "z_top": -11.0, "z_bot": -20.0, "color": "#B7791F"},
    {"name": "Crystalline Basement", "z_top": -20.0, "z_bot": -35.0, "color": "#4A5568"},
]

block_x, block_y = 100.0, 100.0
x_lin = np.linspace(-block_x, block_x, grid_res)
y_lin = np.linspace(-block_y, block_y, grid_res)
X_base, Y_base = np.meshgrid(x_lin, y_lin)

def apply_kinematics(X, Y, Z, side_mask):
    """Menerapkan kinematika Brittle, Brittle-Ductile, atau Ductile secara presisi."""
    pts = np.stack([X, Y, Z], axis=-1)
    signed_dist = np.tensordot(pts, normal_vec, axes=([2], [0]))
    disp_total = (-dip_slip * dip_vec) + (strike_slip * strike_vec)

    # --- MODE 3: DUCTILE (LIPATAN KONTINU / DUCTILE SHEAR ZONE TANPA PATAHAN DISKRET) ---
    if deform_type.startswith("Ductile"):
        # Gunakan kurva transisi hiperbolik tangen (tanh) yang mulus melintasi zona geser
        shear_profile = 0.5 * (1.0 + np.tanh(signed_dist / (drag_width * 0.65)))
        
        X_out = X + shear_profile * disp_total[0]
        Y_out = Y + shear_profile * disp_total[1]
        Z_out = Z + shear_profile * disp_total[2]

        # Tambahkan pembubungan lipatan (buckle/drag folding) di inti zona geser
        sech2 = 1.0 / (np.cosh(signed_dist / (drag_width * 0.65)) ** 2)
        fold_direction = 1.0 if dip_slip >= 0 else -0.6
        Z_out = Z_out + fold_direction * sech2 * (abs(dip_slip) * 0.22) * fold_amp

        strain_val = sech2 * (net_slip / 50.0) * (1.0 + fold_amp * 0.35)

        if enable_erosion:
            Z_out = np.where(Z_out > erosion_lvl, erosion_lvl, Z_out)

        return X_out, Y_out, Z_out, strain_val

    # --- MODE 1 (BRITTLE) & MODE 2 (BRITTLE-DUCTILE) ---
    tol = (2 * block_x / grid_res) * 0.65
    if side_mask == "footwall":
        valid = signed_dist <= tol
        snap_mask = (signed_dist > 0) & (signed_dist <= tol)
    else:
        valid = signed_dist >= -tol
        snap_mask = (signed_dist < 0) & (signed_dist >= -tol)

    X_out = np.where(valid, X, np.nan)
    Y_out = np.where(valid, Y, np.nan)
    Z_out = np.where(valid, Z, np.nan)

    if np.any(snap_mask):
        X_out = np.where(snap_mask, X_out - signed_dist * normal_vec[0], X_out)
        Y_out = np.where(snap_mask, Y_out - signed_dist * normal_vec[1], Y_out)
        Z_out = np.where(snap_mask, Z_out - signed_dist * normal_vec[2], Z_out)
        signed_dist = np.where(snap_mask, 0.0, signed_dist)

    if is_brittle_only:
        # Brittle murni: tidak ada pelengkungan drag fold, patahan tajam 100%
        drag_factor = np.zeros_like(signed_dist)
        strain_val = np.exp(-(signed_dist**2) / (2 * 6.0**2)) * (net_slip / 60.0)
    else:
        # Brittle-Ductile: kombinasi patahan tajam + pelengkungan drag di zona sesar
        # Kedalaman lebih dalam (Z lebih negatif) meningkatkan keplastisan (ductility)
        depth_ductility = np.clip(1.0 + (-Z / 45.0) * 0.5, 0.75, 1.5)
        drag_factor = np.exp(-(signed_dist**2) / (2 * drag_width**2))
        drag_sign = -1.0 if side_mask == "hanging_wall" else 1.0
        Z_out = Z_out + drag_sign * drag_factor * (dip_slip * 0.25) * fold_amp * depth_ductility
        strain_val = drag_factor * (net_slip / 60.0) * (1.0 + fold_amp * 0.3)

    if side_mask == "hanging_wall":
        X_out = X_out + disp_total[0]
        Y_out = Y_out + disp_total[1]
        Z_out = Z_out + disp_total[2]

    if enable_erosion:
        Z_out = np.where(Z_out > erosion_lvl, erosion_lvl, Z_out)

    return X_out, Y_out, Z_out, strain_val

# --- RENDER VISUALISASI 3D & STEREONET ---
col_3d, col_stereo = st.columns([2.4, 1.0])

with col_3d:
    fig3d = go.Figure()
    contour_cfg = dict(
        x=dict(show=show_grid_lines, color="rgba(0,0,0,0.3)", width=1),
        y=dict(show=show_grid_lines, color="rgba(0,0,0,0.3)", width=1)
    )

    # Jika Ductile penuh, cukup render 1 blok kontinu; jika Brittle / Brittle-Ductile, render 2 blok terpisah
    sides_to_render = ["continuous"] if deform_type.startswith("Ductile") else ["footwall", "hanging_wall"]

    for idx, layer in enumerate(layers):
        for s_idx, side in enumerate(sides_to_render):
            Xt, Yt, Zt, St = apply_kinematics(X_base, Y_base, np.full_like(X_base, layer["z_top"]), side)
            
            if color_mode == "Lithology (Formasi Batuan)":
                surf_kwargs = dict(
                    colorscale=[[0, layer["color"]], [1, layer["color"]]],
                    showscale=False
                )
            else:
                surf_kwargs = dict(
                    surfacecolor=St,
                    colorscale="Turbo",
                    cmin=0.0,
                    cmax=1.5,
                    showscale=(idx == 0 and s_idx == 0),
                    colorbar=dict(title="Shear Strain", x=1.02, len=0.6)
                )

            fig3d.add_trace(go.Surface(
                x=Xt, y=Yt, z=Zt,
                name=layer["name"],
                showlegend=(s_idx == 0 and color_mode == "Lithology (Formasi Batuan)"),
                opacity=1.0,
                contours=contour_cfg,
                **surf_kwargs
            ))

            # Dinding Samping Solid (4 Sisi)
            wall_res_z = 6
            z_wall_lin = np.linspace(layer["z_bot"], layer["z_top"], wall_res_z)

            for y_edge in [-block_y, block_y]:
                Xw, Zw = np.meshgrid(x_lin, z_wall_lin)
                Yw = np.full_like(Xw, y_edge)
                Xwd, Ywd, Zwd, Swd = apply_kinematics(Xw, Yw, Zw, side)
                wall_color = dict(surfacecolor=Swd, colorscale="Turbo", cmin=0.0, cmax=1.5, showscale=False) if color_mode != "Lithology (Formasi Batuan)" else dict(colorscale=[[0, layer["color"]], [1, layer["color"]]], showscale=False)
                fig3d.add_trace(go.Surface(x=Xwd, y=Ywd, z=Zwd, showlegend=False, opacity=1.0, **wall_color))

            for x_edge in [-block_x, block_x]:
                Yw, Zw = np.meshgrid(y_lin, z_wall_lin)
                Xw = np.full_like(Yw, x_edge)
                Xwd, Ywd, Zwd, Swd = apply_kinematics(Xw, Yw, Zw, side)
                wall_color = dict(surfacecolor=Swd, colorscale="Turbo", cmin=0.0, cmax=1.5, showscale=False) if color_mode != "Lithology (Formasi Batuan)" else dict(colorscale=[[0, layer["color"]], [1, layer["color"]]], showscale=False)
                fig3d.add_trace(go.Surface(x=Xwd, y=Ywd, z=Zwd, showlegend=False, opacity=1.0, **wall_color))

            # Alas Bawah Basement
            if idx == len(layers) - 1:
                Xb, Yb, Zb, _ = apply_kinematics(X_base, Y_base, np.full_like(X_base, layer["z_bot"]), side)
                fig3d.add_trace(go.Surface(
                    x=Xb, y=Yb, z=Zb,
                    colorscale=[[0, layer["color"]], [1, layer["color"]]],
                    showscale=False,
                    showlegend=False,
                    opacity=1.0
                ))

    # Render Bidang Sesar / Axial Shear Plane
    if show_fault_plane:
        u = np.linspace(-135, 135, 20)
        v = np.linspace(-65, 65, 20)
        U, V = np.meshgrid(u, v)
        Xf = U * strike_vec[0] + V * dip_vec[0]
        Yf = U * strike_vec[1] + V * dip_vec[1]
        Zf = U * strike_vec[2] + V * dip_vec[2]

        plane_label = "Axial / Shear Zone Plane" if deform_type.startswith("Ductile") else "Fault Plane"
        fig3d.add_trace(go.Surface(
            x=Xf, y=Yf, z=Zf,
            colorscale=[[0, "#FF3B3B"], [1, "#FF3B3B"]],
            showscale=False,
            opacity=0.28,
            name=plane_label,
            showlegend=True
        ))

    short_mode = deform_type.split(" (")[0]
    fig3d.update_layout(
        title=f"Model Blok 3D [{short_mode}] (-100 m s.d. 100 m | {grid_res}×{grid_res} Grid)",
        scene=dict(
            xaxis=dict(title="East (m)", range=[-120, 120]),
            yaxis=dict(title="North (m)", range=[-120, 120]),
            zaxis=dict(title="Elevation (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=-1.65, z=1.05))
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=660,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig3d, use_container_width=True)

with col_stereo:
    st.subheader("Proyeksi Stereonet (Schmidt Net)")
    
    theta = np.linspace(0, 2 * np.pi, 360)
    fig_stereo = go.Figure()
    fig_stereo.add_trace(go.Scatter(
        x=np.sin(theta), y=np.cos(theta),
        mode="lines", line=dict(color="#888", width=2),
        showlegend=False
    ))

    fig_stereo.add_trace(go.Scatter(
        x=[0, 1.08, 0, -1.08], y=[1.08, 0, -1.08, 0],
        mode="text", text=["N", "E", "S", "W"],
        textfont=dict(size=12, color="#CBD5E0"),
        showlegend=False
    ))

    alpha = np.linspace(-np.pi/2, np.pi/2, 180)
    gx, gy = [], []
    for a in alpha:
        pt = np.cos(a) * strike_vec + np.sin(a) * dip_vec
        if pt[2] > 0:
            pt = -pt
        horiz_norm = np.sqrt(pt[0]**2 + pt[1]**2)
        if horiz_norm > 1e-6:
            R = np.sqrt(1.0 + pt[2])
            gx.append(R * (pt[0] / horiz_norm))
            gy.append(R * (pt[1] / horiz_norm))

    fig_stereo.add_trace(go.Scatter(
        x=gx, y=gy, mode="lines",
        line=dict(color="#FF4D4D", width=3),
        name=f"Plane ({strike:03d}°/{dip:02d}°)"
    ))

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
        name="Pole to Plane"
    ))

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
            marker=dict(color="#F6AD55", size=12, symbol="star"),
            name=f"Transport / Slip (Rake {rake_deg:.0f}°)"
        ))

    fig_stereo.update_layout(
        xaxis=dict(range=[-1.2, 1.2], visible=False),
        yaxis=dict(range=[-1.2, 1.2], visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
        height=390,
        legend=dict(orientation="h", yanchor="top", y=-0.02)
    )
    st.plotly_chart(fig_stereo, use_container_width=True)

    summary_df = pd.DataFrame({
        "Parameter": [
            "Rheology Mode", "Strike (°)", "Dip (°)", "Dip-Slip (m)", "Strike-Slip (m)",
            "Net Slip (m)", "Throw (m)", "Heave (m)", "Rake (°)", "Regime"
        ],
        "Nilai": [
            short_mode, strike, dip, dip_slip, strike_slip,
            round(net_slip, 2), round(throw, 2), round(heave, 2), round(rake_deg, 1), regime
        ]
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    csv_data = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Parameter Struktur (CSV)",
        data=csv_data,
        file_name=f"geostruct_3d_{short_mode}_S{strike}_D{dip}.csv",
        mime="text/csv",
        use_container_width=True
    )
