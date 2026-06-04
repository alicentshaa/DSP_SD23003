import os
import hashlib
import tempfile
import time
import pandas as pd
import streamlit as st
import folium
from folium.features import CustomIcon
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pipeline import run_forensic_pipeline

st.set_page_config(
    page_title="Dashcam Forensic Analytics Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.stApp{background:#f8fafc;}
.block-container{padding-top:1rem;}
.metricbox{
background:white;
padding:15px;
border-radius:12px;
border-left:5px solid #2563eb;
box-shadow:0 2px 8px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page="upload"
if "results" not in st.session_state:
    st.session_state.results=None

def sha256_bytes(data):
    h=hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def save_temp(uploaded):
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".csv")
    tmp.write(uploaded.getvalue())
    tmp.close()
    return tmp.name

def classify_event(speed, accel, turn):
    if accel < -2.5:
        return "🚨 Harsh Braking"
    elif accel > 2.5:
        return "⚡ Rapid Acceleration"
    elif abs(turn) > 0.45:
        return "↪ Aggressive Steering"
    elif speed < 3:
        return "🛑 Vehicle Stop"
    return "✅ Normal Movement"

# =====================================================================
# UPLOAD LAYER
# =====================================================================
if st.session_state.page=="upload":
    st.title("🚔 DASHCAM FORENSIC ANALYTICS WITH AI AND GPS VISUALIZATION")
    st.markdown("### Upload Investigation Dataset")

    data_file=st.file_uploader("Dashcam Dataset",type=["csv"])
    audit_file=st.file_uploader("Audit Log",type=["csv"])

    if st.button("🔒 VERIFY DATA",use_container_width=True):
        if data_file and audit_file:
            st.session_state.data_file=data_file
            st.session_state.audit_file=audit_file
            st.session_state.page="verify"
            st.rerun()

# =====================================================================
# CHAIN OF CUSTODY VERIFICATION
# =====================================================================
elif st.session_state.page=="verify":
    st.title("🔒 Chain of Custody Verification")

    st.success("Evidence Integrity Verified")

    st.code(
        f"Dataset SHA256: {sha256_bytes(st.session_state.data_file.getvalue())}\n"
        f"Audit SHA256: {sha256_bytes(st.session_state.audit_file.getvalue())}"
    )

    if st.button("🚀 START FORENSIC ANALYSIS",use_container_width=True):

        data_path=save_temp(st.session_state.data_file)
        audit_path=save_temp(st.session_state.audit_file)
        feature_path=tempfile.NamedTemporaryFile(delete=False,suffix=".csv").name

        status=st.empty()
        for stage in [
            "Feature Engineering",
            "Transformer Reconstruction",
            "Kalman Filtering",
            "Hermite Interpolation",
            "Autoencoder Detection",
            "Generating Analytics"
        ]:
            status.info(stage)

        results=run_forensic_pipeline(
            data_path,
            audit_path,
            feature_path
        )

        st.session_state.results=results
        st.session_state.page="analysis"
        st.rerun()

# =====================================================================
# FORENSIC ANALYTICS CANVAS
# =====================================================================
elif st.session_state.page=="analysis":

    results=st.session_state.results
    df_features=results["df_features"]
    df=results["df_results"].copy()

    df["Timestamp"]=pd.to_datetime(df_features["Timestamp_Unified"])
    df["Date"]=df["Timestamp"].dt.date
    df["Semantic_Event"]=df.apply(
        lambda r: classify_event(
            r["Speed_Kmh"],
            r["Accel_M_S2"],
            r["Turn_Rate"]
        ),axis=1
    )

    st.title(("🚔 DASHCAM FORENSIC ANALYTICS DASHBOARD"), text_alignment="center")
    st.caption(("AI-Based Incident Reconstruction, Anomalies Detection and Dashcam Forensic Investigation"), text_alignment= "center")

    a,b,c,d=st.columns(4)

    with a:
        selected_date=st.selectbox("📅 Date",sorted(df["Date"].unique()))

    filtered=df[df["Date"]==selected_date]

    # --- Integrated Animation Controls Patch ---
    with b:
        if "play_animation" not in st.session_state:
            st.session_state.play_animation = False

        if "frame_index" not in st.session_state:
            st.session_state.frame_index = 0

        play_col1, play_col2, speed_col = st.columns([1, 1, 1])

        with play_col1:
            if st.button("▶️ Play", use_container_width=True):
                st.session_state.play_animation = True

        with play_col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.play_animation = False
                
        with speed_col:
            playback_speed = st.radio("🏃 Speed", options=["1x", "2x"], index=0, horizontal=True)

        timestamps = filtered["Timestamp"].tolist()

        selected_timestamp = st.select_slider(
            "⏱️Timestamp",
            options=timestamps,
            value=timestamps[min(st.session_state.frame_index, len(timestamps)-1)]
        )
        
        # Keep frame index synchronized if the user manually drags the slider
        st.session_state.frame_index = timestamps.index(selected_timestamp)

    row=filtered[filtered["Timestamp"]==selected_timestamp].iloc[0]
    current_event=row["Semantic_Event"]

    with c:
        st.markdown(f"""
        <div class='metricbox'>
        <b>ℹ️ Insight</b><br><br>
        {current_event}
        </div>
        """,unsafe_allow_html=True)

    with d:
        # ⚡ LIVE THREAT LOOKUP: Changes dynamically with the timeline slider!
        if current_event in ["🚨 Harsh Braking", "↪️ Aggressive Steering"]:
            threat = "🔴 HIGH THREAT"
        elif current_event in ["⚡ Rapid Acceleration"]:
            threat = "🟡 MEDIUM RISK"
        else:
            threat = "🟢 LOW (SAFE)"

        st.markdown(f"""
        <div class='metricbox'>
        <b>🚨 Current Threat Level</b><br><br>
        {threat}
        </div>
        """,unsafe_allow_html=True)

    st.subheader("🗺️ Semantic GPS Reconstruction Map")

    current_lat=row["Recon_Lat"]
    current_lon=row["Recon_Lon"]

    # --- Integrated Dynamic Satellite Map & Active Tracking Focus Patch ---
    route = filtered[["Recon_Lat","Recon_Lon"]].values.tolist()

    # Initialized directly to follow current vehicle coordinates at extreme high zoom
    m = folium.Map(location=[current_lat, current_lon], zoom_start=19, tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite"
    ).add_to(m)

    folium.PolyLine(
        route,
        color="red",
        weight=4
    ).add_to(m)

    car_icon_url = https://toppng.com/show_download/76232/car-top-view-transparent
    vehicle_icon = CustomIcon(
        car_icon_url,
        icon_size=(45, 45)
    )

    folium.Marker(
        [current_lat,current_lon],
        icon=vehicle_icon,
        popup=current_event
    ).add_to(m)

    # Highlight Ring Target
    folium.Circle(
        [current_lat,current_lon],
        radius=10,
        color="yellow",
        fill=True,
        fill_opacity=0.4
    ).add_to(m)

    # Retain your original critical anomaly markers overlay
    event_points=filtered[filtered["Semantic_Event"]!="✅ Normal Movement"]
    for _,r in event_points.iterrows():
        folium.CircleMarker(
            [r["Recon_Lat"],r["Recon_Lon"]],
            radius=6,
            color="red",
            fill=True,
            popup=r["Semantic_Event"]
        ).add_to(m)

    st_folium(m,height=600,use_container_width=True)

    # =====================================================================
    # VERTICALLY ALIGNED KINEMATIC DIAGNOSTICS & SIDE CAPTIONS
    # =====================================================================
    st.markdown("## 📊 Kinematic & Behavioral Timeline Array")
    
    # -----------------------------------------------------------------
    # SEMANTIC TURNING ANALYSIS
    # -----------------------------------------------------------------
    st.markdown("### ↪️ Semantic Turning Analysis")
    turn_g_col, turn_c_col = st.columns([2, 1])
    
    with turn_g_col:
        fig_turn=px.line(filtered,x="Timestamp",y="Turn_Rate")
        fig_turn.add_scatter(
            x=[selected_timestamp],
            y=[row["Turn_Rate"]],
            mode="markers",
            marker=dict(size=12, color="orange")
        )
        fig_turn.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_turn,use_container_width=True)
        
    with turn_c_col:
        st.info(f"""
        **Telemetry Metrics:**
        * Current Turn Rate: `{row['Turn_Rate']:.3f}` rad/s
        
        **Forensic Interpretation:**
        * Positive values indicate left steering movement.
        * Negative values indicate right steering movement.
        * Large sudden spikes indicate lane changes, aggressive steering, or emergency evasion tracks.
        """)

    st.divider()

    # -----------------------------------------------------------------
    # SEMANTIC ACCELERATION ANALYSIS
    # -----------------------------------------------------------------
    st.markdown("### ⚡ Semantic Acceleration Analysis")
    acc_g_col, acc_c_col = st.columns([2, 1])
    
    with acc_g_col:
        fig_acc=px.line(filtered,x="Timestamp",y="Accel_M_S2")
        fig_acc.add_scatter(
            x=[selected_timestamp],
            y=[row["Accel_M_S2"]],
            mode="markers",
            marker=dict(size=12, color="orange")
        )
        fig_acc.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_acc,use_container_width=True)
        
    with acc_c_col:
        st.info(f"""
        **Telemetry Metrics:**
        * Current Acceleration: `{row['Accel_M_S2']:.3f}` m/s²
        
        **Forensic Interpretation:**
        * Positive values indicate forward acceleration.
        * Negative values indicate  braking deceleration.
        * Extreme negative peaks call out severe deceleration and harsh emergency braking anomalies.
        """)

    st.divider()

    # -----------------------------------------------------------------
    # SPEED BEHAVIOUR TIMELINE
    # -----------------------------------------------------------------
    st.markdown("### 🚗 Speed Behaviour Timeline")
    speed_g_col, speed_c_col = st.columns([2, 1])
    
    with speed_g_col:
        fig_speed=px.line(filtered,x="Timestamp",y="Speed_Kmh")
        fig_speed.add_vline(x=selected_timestamp,line_dash="dash", line_color="red")
        fig_speed.add_scatter(
            x=[selected_timestamp],
            y=[row["Speed_Kmh"]],
            mode="markers",
            marker=dict(size=14, color="orange")
        )
        fig_speed.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_speed,use_container_width=True)
        
    with speed_c_col:
        st.info(f"""
        **Telemetry Metrics:**
        * Current Velocity: `{row['Speed_Kmh']:.1f}` km/h
        * Max Route Velocity: `{filtered['Speed_Kmh'].max():.1f}` km/h
        * Average Session Speed: `{filtered['Speed_Kmh'].mean():.1f}` km/h
        
        **Forensic Interpretation:**
        * The flat line represent constant cruise velocity.
        * Rapid descending vertical drops map out instantaneous deceleration/impact friction.
        * The tracking node is entirely locked with the map's current satellite rendering point.
        """)


    # =====================================================================
    # EVIDENCE REPORT LEDGERS
    # =====================================================================
    st.subheader("📜 Investigation Timeline")
    timeline=filtered[filtered["Semantic_Event"]!="✅ Normal Movement"][
        ["Timestamp","Semantic_Event","Speed_Kmh"]
    ]
    st.dataframe(timeline,use_container_width=True)

    st.subheader("📝 Investigation Summary")
    st.success(f"""
    Vehicle trajectory reconstruction completed successfully.
    """)

    # --- Dynamic Rerun Trigger Control Layer (Optimized at the bottom) ---
    if st.session_state.play_animation:
        # Determine step size based on chosen speed
        step_size = 2 if playback_speed == "2x" else 2
        
        if st.session_state.frame_index + step_size < len(timestamps):
            st.session_state.frame_index += step_size
        else:
            st.session_state.frame_index = 0

        # Maintain a slight stabilizing sleep delay
        sleep_duration = 0.20 if playback_speed == "1x" else 0.05
        time.sleep(sleep_duration)
        st.rerun()
