import streamlit as st
from datetime import datetime
import pandas as pd
from utils.alert_system import AlertSystem
from utils.database import TurbulenceDatabase
from utils.airport_data import get_airports

st.set_page_config(page_title="Alert System", page_icon="🚨", layout="wide")

st.title("🚨 Automated Turbulence Alert System")
st.markdown("Real-time monitoring and automated alerts for turbulence conditions")

# Initialize systems
alert_system = AlertSystem()
db = TurbulenceDatabase()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Active Alerts", "Generate Alerts", "Configure Thresholds"])

with tab1:
    st.header("Active Turbulence Alerts")
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("🔄 Refresh Alerts", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Expired", use_container_width=True):
            deactivated = alert_system.deactivate_expired_alerts()
            st.success(f"Deactivated {deactivated} expired alerts")
            st.rerun()
    
    # Get alert summary
    airports = get_airports()
    airport_names = {f"{a['name']} ({a['code']})": a['code'] for a in airports}
    
    filter_airport = st.selectbox(
        "Filter by Airport",
        options=["All Airports"] + list(airport_names.keys()),
        key="alert_filter"
    )
    
    filter_code = airport_names.get(filter_airport) if filter_airport != "All Airports" else None
    summary = alert_system.get_alert_summary(airport_code=filter_code)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Active Alerts", summary['total_alerts'])
    
    with col2:
        st.metric("🔴 Severe", summary['severe_count'])
    
    with col3:
        st.metric("🟠 Moderate", summary['moderate_count'])
    
    with col4:
        st.metric("🟡 Advisory", summary['advisory_count'])
    
    # Display alerts
    if summary['alerts']:
        st.subheader("Alert Details")
        
        for alert in summary['alerts']:
            # Color based on severity
            if alert['type'] == 'SEVERE':
                color = "🔴"
                severity_color = "#ff4444"
            elif alert['type'] == 'MODERATE':
                color = "🟠"
                severity_color = "#ff9933"
            else:
                color = "🟡"
                severity_color = "#ffcc00"
            
            with st.container():
                st.markdown(f"""
                <div style="border-left: 5px solid {severity_color}; padding-left: 15px; margin-bottom: 20px;">
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.write(f"{color} **{alert['type']} ALERT**")
                    st.write(f"Airport: **{alert['airport']}**")
                
                with col2:
                    st.write(f"Severity: **{alert['severity']}**")
                    if alert.get('level'):
                        st.write(f"Level: **{alert['level']:.1f}/10**")
                
                with col3:
                    if alert.get('confidence'):
                        st.write(f"Confidence: **{alert['confidence']*100:.0f}%**")
                    if alert.get('valid_until'):
                        st.write(f"Valid until: **{alert['valid_until'].strftime('%H:%M')}**")
                
                with col4:
                    if st.button("Dismiss", key=f"dismiss_{alert['id']}"):
                        if db.deactivate_alert(alert['id']):
                            st.success("Alert dismissed")
                            st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("✅ No active alerts at this time")
    
    # Airports affected
    if summary['airports_affected']:
        st.subheader("Airports with Active Alerts")
        st.write(", ".join(summary['airports_affected']))

with tab2:
    st.header("Generate New Alerts")
    st.markdown("Scan current conditions and generate alerts based on configured thresholds")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scan_airport = st.selectbox(
            "Select Airport to Scan",
            options=["All Airports"] + list(airport_names.keys()),
            key="scan_airport"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔍 Scan & Generate Alerts", type="primary", use_container_width=True):
            with st.spinner("Scanning conditions and generating alerts..."):
                scan_code = airport_names.get(scan_airport) if scan_airport != "All Airports" else None
                
                new_alerts = alert_system.check_and_create_alerts(airport_code=scan_code)
                
                if new_alerts:
                    st.success(f"✅ Generated {len(new_alerts)} new alert(s)")
                    
                    # Display new alerts
                    for alert in new_alerts:
                        st.info(f"""
                        **{alert['type']} Alert** created for {alert['airport']}
                        - Severity: {alert['severity']}
                        - Turbulence Level: {alert['level']:.1f}/10
                        - Confidence: {alert['confidence']*100:.0f}%
                        """)
                    
                    st.rerun()
                else:
                    st.info("No new alerts generated. Conditions are within normal parameters.")
    
    st.markdown("---")
    
    # Show current conditions
    st.subheader("Current Conditions Overview")
    
    from utils.weather_api import WeatherAPI
    weather_api = WeatherAPI()
    
    condition_data = []
    
    for airport in airports:
        weather = weather_api.get_current_weather(
            airport['coordinates'][0],
            airport['coordinates'][1]
        )
        
        if weather:
            condition_data.append({
                'Airport': airport['code'],
                'Name': airport['name'],
                'Wind Speed': f"{weather['wind_speed']:.1f} m/s",
                'Temperature': f"{weather['temperature']:.1f}°C",
                'Pressure': f"{weather['pressure']:.0f} hPa",
                'Conditions': weather['weather_condition']
            })
    
    if condition_data:
        df = pd.DataFrame(condition_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Configure Alert Thresholds")
    st.markdown("Set the minimum turbulence level and confidence required for each alert type")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔴 Severe Alert")
        severe_level = st.slider(
            "Min Turbulence Level",
            min_value=0.0,
            max_value=10.0,
            value=alert_system.thresholds['severe']['min_level'],
            step=0.5,
            key="severe_level"
        )
        severe_conf = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=alert_system.thresholds['severe']['min_confidence'],
            step=0.05,
            key="severe_conf"
        )
        
        if st.button("Update Severe", key="update_severe"):
            if alert_system.update_alert_thresholds('severe', severe_level, severe_conf):
                st.success("✅ Updated")
    
    with col2:
        st.subheader("🟠 Moderate Alert")
        moderate_level = st.slider(
            "Min Turbulence Level",
            min_value=0.0,
            max_value=10.0,
            value=alert_system.thresholds['moderate']['min_level'],
            step=0.5,
            key="moderate_level"
        )
        moderate_conf = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=alert_system.thresholds['moderate']['min_confidence'],
            step=0.05,
            key="moderate_conf"
        )
        
        if st.button("Update Moderate", key="update_moderate"):
            if alert_system.update_alert_thresholds('moderate', moderate_level, moderate_conf):
                st.success("✅ Updated")
    
    with col3:
        st.subheader("🟡 Advisory Alert")
        light_level = st.slider(
            "Min Turbulence Level",
            min_value=0.0,
            max_value=10.0,
            value=alert_system.thresholds['light']['min_level'],
            step=0.5,
            key="light_level"
        )
        light_conf = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=alert_system.thresholds['light']['min_confidence'],
            step=0.05,
            key="light_conf"
        )
        
        if st.button("Update Advisory", key="update_light"):
            if alert_system.update_alert_thresholds('light', light_level, light_conf):
                st.success("✅ Updated")
    
    st.markdown("---")
    st.subheader("Current Threshold Settings")
    
    threshold_df = pd.DataFrame([
        {
            'Alert Type': '🔴 Severe',
            'Min Level': alert_system.thresholds['severe']['min_level'],
            'Min Confidence': f"{alert_system.thresholds['severe']['min_confidence']*100:.0f}%"
        },
        {
            'Alert Type': '🟠 Moderate',
            'Min Level': alert_system.thresholds['moderate']['min_level'],
            'Min Confidence': f"{alert_system.thresholds['moderate']['min_confidence']*100:.0f}%"
        },
        {
            'Alert Type': '🟡 Advisory',
            'Min Level': alert_system.thresholds['light']['min_level'],
            'Min Confidence': f"{alert_system.thresholds['light']['min_confidence']*100:.0f}%"
        }
    ])
    
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Alert system automatically monitors conditions and notifies when turbulence thresholds are exceeded")
