import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from utils.airport_data import INDIAN_AIRPORTS
from utils.weather_api import weather_api
from utils.ml_models import turbulence_model
from utils.turbulence_calculator import turbulence_calc
from utils.data_processing import data_processor

st.set_page_config(page_title="Real-Time Prediction", page_icon="⚡", layout="wide")

st.title("⚡ Real-Time Turbulence Prediction System")
st.markdown("Live turbulence monitoring and prediction with confidence scoring")

# Initialize session state
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = []
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Sidebar controls
st.sidebar.header("Real-Time Configuration")

# Auto-refresh settings
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
refresh_interval = st.sidebar.selectbox(
    "Refresh Interval",
    [30, 60, 120, 300],
    index=1,
    format_func=lambda x: f"{x} seconds"
)

# Airport monitoring selection
monitored_airports = st.sidebar.multiselect(
    "Airports to Monitor",
    options=list(INDIAN_AIRPORTS.keys()),
    default=["DEL", "BOM", "BLR"],
    format_func=lambda x: f"{x} - {INDIAN_AIRPORTS[x]['city']}"
)

# Alert thresholds
st.sidebar.subheader("Alert Thresholds")
moderate_threshold = st.sidebar.slider("Moderate Risk Threshold", 0.5, 2.0, 1.0, 0.1)
high_threshold = st.sidebar.slider("High Risk Threshold", 1.5, 3.5, 2.5, 0.1)
severe_threshold = st.sidebar.slider("Severe Risk Threshold", 2.5, 5.0, 4.0, 0.1)

# Display preferences
show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)
show_weather_details = st.sidebar.checkbox("Show Weather Details", value=True)
max_log_entries = st.sidebar.number_input("Max Log Entries", 10, 1000, 100)

if not monitored_airports:
    st.warning("Please select at least one airport to monitor.")
    st.stop()

# Main dashboard layout
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

with header_col1:
    st.markdown("## 🎯 Current Turbulence Status")

with header_col2:
    last_update = st.empty()

with header_col3:
    refresh_button = st.button("🔄 Refresh Now", type="primary")

# Real-time data fetching function
def fetch_real_time_data():
    """Fetch current turbulence predictions for all monitored airports"""
    current_data = []
    current_time = datetime.now()
    
    for airport_code in monitored_airports:
        airport_info = INDIAN_AIRPORTS[airport_code]
        
        # Get current weather
        weather_data = weather_api.get_current_weather(
            airport_info['lat'], 
            airport_info['lon']
        )
        
        if weather_data:
            # Standard flight parameters
            flight_params = {'altitude': 35000, 'aircraft_type': 'Heavy'}
            
            # Get turbulence prediction
            predictions, confidence = turbulence_model.predict_turbulence(
                weather_data, flight_params
            )
            
            turbulence_intensity = predictions.get('ensemble', 0)
            risk_level, risk_color = turbulence_model.get_turbulence_risk_level(turbulence_intensity)
            
            # Calculate additional metrics
            comfort_index = turbulence_calc.calculate_passenger_comfort_index(
                turbulence_intensity, 60
            )
            delay_probability = turbulence_calc.calculate_flight_delay_probability(
                turbulence_intensity, weather_data.get('weather_condition', 'clear')
            )
            
            airport_data = {
                'timestamp': current_time,
                'airport': airport_code,
                'airport_name': airport_info['name'],
                'city': airport_info['city'],
                'turbulence_intensity': turbulence_intensity,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'confidence': confidence.get('ensemble', 0),
                'temperature': weather_data['temperature'],
                'wind_speed': weather_data['wind_speed'],
                'wind_direction': weather_data['wind_direction'],
                'pressure': weather_data['pressure'],
                'humidity': weather_data['humidity'],
                'weather_condition': weather_data['weather_condition'],
                'description': weather_data['description'],
                'comfort_index': comfort_index,
                'delay_probability': delay_probability
            }
            
            current_data.append(airport_data)
            
            # Check for alerts
            check_alerts(airport_data)
    
    return current_data

def check_alerts(airport_data):
    """Check for turbulence alerts and add to alerts list"""
    intensity = airport_data['turbulence_intensity']
    airport = airport_data['airport']
    timestamp = airport_data['timestamp']
    
    alert_message = None
    alert_level = None
    
    if intensity >= severe_threshold:
        alert_message = f"🔴 SEVERE TURBULENCE ALERT at {airport}: {intensity:.2f}"
        alert_level = "severe"
    elif intensity >= high_threshold:
        alert_message = f"🟠 HIGH TURBULENCE ALERT at {airport}: {intensity:.2f}"
        alert_level = "high"
    elif intensity >= moderate_threshold:
        alert_message = f"🟡 MODERATE TURBULENCE ALERT at {airport}: {intensity:.2f}"
        alert_level = "moderate"
    
    if alert_message:
        alert_entry = {
            'timestamp': timestamp,
            'airport': airport,
            'message': alert_message,
            'level': alert_level,
            'intensity': intensity
        }
        
        # Add to alerts list (avoid duplicates within 5 minutes)
        recent_alerts = [a for a in st.session_state.alerts 
                        if a['airport'] == airport and 
                        (timestamp - a['timestamp']).total_seconds() < 300]
        
        if not recent_alerts:
            st.session_state.alerts.append(alert_entry)
            # Keep only last 50 alerts
            st.session_state.alerts = st.session_state.alerts[-50:]

# Fetch data on button click or auto-refresh
if refresh_button or (auto_refresh and 'last_refresh' not in st.session_state) or \
   (auto_refresh and 'last_refresh' in st.session_state and 
    (datetime.now() - st.session_state.last_refresh).total_seconds() > refresh_interval):
    
    with st.spinner("Fetching real-time turbulence data..."):
        current_predictions = fetch_real_time_data()
        
        if current_predictions:
            st.session_state.real_time_data = current_predictions
            st.session_state.last_refresh = datetime.now()
            
            # Add to prediction log
            for prediction in current_predictions:
                st.session_state.prediction_log.append(prediction.copy())
            
            # Keep only recent log entries
            st.session_state.prediction_log = st.session_state.prediction_log[-max_log_entries:]

# Update last update time
if 'last_refresh' in st.session_state:
    last_update.text(f"Last Update: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# Display current predictions
if st.session_state.real_time_data:
    
    # Current status cards
    st.markdown("### 📊 Current Airport Status")
    
    cols = st.columns(len(monitored_airports))
    
    for i, airport_data in enumerate(st.session_state.real_time_data):
        with cols[i]:
            # Status card
            intensity = airport_data['turbulence_intensity']
            risk_level = airport_data['risk_level']
            confidence = airport_data['confidence']
            
            # Color coding
            if risk_level == "Severe":
                status_color = "🔴"
            elif risk_level == "High":
                status_color = "🟠"
            elif risk_level == "Moderate":
                status_color = "🟡"
            else:
                status_color = "🟢"
            
            st.markdown(f"""
            <div style="
                padding: 15px; 
                border: 2px solid {airport_data['risk_color']}; 
                border-radius: 10px; 
                margin: 5px 0;
            ">
                <h4>{status_color} {airport_data['airport']} - {airport_data['city']}</h4>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <p><strong>Intensity:</strong> {intensity:.3f}</p>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            with st.expander("Detailed Metrics"):
                st.metric("Comfort Index", f"{airport_data['comfort_index']:.1f}/100")
                st.metric("Delay Probability", f"{airport_data['delay_probability']:.1f}%")
                
                if show_weather_details:
                    st.text(f"Temperature: {airport_data['temperature']:.1f}°C")
                    st.text(f"Wind: {airport_data['wind_speed']:.1f} m/s @ {airport_data['wind_direction']:.0f}°")
                    st.text(f"Pressure: {airport_data['pressure']:.0f} hPa")
                    st.text(f"Weather: {airport_data['weather_condition']}")
    
    # Real-time trends
    if len(st.session_state.prediction_log) > 1:
        st.markdown("## 📈 Real-Time Trends")
        
        # Convert log to DataFrame
        log_df = pd.DataFrame(st.session_state.prediction_log)
        
        # Turbulence intensity trends
        fig_trends = px.line(
            log_df.tail(50),  # Last 50 entries
            x='timestamp',
            y='turbulence_intensity',
            color='airport',
            title="Real-Time Turbulence Intensity Trends",
            labels={'timestamp': 'Time', 'turbulence_intensity': 'Turbulence Intensity'}
        )
        
        # Add threshold lines
        fig_trends.add_hline(y=moderate_threshold, line_dash="dash", line_color="yellow", 
                           annotation_text="Moderate Threshold")
        fig_trends.add_hline(y=high_threshold, line_dash="dash", line_color="orange", 
                           annotation_text="High Threshold")
        fig_trends.add_hline(y=severe_threshold, line_dash="dash", line_color="red", 
                           annotation_text="Severe Threshold")
        
        fig_trends.update_layout(height=400)
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Confidence trends
        if show_confidence:
            fig_confidence = px.line(
                log_df.tail(50),
                x='timestamp',
                y='confidence',
                color='airport',
                title="Prediction Confidence Trends",
                labels={'timestamp': 'Time', 'confidence': 'Confidence (%)'}
            )
            
            fig_confidence.update_layout(height=300)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Multi-parameter dashboard
        st.markdown("### 🌤️ Multi-Parameter Real-Time Dashboard")
        
        # Create subplots
        subplot_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Temperature", "Wind Speed", "Pressure", "Turbulence vs Confidence"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Recent data for plotting
        recent_data = log_df.tail(20)
        
        for airport in monitored_airports:
            airport_data = recent_data[recent_data['airport'] == airport]
            
            if not airport_data.empty:
                # Temperature
                subplot_fig.add_trace(
                    go.Scatter(x=airport_data['timestamp'], y=airport_data['temperature'],
                              mode='lines+markers', name=f'{airport} Temp', 
                              line=dict(width=2)), row=1, col=1
                )
                
                # Wind Speed
                subplot_fig.add_trace(
                    go.Scatter(x=airport_data['timestamp'], y=airport_data['wind_speed'],
                              mode='lines+markers', name=f'{airport} Wind',
                              line=dict(width=2)), row=1, col=2
                )
                
                # Pressure
                subplot_fig.add_trace(
                    go.Scatter(x=airport_data['timestamp'], y=airport_data['pressure'],
                              mode='lines+markers', name=f'{airport} Pressure',
                              line=dict(width=2)), row=2, col=1
                )
                
                # Turbulence vs Confidence
                subplot_fig.add_trace(
                    go.Scatter(x=airport_data['timestamp'], y=airport_data['turbulence_intensity'],
                              mode='lines+markers', name=f'{airport} Turbulence',
                              line=dict(width=2)), row=2, col=2
                )
                
                subplot_fig.add_trace(
                    go.Scatter(x=airport_data['timestamp'], y=airport_data['confidence'],
                              mode='lines+markers', name=f'{airport} Confidence',
                              line=dict(width=2, dash='dash'), yaxis='y2'), row=2, col=2
                )
        
        subplot_fig.update_layout(height=600, showlegend=True)
        subplot_fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        subplot_fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=2)
        subplot_fig.update_yaxes(title_text="Pressure (hPa)", row=2, col=1)
        subplot_fig.update_yaxes(title_text="Turbulence Intensity", row=2, col=2)
        subplot_fig.update_yaxes(title_text="Confidence (%)", secondary_y=True, row=2, col=2)
        
        st.plotly_chart(subplot_fig, use_container_width=True)

# Alerts section
st.markdown("## ⚠️ Recent Alerts")

if st.session_state.alerts:
    # Show recent alerts
    recent_alerts = sorted(st.session_state.alerts, key=lambda x: x['timestamp'], reverse=True)[:10]
    
    for alert in recent_alerts:
        alert_time = alert['timestamp'].strftime('%H:%M:%S')
        
        if alert['level'] == 'severe':
            st.error(f"[{alert_time}] {alert['message']}")
        elif alert['level'] == 'high':
            st.warning(f"[{alert_time}] {alert['message']}")
        else:
            st.info(f"[{alert_time}] {alert['message']}")
    
    # Alert statistics
    col1, col2 = st.columns(2)
    
    with col1:
        alert_counts = {}
        for alert in st.session_state.alerts:
            level = alert['level']
            alert_counts[level] = alert_counts.get(level, 0) + 1
        
        if alert_counts:
            fig_alerts = px.pie(
                values=list(alert_counts.values()),
                names=list(alert_counts.keys()),
                title="Alert Distribution",
                color_discrete_map={'severe': 'red', 'high': 'orange', 'moderate': 'yellow'}
            )
            st.plotly_chart(fig_alerts, use_container_width=True)
    
    with col2:
        # Alerts by airport
        airport_alerts = {}
        for alert in st.session_state.alerts:
            airport = alert['airport']
            airport_alerts[airport] = airport_alerts.get(airport, 0) + 1
        
        if airport_alerts:
            fig_airport_alerts = px.bar(
                x=list(airport_alerts.keys()),
                y=list(airport_alerts.values()),
                title="Alerts by Airport",
                labels={'x': 'Airport', 'y': 'Alert Count'}
            )
            st.plotly_chart(fig_airport_alerts, use_container_width=True)
    
    # Clear alerts button
    if st.button("🗑️ Clear All Alerts"):
        st.session_state.alerts = []
        st.rerun()

else:
    st.success("✅ No recent alerts - All airports operating within normal parameters")

# System status
st.markdown("## 🖥️ System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    weather_api_status = "🟢 Online" if st.session_state.real_time_data else "🔴 Offline"
    st.metric("Weather API", weather_api_status)

with status_col2:
    ml_model_status = "🟢 Active" if turbulence_model.is_trained else "🟡 Training"
    st.metric("ML Models", ml_model_status)

with status_col3:
    monitored_count = len(monitored_airports)
    st.metric("Monitored Airports", monitored_count)

with status_col4:
    log_entries = len(st.session_state.prediction_log)
    st.metric("Log Entries", log_entries)

# Performance metrics
if st.session_state.prediction_log:
    st.markdown("### 📊 Performance Metrics")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        avg_confidence = np.mean([entry['confidence'] for entry in st.session_state.prediction_log])
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
    
    with perf_col2:
        predictions_last_hour = len([
            entry for entry in st.session_state.prediction_log 
            if (datetime.now() - entry['timestamp']).total_seconds() < 3600
        ])
        st.metric("Predictions/Hour", predictions_last_hour)
    
    with perf_col3:
        total_alerts = len(st.session_state.alerts)
        st.metric("Total Alerts", total_alerts)

# Data export
st.markdown("## 💾 Export Real-Time Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Current Data") and st.session_state.real_time_data:
        current_df = pd.DataFrame(st.session_state.real_time_data)
        csv_data = current_df.to_csv(index=False)
        
        st.download_button(
            label="Download Current Data CSV",
            data=csv_data,
            file_name=f"realtime_turbulence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Prediction Log") and st.session_state.prediction_log:
        log_df = pd.DataFrame(st.session_state.prediction_log)
        csv_log = log_df.to_csv(index=False)
        
        st.download_button(
            label="Download Log CSV",
            data=csv_log,
            file_name=f"prediction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(1)  # Small delay to prevent too frequent updates
    st.rerun()
