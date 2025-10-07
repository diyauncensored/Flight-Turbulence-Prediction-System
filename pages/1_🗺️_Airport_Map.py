import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.airport_data import INDIAN_AIRPORTS
from utils.weather_api import weather_api
from utils.ml_models import turbulence_model
from utils.turbulence_calculator import turbulence_calc

st.set_page_config(page_title="Airport Map", page_icon="🗺️", layout="wide")

st.title("🗺️ Interactive Airport Map - Turbulence Risk Visualization")
st.markdown("Real-time turbulence conditions across major Indian airports")

# Control panel
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Controls")
    auto_refresh = st.checkbox("Auto Refresh", value=False)
    show_weather_details = st.checkbox("Show Weather Details", value=True)
    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "Low", "Moderate", "High", "Severe"]
    )

with col1:
    # Fetch current weather and turbulence data for all airports
    if 'airport_data' not in st.session_state or st.button("Refresh Data") or auto_refresh:
        st.info("Fetching current weather data for all airports...")
        
        airport_status_data = []
        
        for airport_code, airport_info in INDIAN_AIRPORTS.items():
            # Get current weather
            weather_data = weather_api.get_current_weather(
                airport_info['lat'], 
                airport_info['lon']
            )
            
            if weather_data:
                # Predict turbulence
                flight_params = {'altitude': 35000}  # Standard cruise altitude
                predictions, confidence = turbulence_model.predict_turbulence(
                    weather_data, flight_params
                )
                
                turbulence_intensity = predictions.get('ensemble', 0)
                risk_level, color = turbulence_model.get_turbulence_risk_level(turbulence_intensity)
                
                airport_status = {
                    'airport_code': airport_code,
                    'name': airport_info['name'],
                    'city': airport_info['city'],
                    'lat': airport_info['lat'],
                    'lon': airport_info['lon'],
                    'elevation': airport_info['elevation'],
                    'temperature': weather_data['temperature'],
                    'wind_speed': weather_data['wind_speed'],
                    'wind_direction': weather_data['wind_direction'],
                    'pressure': weather_data['pressure'],
                    'humidity': weather_data['humidity'],
                    'weather_condition': weather_data['weather_condition'],
                    'description': weather_data['description'],
                    'turbulence_intensity': turbulence_intensity,
                    'risk_level': risk_level,
                    'color': color,
                    'confidence': confidence.get('ensemble', 0)
                }
                
                airport_status_data.append(airport_status)
            else:
                # Fallback data when API fails
                airport_status = {
                    'airport_code': airport_code,
                    'name': airport_info['name'],
                    'city': airport_info['city'],
                    'lat': airport_info['lat'],
                    'lon': airport_info['lon'],
                    'elevation': airport_info['elevation'],
                    'temperature': 'N/A',
                    'wind_speed': 'N/A',
                    'wind_direction': 'N/A',
                    'pressure': 'N/A',
                    'humidity': 'N/A',
                    'weather_condition': 'Unknown',
                    'description': 'Data unavailable',
                    'turbulence_intensity': 0,
                    'risk_level': 'Unknown',
                    'color': 'gray',
                    'confidence': 0
                }
                airport_status_data.append(airport_status)
        
        st.session_state.airport_data = pd.DataFrame(airport_status_data)
        st.success("Data updated successfully!")

    # Filter data based on risk level
    display_data = st.session_state.get('airport_data', pd.DataFrame())
    
    if not display_data.empty and risk_filter != "All":
        display_data = display_data[display_data['risk_level'] == risk_filter]
    
    if not display_data.empty:
        # Create the map
        fig = go.Figure()
        
        # Add airport markers
        for _, airport in display_data.iterrows():
            # Create hover text
            hover_text = f"""
            <b>{airport['name']}</b><br>
            City: {airport['city']}<br>
            Risk Level: <b>{airport['risk_level']}</b><br>
            Turbulence Intensity: {airport['turbulence_intensity']:.2f}<br>
            Confidence: {airport['confidence']:.1f}%<br>
            Temperature: {airport['temperature']}°C<br>
            Wind: {airport['wind_speed']} m/s @ {airport['wind_direction']}°<br>
            Pressure: {airport['pressure']} hPa<br>
            Weather: {airport['weather_condition']}
            """
            
            # Determine marker size based on turbulence intensity
            marker_size = max(15, min(40, airport['turbulence_intensity'] * 8 + 15))
            
            fig.add_trace(go.Scattermapbox(
                lat=[airport['lat']],
                lon=[airport['lon']],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=airport['color'],
                    opacity=0.8,
                    sizemode='diameter'
                ),
                text=airport['airport_code'],
                textfont=dict(size=12, color='white'),
                textposition='middle center',
                hovertemplate=hover_text + '<extra></extra>',
                name=airport['airport_code']
            ))
        
        # Update map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=20.5937, lon=78.9629),  # Center of India
                zoom=4
            ),
            height=600,
            title="Indian Airports - Current Turbulence Risk Levels",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level summary
        st.markdown("### Risk Level Summary")
        
        if not display_data.empty:
            risk_summary = display_data['risk_level'].value_counts()
            
            summary_cols = st.columns(len(risk_summary))
            
            color_map = {'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Severe': 'red'}
            
            for i, (risk_level, count) in enumerate(risk_summary.items()):
                with summary_cols[i]:
                    st.metric(
                        f"{risk_level} Risk",
                        count,
                        delta=None
                    )
        
        # Detailed airport information
        if show_weather_details:
            st.markdown("### Detailed Airport Information")
            
            # Create expandable sections for each airport
            for _, airport in display_data.iterrows():
                with st.expander(f"{airport['airport_code']} - {airport['name']} ({airport['risk_level']} Risk)"):
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.markdown("**Location & Basic Info**")
                        st.write(f"City: {airport['city']}")
                        st.write(f"Elevation: {airport['elevation']} ft")
                        st.write(f"Coordinates: {airport['lat']:.3f}, {airport['lon']:.3f}")
                    
                    with info_col2:
                        st.markdown("**Current Weather**")
                        st.write(f"Temperature: {airport['temperature']}°C")
                        st.write(f"Pressure: {airport['pressure']} hPa")
                        st.write(f"Humidity: {airport['humidity']}%")
                        st.write(f"Condition: {airport['weather_condition']}")
                    
                    with info_col3:
                        st.markdown("**Wind & Turbulence**")
                        st.write(f"Wind Speed: {airport['wind_speed']} m/s")
                        st.write(f"Wind Direction: {airport['wind_direction']}°")
                        st.write(f"Turbulence Intensity: {airport['turbulence_intensity']:.2f}")
                        st.write(f"Confidence: {airport['confidence']:.1f}%")
    else:
        if display_data.empty:
            st.error("No airport data available. Please check the weather API connection.")
        else:
            st.info(f"No airports found with {risk_filter} risk level.")

# Legend
st.markdown("### Legend")
legend_col1, legend_col2 = st.columns(2)

with legend_col1:
    st.markdown("""
    **Risk Levels:**
    - 🟢 **Low**: Minimal turbulence expected
    - 🟡 **Moderate**: Light to moderate turbulence possible
    - 🟠 **High**: Moderate to severe turbulence likely
    - 🔴 **Severe**: Severe turbulence expected
    """)

with legend_col2:
    st.markdown("""
    **Marker Size**: Proportional to turbulence intensity
    
    **Weather Data**: Real-time from OpenWeatherMap API
    
    **Predictions**: ML-based turbulence forecasting
    """)

# Auto refresh
if auto_refresh:
    st.rerun()
