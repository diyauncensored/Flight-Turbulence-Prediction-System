import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.airport_data import INDIAN_AIRPORTS, INDIAN_ROUTES, calculate_great_circle_distance
from utils.weather_api import weather_api
from utils.ml_models import turbulence_model
from utils.turbulence_calculator import turbulence_calc
from datetime import datetime, timedelta

st.set_page_config(page_title="Route Assessment", page_icon="✈️", layout="wide")

st.title("✈️ Flight Route Turbulence Risk Assessment")
st.markdown("Comprehensive route analysis for turbulence risk evaluation")

# Sidebar controls
st.sidebar.header("Route Configuration")

# Route selection method
route_method = st.sidebar.radio(
    "Route Selection Method",
    ["Predefined Routes", "Custom Route"]
)

origin_airport = None
destination_airport = None

if route_method == "Predefined Routes":
    # Select from predefined routes
    route_options = []
    for route in INDIAN_ROUTES:
        origin_name = INDIAN_AIRPORTS[route['origin']]['city']
        dest_name = INDIAN_AIRPORTS[route['destination']]['city']
        route_label = f"{route['origin']} ({origin_name}) → {route['destination']} ({dest_name})"
        route_options.append((route_label, route))
    
    selected_route_label = st.sidebar.selectbox(
        "Select Route",
        options=[label for label, _ in route_options]
    )
    
    # Find selected route
    selected_route = None
    for label, route in route_options:
        if label == selected_route_label:
            selected_route = route
            break
    
    if selected_route:
        origin_airport = selected_route['origin']
        destination_airport = selected_route['destination']
        predefined_distance = selected_route['distance']
        typical_altitude = selected_route['typical_altitude']

else:
    # Custom route selection
    origin_airport = st.sidebar.selectbox(
        "Origin Airport",
        options=list(INDIAN_AIRPORTS.keys()),
        format_func=lambda x: f"{x} - {INDIAN_AIRPORTS[x]['city']}"
    )
    
    destination_airport = st.sidebar.selectbox(
        "Destination Airport",
        options=list(INDIAN_AIRPORTS.keys()),
        format_func=lambda x: f"{x} - {INDIAN_AIRPORTS[x]['city']}"
    )
    
    if origin_airport == destination_airport:
        st.sidebar.error("Origin and destination cannot be the same!")
        st.stop()
    
    # Calculate distance
    origin_info = INDIAN_AIRPORTS[origin_airport]
    dest_info = INDIAN_AIRPORTS[destination_airport]
    
    predefined_distance = calculate_great_circle_distance(
        origin_info['lat'], origin_info['lon'],
        dest_info['lat'], dest_info['lon']
    )
    
    typical_altitude = 37000  # Default cruise altitude

# Flight parameters
st.sidebar.subheader("Flight Parameters")

flight_altitude = st.sidebar.slider(
    "Cruise Altitude (ft)",
    min_value=25000,
    max_value=45000,
    value=typical_altitude if route_method == "Predefined Routes" else 37000,
    step=1000
)

aircraft_category = st.sidebar.selectbox(
    "Aircraft Category",
    ["Light", "Medium", "Heavy", "Super Heavy"],
    index=2
)

flight_duration = st.sidebar.number_input(
    "Estimated Flight Duration (hours)",
    min_value=0.5,
    max_value=6.0,
    value=predefined_distance / 850 if predefined_distance else 2.0,  # ~850 km/h average
    step=0.1
)

# Analysis parameters
analysis_segments = st.sidebar.slider(
    "Route Analysis Segments",
    min_value=3,
    max_value=20,
    value=10,
    help="Number of waypoints to analyze along the route"
)

# Main content
if origin_airport and destination_airport:
    
    # Route information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🗺️ Route Visualization")
        
        # Get airport information
        origin_info = INDIAN_AIRPORTS[origin_airport]
        dest_info = INDIAN_AIRPORTS[destination_airport]
        
        # Create route map
        fig = go.Figure()
        
        # Add airports
        airports_data = [
            {"lat": origin_info['lat'], "lon": origin_info['lon'], 
             "name": f"{origin_airport} - {origin_info['city']}", "type": "Origin"},
            {"lat": dest_info['lat'], "lon": dest_info['lon'], 
             "name": f"{destination_airport} - {dest_info['city']}", "type": "Destination"}
        ]
        
        for airport in airports_data:
            color = "green" if airport["type"] == "Origin" else "red"
            fig.add_trace(go.Scattermapbox(
                lat=[airport["lat"]],
                lon=[airport["lon"]],
                mode='markers',
                marker=dict(size=15, color=color),
                text=airport["name"],
                hovertemplate=f"<b>{airport['name']}</b><br>Type: {airport['type']}<extra></extra>",
                name=airport["type"]
            ))
        
        # Add route line
        fig.add_trace(go.Scattermapbox(
            lat=[origin_info['lat'], dest_info['lat']],
            lon=[origin_info['lon'], dest_info['lon']],
            mode='lines',
            line=dict(width=3, color='blue'),
            name="Flight Route",
            hovertemplate="Flight Route<extra></extra>"
        ))
        
        # Calculate center point for map
        center_lat = (origin_info['lat'] + dest_info['lat']) / 2
        center_lon = (origin_info['lon'] + dest_info['lon']) / 2
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=5
            ),
            height=500,
            title=f"Flight Route: {origin_airport} → {destination_airport}",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("## 📊 Route Summary")
        
        # Route metrics
        st.metric("Distance", f"{predefined_distance:.0f} km")
        st.metric("Flight Duration", f"{flight_duration:.1f} hours")
        st.metric("Cruise Altitude", f"{flight_altitude:,} ft")
        st.metric("Aircraft Category", aircraft_category)
        
        # Airport details
        st.markdown("### Origin Airport")
        st.text(f"Code: {origin_airport}")
        st.text(f"Name: {origin_info['name']}")
        st.text(f"City: {origin_info['city']}")
        st.text(f"Elevation: {origin_info['elevation']} ft")
        
        st.markdown("### Destination Airport")
        st.text(f"Code: {destination_airport}")
        st.text(f"Name: {dest_info['name']}")
        st.text(f"City: {dest_info['city']}")
        st.text(f"Elevation: {dest_info['elevation']} ft")
    
    # Route analysis
    if st.button("🔍 Analyze Route Turbulence Risk", type="primary"):
        with st.spinner("Analyzing turbulence risk along the route..."):
            
            # Create waypoints along the route
            waypoints = []
            
            for i in range(analysis_segments + 1):
                fraction = i / analysis_segments
                
                # Linear interpolation between origin and destination
                lat = origin_info['lat'] + fraction * (dest_info['lat'] - origin_info['lat'])
                lon = origin_info['lon'] + fraction * (dest_info['lon'] - dest_info['lon'])
                
                # Calculate distance from origin
                segment_distance = fraction * predefined_distance
                
                waypoints.append({
                    'segment': i,
                    'lat': lat,
                    'lon': lon,
                    'distance_from_origin': segment_distance,
                    'estimated_time': fraction * flight_duration
                })
            
            # Analyze turbulence at each waypoint
            route_analysis = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, waypoint in enumerate(waypoints):
                status_text.text(f"Analyzing waypoint {i+1}/{len(waypoints)}...")
                progress_bar.progress((i + 1) / len(waypoints))
                
                # Get weather data for waypoint
                weather_data = weather_api.get_current_weather(waypoint['lat'], waypoint['lon'])
                
                if weather_data:
                    # Flight parameters for this segment
                    flight_params = {
                        'altitude': flight_altitude,
                        'aircraft_type': aircraft_category
                    }
                    
                    # Predict turbulence
                    predictions, confidence = turbulence_model.predict_turbulence(
                        weather_data, flight_params
                    )
                    
                    turbulence_intensity = predictions.get('ensemble', 0)
                    risk_level, risk_color = turbulence_model.get_turbulence_risk_level(turbulence_intensity)
                    
                    # Additional calculations
                    comfort_index = turbulence_calc.calculate_passenger_comfort_index(
                        turbulence_intensity, 30  # 30 minutes per segment
                    )
                    
                    route_analysis.append({
                        'segment': waypoint['segment'],
                        'lat': waypoint['lat'],
                        'lon': waypoint['lon'],
                        'distance_km': waypoint['distance_from_origin'],
                        'estimated_time_hours': waypoint['estimated_time'],
                        'temperature': weather_data['temperature'],
                        'wind_speed': weather_data['wind_speed'],
                        'wind_direction': weather_data['wind_direction'],
                        'pressure': weather_data['pressure'],
                        'humidity': weather_data['humidity'],
                        'weather_condition': weather_data['weather_condition'],
                        'turbulence_intensity': turbulence_intensity,
                        'risk_level': risk_level,
                        'risk_color': risk_color,
                        'confidence': confidence.get('ensemble', 0),
                        'comfort_index': comfort_index
                    })
                else:
                    # Fallback data when weather API fails
                    route_analysis.append({
                        'segment': waypoint['segment'],
                        'lat': waypoint['lat'],
                        'lon': waypoint['lon'],
                        'distance_km': waypoint['distance_from_origin'],
                        'estimated_time_hours': waypoint['estimated_time'],
                        'temperature': 'N/A',
                        'wind_speed': 'N/A',
                        'wind_direction': 'N/A',
                        'pressure': 'N/A',
                        'humidity': 'N/A',
                        'weather_condition': 'Unknown',
                        'turbulence_intensity': 0,
                        'risk_level': 'Unknown',
                        'risk_color': 'gray',
                        'confidence': 0,
                        'comfort_index': 100
                    })
            
            progress_bar.empty()
            status_text.empty()
            
            # Store analysis results
            st.session_state.route_analysis = pd.DataFrame(route_analysis)
    
    # Display analysis results
    if 'route_analysis' in st.session_state:
        analysis_df = st.session_state.route_analysis
        
        if not analysis_df.empty:
            st.markdown("## 📈 Route Analysis Results")
            
            # Overall route risk assessment
            avg_turbulence = analysis_df['turbulence_intensity'].mean()
            max_turbulence = analysis_df['turbulence_intensity'].max()
            avg_comfort = analysis_df['comfort_index'].mean()
            
            overall_risk_level, overall_color = turbulence_model.get_turbulence_risk_level(avg_turbulence)
            
            # Summary metrics
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Average Risk Level", overall_risk_level)
            
            with summary_col2:
                st.metric("Average Turbulence", f"{avg_turbulence:.2f}")
            
            with summary_col3:
                st.metric("Maximum Turbulence", f"{max_turbulence:.2f}")
            
            with summary_col4:
                st.metric("Comfort Index", f"{avg_comfort:.1f}/100")
            
            # Route risk visualization
            st.markdown("### 🎯 Turbulence Risk Along Route")
            
            # Create turbulence intensity chart
            fig = go.Figure()
            
            # Add turbulence intensity line
            fig.add_trace(go.Scatter(
                x=analysis_df['distance_km'],
                y=analysis_df['turbulence_intensity'],
                mode='lines+markers',
                name='Turbulence Intensity',
                line=dict(width=3),
                marker=dict(
                    size=8,
                    color=analysis_df['turbulence_intensity'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Turbulence Intensity")
                ),
                hovertemplate='Distance: %{x:.0f} km<br>Turbulence: %{y:.2f}<extra></extra>'
            ))
            
            # Add risk threshold lines
            fig.add_hline(y=1.0, line_dash="dash", line_color="yellow", 
                         annotation_text="Moderate Risk Threshold")
            fig.add_hline(y=2.5, line_dash="dash", line_color="orange", 
                         annotation_text="High Risk Threshold")
            fig.add_hline(y=4.0, line_dash="dash", line_color="red", 
                         annotation_text="Severe Risk Threshold")
            
            fig.update_layout(
                title="Turbulence Intensity Along Flight Route",
                xaxis_title="Distance from Origin (km)",
                yaxis_title="Turbulence Intensity",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weather parameters along route
            st.markdown("### 🌤️ Weather Conditions Along Route")
            
            # Create subplots for weather parameters
            from plotly.subplots import make_subplots
            
            weather_fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Temperature", "Wind Speed", "Pressure", "Humidity"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Temperature
            weather_fig.add_trace(
                go.Scatter(x=analysis_df['distance_km'], y=analysis_df['temperature'],
                          mode='lines', name='Temperature', line=dict(color='red')),
                row=1, col=1
            )
            
            # Wind Speed
            weather_fig.add_trace(
                go.Scatter(x=analysis_df['distance_km'], y=analysis_df['wind_speed'],
                          mode='lines', name='Wind Speed', line=dict(color='blue')),
                row=1, col=2
            )
            
            # Pressure
            weather_fig.add_trace(
                go.Scatter(x=analysis_df['distance_km'], y=analysis_df['pressure'],
                          mode='lines', name='Pressure', line=dict(color='green')),
                row=2, col=1
            )
            
            # Humidity
            weather_fig.add_trace(
                go.Scatter(x=analysis_df['distance_km'], y=analysis_df['humidity'],
                          mode='lines', name='Humidity', line=dict(color='purple')),
                row=2, col=2
            )
            
            weather_fig.update_layout(height=500, showlegend=False)
            weather_fig.update_xaxes(title_text="Distance (km)")
            
            st.plotly_chart(weather_fig, use_container_width=True)
            
            # Route segments table
            st.markdown("### 📋 Detailed Segment Analysis")
            
            # Select columns for display
            display_columns = ['segment', 'distance_km', 'estimated_time_hours', 
                             'temperature', 'wind_speed', 'pressure', 'humidity',
                             'turbulence_intensity', 'risk_level', 'confidence', 'comfort_index']
            
            display_df = analysis_df[display_columns].copy()
            
            # Format columns
            display_df['distance_km'] = display_df['distance_km'].round(0)
            display_df['estimated_time_hours'] = display_df['estimated_time_hours'].round(2)
            display_df['turbulence_intensity'] = display_df['turbulence_intensity'].round(3)
            display_df['confidence'] = display_df['confidence'].round(1)
            display_df['comfort_index'] = display_df['comfort_index'].round(1)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "segment": "Segment",
                    "distance_km": "Distance (km)",
                    "estimated_time_hours": "Time (hrs)",
                    "temperature": "Temp (°C)",
                    "wind_speed": "Wind (m/s)",
                    "pressure": "Pressure (hPa)",
                    "humidity": "Humidity (%)",
                    "turbulence_intensity": "Turbulence",
                    "risk_level": "Risk Level",
                    "confidence": "Confidence (%)",
                    "comfort_index": "Comfort"
                }
            )
            
            # Risk hotspots identification
            st.markdown("### ⚠️ Risk Hotspots")
            
            high_risk_segments = analysis_df[analysis_df['turbulence_intensity'] >= 2.5]
            
            if not high_risk_segments.empty:
                st.warning(f"Found {len(high_risk_segments)} high-risk segments:")
                
                for _, segment in high_risk_segments.iterrows():
                    st.error(f"Segment {segment['segment']}: {segment['distance_km']:.0f}km from origin - "
                           f"Risk Level: {segment['risk_level']} (Intensity: {segment['turbulence_intensity']:.2f})")
            else:
                st.success("✅ No high-risk segments detected along the route")
            
            # Route recommendations
            st.markdown("### 💡 Route Recommendations")
            
            if avg_turbulence < 1.0:
                st.success("✅ **Excellent Route**: Low turbulence expected throughout the journey")
            elif avg_turbulence < 2.5:
                st.info("⚠️ **Moderate Route**: Some turbulence expected, standard precautions advised")
            elif avg_turbulence < 4.0:
                st.warning("🔶 **Challenging Route**: Significant turbulence expected, consider alternative timing")
            else:
                st.error("🔴 **High Risk Route**: Severe turbulence possible, consider route change or delay")
            
            # Additional recommendations
            recommendations = []
            
            if max_turbulence >= 4.0:
                recommendations.append("Consider alternative routing to avoid severe turbulence areas")
            
            if analysis_df['wind_speed'].max() > 25:
                recommendations.append("Strong winds detected - increased fuel consumption expected")
            
            if (analysis_df['weather_condition'] == 'thunderstorm').any():
                recommendations.append("Thunderstorms along route - monitor weather updates closely")
            
            if avg_comfort < 60:
                recommendations.append("Low passenger comfort expected - inform passengers and crew")
            
            if recommendations:
                for rec in recommendations:
                    st.info(f"💡 {rec}")
    
    # Route comparison tool
    with st.expander("🔄 Compare Alternative Routes"):
        st.markdown("### Route Alternatives")
        st.info("This feature would compare different routing options based on current weather conditions and historical turbulence data.")
        
        # Placeholder for alternative route suggestions
        if origin_airport and destination_airport:
            st.markdown(f"**Current Route**: Direct path from {origin_airport} to {destination_airport}")
            st.markdown("**Alternative Routes** (Future Enhancement):")
            st.markdown("- Northern route via waypoints")
            st.markdown("- Southern route via waypoints") 
            st.markdown("- High altitude routing")
            st.markdown("- Weather avoidance routing")

else:
    st.info("Please select origin and destination airports to begin route analysis.")
