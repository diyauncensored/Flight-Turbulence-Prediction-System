import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.airport_data import INDIAN_AIRPORTS
from utils.weather_api import weather_api
from utils.turbulence_calculator import turbulence_calc

st.set_page_config(page_title="Atmospheric Visualization", page_icon="🌬️", layout="wide")

st.title("🌬️ Atmospheric Data Visualization")
st.markdown("Wind patterns, pressure systems, and atmospheric analysis")

# Sidebar controls
st.sidebar.header("Visualization Parameters")

# Visualization type
viz_type = st.sidebar.selectbox(
    "Visualization Type",
    ["Wind Patterns", "Pressure Systems", "Atmospheric Stability", "3D Atmospheric Model", "Regional Weather Map"]
)

# Airport/Region selection
if viz_type == "Regional Weather Map":
    st.sidebar.subheader("Regional Analysis")
    region_center = st.sidebar.selectbox(
        "Focus Region",
        ["All India", "North India", "South India", "West India", "East India"],
        help="Select geographical region for analysis"
    )
else:
    selected_airport = st.sidebar.selectbox(
        "Select Airport",
        options=list(INDIAN_AIRPORTS.keys()),
        format_func=lambda x: f"{x} - {INDIAN_AIRPORTS[x]['city']}"
    )

# Altitude levels for analysis
altitude_levels = st.sidebar.multiselect(
    "Altitude Levels (ft)",
    options=[20000, 25000, 30000, 35000, 40000, 45000],
    default=[30000, 35000, 40000],
    help="Select flight altitudes for atmospheric analysis"
)

# Time parameters
time_range = st.sidebar.selectbox(
    "Time Analysis",
    ["Current", "Next 12 Hours", "Next 24 Hours", "Next 48 Hours"]
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    show_turbulence_overlay = st.checkbox("Show Turbulence Overlay", value=True)
    show_wind_vectors = st.checkbox("Show Wind Vectors", value=True)
    animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.0, 0.1)
    grid_resolution = st.selectbox("Grid Resolution", ["Low", "Medium", "High"], index=1)

# Helper functions for atmospheric calculations
def calculate_wind_vectors(lat, lon, wind_speed, wind_direction):
    """Calculate wind vector components"""
    # Convert wind direction to radians (meteorological to mathematical)
    wind_dir_rad = np.radians(270 - wind_direction)
    
    # Calculate vector components
    u_component = wind_speed * np.cos(wind_dir_rad)  # East-West component
    v_component = wind_speed * np.sin(wind_dir_rad)  # North-South component
    
    return u_component, v_component

def simulate_atmospheric_profile(airport_code, altitudes):
    """Simulate atmospheric conditions at different altitudes"""
    airport_info = INDIAN_AIRPORTS[airport_code]
    
    # Get surface weather
    surface_weather = weather_api.get_current_weather(
        airport_info['lat'], 
        airport_info['lon']
    )
    
    if not surface_weather:
        return pd.DataFrame()
    
    profile_data = []
    
    for altitude in altitudes:
        # Simulate atmospheric conditions at altitude
        # (In reality, this would come from numerical weather prediction models)
        
        # Temperature lapse rate approximation
        temp_surface = surface_weather['temperature']
        temp_altitude = temp_surface - (altitude / 1000) * 6.5  # Standard lapse rate
        
        # Pressure altitude formula
        pressure_altitude = surface_weather['pressure'] * (1 - 0.0065 * altitude / 288.15) ** 5.255
        
        # Wind typically increases with altitude
        wind_factor = 1 + (altitude - 1000) / 50000  # Simplified wind profile
        wind_speed_altitude = surface_weather['wind_speed'] * wind_factor
        
        # Wind direction can shift with altitude (simulate jet stream effects)
        wind_dir_shift = np.sin(altitude / 10000) * 20  # Simplified directional shear
        wind_direction_altitude = (surface_weather['wind_direction'] + wind_dir_shift) % 360
        
        profile_data.append({
            'altitude': altitude,
            'temperature': temp_altitude,
            'pressure': pressure_altitude,
            'wind_speed': wind_speed_altitude,
            'wind_direction': wind_direction_altitude,
            'latitude': airport_info['lat'],
            'longitude': airport_info['lon']
        })
    
    return pd.DataFrame(profile_data)

# Main visualization logic
if viz_type == "Wind Patterns":
    st.markdown("## 💨 Wind Pattern Analysis")
    
    if not altitude_levels:
        st.warning("Please select at least one altitude level.")
        st.stop()
    
    airport_info = INDIAN_AIRPORTS[selected_airport]
    
    # Get atmospheric profile
    with st.spinner("Calculating atmospheric profile..."):
        profile_df = simulate_atmospheric_profile(selected_airport, altitude_levels)
    
    if not profile_df.empty:
        # Wind speed by altitude
        fig_wind_profile = px.line(
            profile_df,
            x='wind_speed',
            y='altitude',
            title=f"Wind Speed Profile - {selected_airport}",
            labels={'wind_speed': 'Wind Speed (m/s)', 'altitude': 'Altitude (ft)'},
            markers=True
        )
        
        fig_wind_profile.update_layout(height=500)
        st.plotly_chart(fig_wind_profile, use_container_width=True)
        
        # Wind direction by altitude
        fig_wind_dir = px.line(
            profile_df,
            x='wind_direction',
            y='altitude',
            title=f"Wind Direction Profile - {selected_airport}",
            labels={'wind_direction': 'Wind Direction (°)', 'altitude': 'Altitude (ft)'},
            markers=True
        )
        
        fig_wind_dir.update_layout(height=400)
        st.plotly_chart(fig_wind_dir, use_container_width=True)
        
        # Wind barbs visualization
        st.markdown("### 🎯 Wind Barbs by Altitude")
        
        fig_barbs = go.Figure()
        
        for _, row in profile_df.iterrows():
            u_comp, v_comp = calculate_wind_vectors(
                row['latitude'], row['longitude'], 
                row['wind_speed'], row['wind_direction']
            )
            
            # Add wind vector
            fig_barbs.add_annotation(
                x=0,
                y=row['altitude'],
                ax=u_comp * 5,  # Scale for visibility
                ay=v_comp * 5,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="blue",
                text=f"{row['wind_speed']:.1f} m/s"
            )
        
        fig_barbs.update_layout(
            title=f"Wind Vectors by Altitude - {selected_airport}",
            xaxis_title="East-West Component",
            yaxis_title="Altitude (ft)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_barbs, use_container_width=True)
        
        # Turbulence potential analysis
        if show_turbulence_overlay:
            st.markdown("### 🌪️ Wind Shear and Turbulence Potential")
            
            # Calculate wind shear between levels
            shear_data = []
            for i in range(len(profile_df) - 1):
                current_level = profile_df.iloc[i]
                next_level = profile_df.iloc[i + 1]
                
                alt_diff = next_level['altitude'] - current_level['altitude']
                wind_speed_diff = next_level['wind_speed'] - current_level['wind_speed']
                wind_dir_diff = abs(next_level['wind_direction'] - current_level['wind_direction'])
                
                # Handle direction wrap-around
                if wind_dir_diff > 180:
                    wind_dir_diff = 360 - wind_dir_diff
                
                wind_shear = abs(wind_speed_diff) / (alt_diff / 1000)  # Per 1000 ft
                directional_shear = wind_dir_diff / (alt_diff / 1000)
                
                # Calculate turbulence potential
                turb_potential = turbulence_calc.calculate_clear_air_turbulence_potential(
                    current_level['wind_speed'],
                    current_level['wind_direction'],
                    current_level['pressure'],
                    current_level['temperature'],
                    current_level['altitude']
                )
                
                shear_data.append({
                    'altitude_mid': (current_level['altitude'] + next_level['altitude']) / 2,
                    'wind_shear': wind_shear,
                    'directional_shear': directional_shear,
                    'turbulence_potential': turb_potential
                })
            
            if shear_data:
                shear_df = pd.DataFrame(shear_data)
                
                # Wind shear plot
                fig_shear = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["Wind Speed Shear", "Turbulence Potential"]
                )
                
                fig_shear.add_trace(
                    go.Scatter(
                        x=shear_df['wind_shear'],
                        y=shear_df['altitude_mid'],
                        mode='lines+markers',
                        name='Wind Shear',
                        line=dict(color='red', width=3)
                    ),
                    row=1, col=1
                )
                
                fig_shear.add_trace(
                    go.Scatter(
                        x=shear_df['turbulence_potential'],
                        y=shear_df['altitude_mid'],
                        mode='lines+markers',
                        name='Turbulence Potential',
                        line=dict(color='orange', width=3)
                    ),
                    row=1, col=2
                )
                
                fig_shear.update_xaxes(title_text="Wind Shear (m/s per 1000ft)", row=1, col=1)
                fig_shear.update_xaxes(title_text="Turbulence Potential (0-5)", row=1, col=2)
                fig_shear.update_yaxes(title_text="Altitude (ft)")
                
                fig_shear.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_shear, use_container_width=True)

elif viz_type == "Pressure Systems":
    st.markdown("## 🌀 Pressure System Analysis")
    
    airport_info = INDIAN_AIRPORTS[selected_airport]
    
    # Get current weather data
    with st.spinner("Analyzing pressure systems..."):
        current_weather = weather_api.get_current_weather(
            airport_info['lat'], 
            airport_info['lon']
        )
    
    if current_weather:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Simulate pressure gradient (in reality, would use multiple station data)
            # Create a grid around the airport
            lat_range = np.linspace(airport_info['lat'] - 2, airport_info['lat'] + 2, 20)
            lon_range = np.linspace(airport_info['lon'] - 2, airport_info['lon'] + 2, 20)
            
            lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
            
            # Simulate pressure field (simplified model)
            center_lat, center_lon = airport_info['lat'], airport_info['lon']
            base_pressure = current_weather['pressure']
            
            # Create a simple pressure system model
            pressure_grid = np.zeros_like(lat_grid)
            
            for i in range(lat_grid.shape[0]):
                for j in range(lat_grid.shape[1]):
                    distance = np.sqrt((lat_grid[i,j] - center_lat)**2 + (lon_grid[i,j] - center_lon)**2)
                    
                    # Simulate low pressure system (simplified)
                    pressure_variation = np.sin(distance * 2) * 10 + np.random.normal(0, 2)
                    pressure_grid[i,j] = base_pressure + pressure_variation
            
            # Create pressure contour map
            fig_pressure = go.Figure()
            
            fig_pressure.add_trace(go.Contour(
                z=pressure_grid,
                x=lon_range,
                y=lat_range,
                colorscale='RdBu_r',
                contours=dict(
                    start=pressure_grid.min(),
                    end=pressure_grid.max(),
                    size=2
                ),
                colorbar=dict(title="Pressure (hPa)")
            ))
            
            # Add airport marker
            fig_pressure.add_trace(go.Scatter(
                x=[airport_info['lon']],
                y=[airport_info['lat']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond'),
                name=f"{selected_airport} Airport",
                text=f"{selected_airport}<br>Pressure: {current_weather['pressure']:.1f} hPa"
            ))
            
            fig_pressure.update_layout(
                title=f"Pressure System Analysis - {selected_airport}",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=500
            )
            
            st.plotly_chart(fig_pressure, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Pressure Metrics")
            
            st.metric("Surface Pressure", f"{current_weather['pressure']:.1f} hPa")
            
            # Pressure category
            if current_weather['pressure'] < 1000:
                pressure_cat = "Very Low Pressure"
                pressure_color = "🔴"
            elif current_weather['pressure'] < 1010:
                pressure_cat = "Low Pressure"
                pressure_color = "🟠"
            elif current_weather['pressure'] < 1020:
                pressure_cat = "Normal Pressure"
                pressure_color = "🟢"
            else:
                pressure_cat = "High Pressure"
                pressure_color = "🔵"
            
            st.info(f"{pressure_color} **{pressure_cat}**")
            
            # Pressure tendency (simulated)
            pressure_tendency = np.random.normal(0, 1.5)  # Simulated 3-hour tendency
            tendency_direction = "Rising" if pressure_tendency > 0 else "Falling"
            
            st.metric(
                "3-Hour Tendency", 
                f"{abs(pressure_tendency):.1f} hPa",
                delta=f"{tendency_direction}"
            )
            
            # Weather implications
            st.markdown("### 🌤️ Weather Implications")
            
            if current_weather['pressure'] < 1010:
                st.warning("⚠️ Low pressure system - increased turbulence risk")
                st.info("💡 Expect: Unstable conditions, possible precipitation")
            elif current_weather['pressure'] > 1020:
                st.success("✅ High pressure system - stable conditions")
                st.info("💡 Expect: Clear skies, low turbulence risk")
            else:
                st.info("🟡 Moderate pressure conditions")

elif viz_type == "Atmospheric Stability":
    st.markdown("## 🌡️ Atmospheric Stability Analysis")
    
    airport_info = INDIAN_AIRPORTS[selected_airport]
    
    # Get atmospheric data
    with st.spinner("Calculating atmospheric stability..."):
        atmospheric_data = weather_api.get_atmospheric_data(
            airport_info['lat'], 
            airport_info['lon']
        )
    
    if atmospheric_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Stability Metrics")
            
            st.metric("Temperature", f"{atmospheric_data['temperature']:.1f}°C")
            st.metric("Pressure", f"{atmospheric_data['pressure']:.1f} hPa")
            st.metric("Humidity", f"{atmospheric_data['humidity']:.1f}%")
            st.metric("Wind Speed", f"{atmospheric_data['wind_speed']:.1f} m/s")
            
            # Atmospheric stability
            stability = atmospheric_data.get('atmospheric_stability', 'Unknown')
            st.info(f"**Atmospheric Stability:** {stability}")
            
            # Wind shear potential
            shear_potential = atmospheric_data.get('wind_shear_potential', 'Unknown')
            st.info(f"**Wind Shear Potential:** {shear_potential}")
        
        with col2:
            st.markdown("### 🎯 Turbulence Assessment")
            
            # Turbulence indicator
            turb_indicator = atmospheric_data.get('turbulence_indicator', 'Unknown')
            
            if turb_indicator == 'Severe':
                st.error(f"🔴 **Turbulence Risk:** {turb_indicator}")
            elif turb_indicator == 'High':
                st.warning(f"🟠 **Turbulence Risk:** {turb_indicator}")
            elif turb_indicator == 'Moderate':
                st.info(f"🟡 **Turbulence Risk:** {turb_indicator}")
            else:
                st.success(f"🟢 **Turbulence Risk:** {turb_indicator}")
            
            # Richardson Number simulation
            # (Simplified calculation for demonstration)
            temp_gradient = -6.5  # Standard atmosphere lapse rate
            wind_shear = atmospheric_data['wind_speed'] / 1000  # Simplified
            
            if wind_shear > 0:
                richardson_number = (9.81 / 288.15) * temp_gradient / (wind_shear ** 2)
            else:
                richardson_number = float('inf')
            
            st.metric("Richardson Number", f"{richardson_number:.2f}")
            
            if richardson_number < 0.25:
                st.error("⚠️ **Unstable conditions** - High turbulence risk")
            elif richardson_number < 1.0:
                st.warning("🟡 **Marginally stable** - Moderate turbulence risk")
            else:
                st.success("✅ **Stable conditions** - Low turbulence risk")
        
        # Atmospheric profile visualization
        if altitude_levels:
            profile_df = simulate_atmospheric_profile(selected_airport, altitude_levels)
            
            if not profile_df.empty:
                # Temperature profile
                fig_temp_profile = px.line(
                    profile_df,
                    x='temperature',
                    y='altitude',
                    title=f"Temperature Profile - {selected_airport}",
                    labels={'temperature': 'Temperature (°C)', 'altitude': 'Altitude (ft)'},
                    markers=True
                )
                
                # Add standard atmosphere reference
                std_temps = [15 - (alt/1000) * 6.5 for alt in altitude_levels]
                fig_temp_profile.add_trace(
                    go.Scatter(
                        x=std_temps,
                        y=altitude_levels,
                        mode='lines',
                        name='Standard Atmosphere',
                        line=dict(dash='dash', color='red')
                    )
                )
                
                fig_temp_profile.update_layout(height=400)
                st.plotly_chart(fig_temp_profile, use_container_width=True)

elif viz_type == "3D Atmospheric Model":
    st.markdown("## 🌐 3D Atmospheric Model")
    
    airport_info = INDIAN_AIRPORTS[selected_airport]
    
    # Create 3D atmospheric visualization
    if altitude_levels:
        profile_df = simulate_atmospheric_profile(selected_airport, altitude_levels)
        
        if not profile_df.empty:
            # Create 3D scatter plot
            fig_3d = go.Figure(data=[
                go.Scatter3d(
                    x=profile_df['wind_speed'],
                    y=profile_df['temperature'],
                    z=profile_df['altitude'],
                    mode='markers+lines',
                    marker=dict(
                        size=8,
                        color=profile_df['pressure'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Pressure (hPa)")
                    ),
                    line=dict(color='blue', width=4),
                    text=[f"Alt: {alt}ft<br>Wind: {ws:.1f}m/s<br>Temp: {temp:.1f}°C<br>Press: {press:.1f}hPa" 
                          for alt, ws, temp, press in zip(
                              profile_df['altitude'], profile_df['wind_speed'],
                              profile_df['temperature'], profile_df['pressure']
                          )],
                    hovertemplate='%{text}<extra></extra>'
                )
            ])
            
            fig_3d.update_layout(
                title=f"3D Atmospheric Model - {selected_airport}",
                scene=dict(
                    xaxis_title='Wind Speed (m/s)',
                    yaxis_title='Temperature (°C)',
                    zaxis_title='Altitude (ft)'
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 3D wind field visualization
            st.markdown("### 💨 3D Wind Field")
            
            # Create wind vector visualization
            fig_wind_3d = go.Figure()
            
            for _, row in profile_df.iterrows():
                u_comp, v_comp = calculate_wind_vectors(
                    row['latitude'], row['longitude'],
                    row['wind_speed'], row['wind_direction']
                )
                
                # Wind vector in 3D
                fig_wind_3d.add_trace(go.Scatter3d(
                    x=[0, u_comp],
                    y=[0, v_comp],
                    z=[row['altitude'], row['altitude']],
                    mode='lines+markers',
                    line=dict(width=6, color='red'),
                    marker=dict(size=4),
                    showlegend=False
                ))
            
            fig_wind_3d.update_layout(
                title=f"3D Wind Vectors - {selected_airport}",
                scene=dict(
                    xaxis_title='East-West (m/s)',
                    yaxis_title='North-South (m/s)',
                    zaxis_title='Altitude (ft)'
                ),
                height=600
            )
            
            st.plotly_chart(fig_wind_3d, use_container_width=True)

elif viz_type == "Regional Weather Map":
    st.markdown("## 🗺️ Regional Weather Patterns")
    
    # Define region boundaries
    region_coords = {
        "All India": {"lat_range": [6, 38], "lon_range": [68, 98]},
        "North India": {"lat_range": [28, 38], "lon_range": [70, 88]},
        "South India": {"lat_range": [6, 20], "lon_range": [72, 88]},
        "West India": {"lat_range": [15, 28], "lon_range": [68, 78]},
        "East India": {"lat_range": [20, 28], "lon_range": [85, 98]}
    }
    
    coords = region_coords[region_center]
    
    # Create regional weather map
    regional_data = []
    
    # Get data for airports in the region
    for airport_code, airport_info in INDIAN_AIRPORTS.items():
        lat, lon = airport_info['lat'], airport_info['lon']
        
        # Check if airport is in selected region
        if (coords['lat_range'][0] <= lat <= coords['lat_range'][1] and
            coords['lon_range'][0] <= lon <= coords['lon_range'][1]):
            
            weather_data = weather_api.get_current_weather(lat, lon)
            
            if weather_data:
                regional_data.append({
                    'airport': airport_code,
                    'name': airport_info['name'],
                    'city': airport_info['city'],
                    'lat': lat,
                    'lon': lon,
                    'temperature': weather_data['temperature'],
                    'wind_speed': weather_data['wind_speed'],
                    'wind_direction': weather_data['wind_direction'],
                    'pressure': weather_data['pressure'],
                    'humidity': weather_data['humidity'],
                    'weather_condition': weather_data['weather_condition']
                })
    
    if regional_data:
        regional_df = pd.DataFrame(regional_data)
        
        # Temperature map
        fig_temp_map = px.scatter_mapbox(
            regional_df,
            lat="lat",
            lon="lon",
            color="temperature",
            size="wind_speed",
            hover_name="city",
            hover_data=["airport", "temperature", "wind_speed", "pressure"],
            color_continuous_scale="RdYlBu_r",
            size_max=20,
            zoom=5,
            title=f"Regional Temperature Map - {region_center}"
        )
        
        fig_temp_map.update_layout(
            mapbox_style="open-street-map",
            height=600
        )
        
        st.plotly_chart(fig_temp_map, use_container_width=True)
        
        # Wind speed map
        fig_wind_map = px.scatter_mapbox(
            regional_df,
            lat="lat",
            lon="lon",
            color="wind_speed",
            size="pressure",
            hover_name="city",
            hover_data=["airport", "wind_speed", "wind_direction", "pressure"],
            color_continuous_scale="Viridis",
            size_max=20,
            zoom=5,
            title=f"Regional Wind Speed Map - {region_center}"
        )
        
        fig_wind_map.update_layout(
            mapbox_style="open-street-map",
            height=600
        )
        
        st.plotly_chart(fig_wind_map, use_container_width=True)
        
        # Regional statistics
        st.markdown("### 📊 Regional Weather Statistics")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            avg_temp = regional_df['temperature'].mean()
            st.metric("Average Temperature", f"{avg_temp:.1f}°C")
        
        with stats_col2:
            avg_wind = regional_df['wind_speed'].mean()
            st.metric("Average Wind Speed", f"{avg_wind:.1f} m/s")
        
        with stats_col3:
            avg_pressure = regional_df['pressure'].mean()
            st.metric("Average Pressure", f"{avg_pressure:.1f} hPa")
        
        with stats_col4:
            avg_humidity = regional_df['humidity'].mean()
            st.metric("Average Humidity", f"{avg_humidity:.1f}%")
    else:
        st.warning("No weather data available for the selected region.")

# Analysis summary
st.markdown("## 📋 Analysis Summary")

summary_points = []

if viz_type == "Wind Patterns" and 'profile_df' in locals() and not profile_df.empty:
    max_wind = profile_df['wind_speed'].max()
    max_wind_alt = profile_df.loc[profile_df['wind_speed'].idxmax(), 'altitude']
    summary_points.append(f"🌪️ Maximum wind speed: {max_wind:.1f} m/s at {max_wind_alt:,.0f} ft")
    
    if max_wind > 25:
        summary_points.append("⚠️ Strong winds detected - increased turbulence risk")

elif viz_type == "Pressure Systems" and 'current_weather' in locals():
    pressure = current_weather['pressure']
    if pressure < 1010:
        summary_points.append("🌀 Low pressure system detected - monitor for weather changes")
    elif pressure > 1020:
        summary_points.append("🌤️ High pressure system - generally stable conditions")

elif viz_type == "Regional Weather Map" and 'regional_df' in locals():
    temp_range = regional_df['temperature'].max() - regional_df['temperature'].min()
    summary_points.append(f"🌡️ Temperature variation across region: {temp_range:.1f}°C")
    
    strong_wind_airports = regional_df[regional_df['wind_speed'] > 15]['airport'].tolist()
    if strong_wind_airports:
        summary_points.append(f"💨 Strong winds detected at: {', '.join(strong_wind_airports)}")

if not summary_points:
    summary_points.append("✅ Analysis completed - check visualizations above for detailed insights")

for point in summary_points:
    st.info(point)
