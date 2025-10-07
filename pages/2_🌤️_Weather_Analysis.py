import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from utils.airport_data import INDIAN_AIRPORTS
from utils.weather_api import weather_api
from utils.data_processing import data_processor

st.set_page_config(page_title="Weather Analysis", page_icon="🌤️", layout="wide")

st.title("🌤️ Weather Data Analysis - Atmospheric Conditions")
st.markdown("Comprehensive weather analysis for turbulence prediction")

# Sidebar controls
st.sidebar.header("Analysis Parameters")

# Airport selection
selected_airports = st.sidebar.multiselect(
    "Select Airports",
    options=list(INDIAN_AIRPORTS.keys()),
    default=["DEL", "BOM", "BLR"],
    help="Choose airports for weather analysis"
)

# Time range selection
forecast_hours = st.sidebar.slider(
    "Forecast Hours",
    min_value=12,
    max_value=48,
    value=24,
    step=6
)

# Weather parameters to analyze
weather_params = st.sidebar.multiselect(
    "Weather Parameters",
    options=["Temperature", "Pressure", "Wind Speed", "Humidity", "Visibility"],
    default=["Temperature", "Pressure", "Wind Speed", "Humidity"]
)

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Current Conditions", "Forecast Analysis", "Comparative Analysis"]
)

if not selected_airports:
    st.warning("Please select at least one airport for analysis.")
    st.stop()

# Main analysis section
if analysis_type == "Current Conditions":
    st.markdown("## 📊 Current Weather Conditions")
    
    current_data = []
    
    for airport_code in selected_airports:
        airport_info = INDIAN_AIRPORTS[airport_code]
        
        with st.spinner(f"Fetching weather data for {airport_code}..."):
            weather_data = weather_api.get_current_weather(
                airport_info['lat'], 
                airport_info['lon']
            )
            
            if weather_data:
                processed_weather = data_processor.process_weather_data(weather_data)
                processed_weather['airport'] = airport_code
                processed_weather['airport_name'] = airport_info['name']
                current_data.append(processed_weather)
    
    if current_data:
        # Display current conditions in cards
        cols = st.columns(len(selected_airports))
        
        for i, data in enumerate(current_data):
            with cols[i]:
                st.markdown(f"### {data['airport']} - {data['airport_name'].split()[0]}")
                
                # Weather metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Temperature", f"{data['temperature']:.1f}°C")
                    st.metric("Wind Speed", f"{data['wind_speed']:.1f} m/s")
                    st.metric("Pressure", f"{data['pressure']:.0f} hPa")
                
                with col2:
                    st.metric("Humidity", f"{data['humidity']:.0f}%")
                    st.metric("Visibility", f"{data['visibility']:.1f} km")
                    st.metric("Risk Score", f"{data['weather_risk_score']:.1f}/10")
                
                # Weather condition
                st.info(f"Condition: {data['weather_condition']}")
                st.caption(f"Updated: {data['timestamp'].strftime('%H:%M:%S')}")
        
        # Comparative charts
        st.markdown("## 📈 Comparative Analysis")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(current_data)
        
        if len(weather_params) > 0:
            # Create subplots for selected parameters
            param_mapping = {
                "Temperature": "temperature",
                "Pressure": "pressure", 
                "Wind Speed": "wind_speed",
                "Humidity": "humidity",
                "Visibility": "visibility"
            }
            
            num_params = len(weather_params)
            cols_per_row = 2
            rows = (num_params + cols_per_row - 1) // cols_per_row
            
            fig = make_subplots(
                rows=rows,
                cols=cols_per_row,
                subplot_titles=weather_params,
                specs=[[{"type": "bar"} for _ in range(cols_per_row)] for _ in range(rows)]
            )
            
            for i, param in enumerate(weather_params):
                row = (i // cols_per_row) + 1
                col = (i % cols_per_row) + 1
                
                param_key = param_mapping[param]
                
                fig.add_trace(
                    go.Bar(
                        x=comparison_df['airport'],
                        y=comparison_df[param_key],
                        name=param,
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )
            
            fig.update_layout(
                height=300 * rows,
                title="Current Weather Parameters Comparison",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Forecast Analysis":
    st.markdown("## 🔮 Weather Forecast Analysis")
    
    # Airport selection for detailed forecast
    forecast_airport = st.selectbox(
        "Select Airport for Detailed Forecast",
        options=selected_airports,
        help="Choose airport for detailed forecast analysis"
    )
    
    airport_info = INDIAN_AIRPORTS[forecast_airport]
    
    with st.spinner("Fetching forecast data..."):
        forecast_df = weather_api.get_forecast_data(
            airport_info['lat'], 
            airport_info['lon'], 
            forecast_hours
        )
    
    if not forecast_df.empty:
        # Process forecast data
        forecast_processed = data_processor.create_time_series_data(forecast_df)
        
        st.markdown(f"### {forecast_airport} - {airport_info['name']} Forecast")
        
        # Time series charts
        param_mapping = {
            "Temperature": ("temperature", "°C"),
            "Pressure": ("pressure", "hPa"),
            "Wind Speed": ("wind_speed", "m/s"),
            "Humidity": ("humidity", "%"),
            "Risk Score": ("risk_score", "Score")
        }
        
        # Create time series plots
        for param in weather_params:
            if param in param_mapping:
                param_key, unit = param_mapping[param]
                
                fig = px.line(
                    forecast_processed,
                    x='datetime',
                    y=param_key,
                    title=f"{param} Forecast - {forecast_airport}",
                    labels={'datetime': 'Time', param_key: f'{param} ({unit})'}
                )
                
                fig.update_traces(line=dict(width=3))
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment over time
        st.markdown("### Turbulence Risk Assessment Over Time")
        
        fig = px.scatter(
            forecast_processed,
            x='datetime',
            y='risk_score',
            color='risk_score',
            size='wind_speed',
            title=f"Weather Risk Score Forecast - {forecast_airport}",
            labels={'datetime': 'Time', 'risk_score': 'Risk Score (0-10)'},
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed forecast table
        st.markdown("### Detailed Forecast Data")
        
        # Select columns for display
        display_columns = ['datetime', 'temperature', 'pressure', 'wind_speed', 
                         'humidity', 'weather_condition', 'risk_score']
        
        display_df = forecast_processed[display_columns].copy()
        display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Format numeric columns
        for col in ['temperature', 'pressure', 'wind_speed', 'humidity', 'risk_score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(1)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error("Unable to fetch forecast data. Please check your internet connection.")

elif analysis_type == "Comparative Analysis":
    st.markdown("## ⚖️ Multi-Airport Comparative Analysis")
    
    comparative_data = []
    
    # Fetch current weather for all selected airports
    for airport_code in selected_airports:
        airport_info = INDIAN_AIRPORTS[airport_code]
        
        weather_data = weather_api.get_current_weather(
            airport_info['lat'], 
            airport_info['lon']
        )
        
        if weather_data:
            processed_weather = data_processor.process_weather_data(weather_data)
            processed_weather['airport'] = airport_code
            processed_weather['airport_name'] = airport_info['name']
            processed_weather['city'] = airport_info['city']
            processed_weather['elevation'] = airport_info['elevation']
            comparative_data.append(processed_weather)
    
    if comparative_data:
        comp_df = pd.DataFrame(comparative_data)
        
        # Weather parameter correlation matrix
        st.markdown("### Weather Parameters Correlation")
        
        numeric_columns = ['temperature', 'pressure', 'wind_speed', 'humidity', 
                          'visibility', 'weather_risk_score']
        correlation_matrix = comp_df[numeric_columns].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Weather Parameters Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-parameter comparison radar chart
        st.markdown("### Multi-Parameter Comparison")
        
        # Normalize data for radar chart
        normalized_df = comp_df[['airport'] + numeric_columns].copy()
        
        for col in numeric_columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val) * 100
        
        # Create radar chart
        fig = go.Figure()
        
        for _, row in normalized_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row[numeric_columns].values,
                theta=numeric_columns,
                fill='toself',
                name=row['airport']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Normalized Weather Parameters Comparison",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.markdown("### Statistical Summary")
        
        summary_stats = comp_df[numeric_columns].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Airport rankings
        st.markdown("### Airport Rankings by Risk Score")
        
        risk_ranking = comp_df[['airport', 'city', 'weather_risk_score']].sort_values(
            'weather_risk_score', ascending=False
        )
        
        risk_ranking['rank'] = range(1, len(risk_ranking) + 1)
        risk_ranking = risk_ranking[['rank', 'airport', 'city', 'weather_risk_score']]
        
        st.dataframe(
            risk_ranking,
            use_container_width=True,
            hide_index=True,
            column_config={
                "weather_risk_score": st.column_config.NumberColumn(
                    "Risk Score",
                    format="%.2f"
                )
            }
        )
    else:
        st.error("Unable to fetch weather data for comparison.")

# Weather alerts section
st.markdown("## ⚠️ Weather Alerts")

if 'current_data' in locals() and current_data:
    alerts = []
    
    for data in current_data:
        # Check for severe weather conditions
        if data['weather_risk_score'] >= 6:
            alerts.append(f"🔴 **HIGH RISK** at {data['airport']}: Weather risk score {data['weather_risk_score']:.1f}/10")
        
        if data['wind_speed'] > 20:
            alerts.append(f"💨 **STRONG WINDS** at {data['airport']}: {data['wind_speed']:.1f} m/s")
        
        if 'thunderstorm' in data['weather_condition'].lower():
            alerts.append(f"⛈️ **THUNDERSTORM** at {data['airport']}: Active thunderstorm conditions")
        
        if data['visibility'] < 5:
            alerts.append(f"🌫️ **LOW VISIBILITY** at {data['airport']}: {data['visibility']:.1f} km")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ No severe weather alerts for selected airports")
else:
    st.info("Select airports and analysis type to view weather alerts")
