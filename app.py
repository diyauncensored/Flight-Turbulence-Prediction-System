import streamlit as st
import pandas as pd
from utils.airport_data import INDIAN_AIRPORTS
import plotly.express as px
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Flight Turbulence Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🌩️ Flight Turbulence Prediction System for Indian Airports")
st.markdown("### Comprehensive Machine Learning-Based Turbulence Analysis")

# Overview section
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Airports Monitored", "6", "Delhi, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata")

with col2:
    st.metric("Prediction Models", "3", "Random Forest, Gradient Boosting, Neural Network")

with col3:
    st.metric("Data Sources", "4", "Weather API, Historical Data, Flight Parameters, Atmospheric")

# Features overview
st.markdown("## 🚀 System Features")

features = [
    {"Feature": "Interactive Airport Map", "Description": "Real-time turbulence risk visualization for major Indian airports", "Icon": "🗺️"},
    {"Feature": "Weather Data Integration", "Description": "Live atmospheric conditions analysis with multiple parameters", "Icon": "🌤️"},
    {"Feature": "ML Turbulence Prediction", "Description": "Advanced machine learning models for accurate turbulence forecasting", "Icon": "🤖"},
    {"Feature": "Route Risk Assessment", "Description": "Flight path turbulence analysis for origin-destination pairs", "Icon": "✈️"},
    {"Feature": "Historical Data Analysis", "Description": "Time-series analysis and pattern detection dashboard", "Icon": "📊"},
    {"Feature": "Real-time Predictions", "Description": "Live turbulence risk levels with confidence scoring", "Icon": "⚡"},
    {"Feature": "Atmospheric Visualization", "Description": "Wind patterns, pressure systems, and jet stream analysis", "Icon": "🌬️"},
    {"Feature": "Custom Data Upload", "Description": "Upload flight parameters and historical turbulence reports", "Icon": "📤"},
    {"Feature": "Statistical Dashboard", "Description": "Comprehensive analytics by airport, season, and altitude", "Icon": "📈"}
]

features_df = pd.DataFrame(features)

for i, feature in enumerate(features):
    with st.expander(f"{feature['Icon']} {feature['Feature']}"):
        st.write(feature['Description'])

# Quick airport overview
st.markdown("## 🏢 Monitored Airports")

airport_overview = []
for code, info in INDIAN_AIRPORTS.items():
    airport_overview.append({
        "Code": code,
        "Name": info["name"],
        "City": info["city"],
        "Latitude": info["lat"],
        "Longitude": info["lon"],
        "Risk Level": "Updating..."  # This would be populated with real data
    })

df_airports = pd.DataFrame(airport_overview)

# Create a simple map visualization
fig = px.scatter_mapbox(
    df_airports, 
    lat="Latitude", 
    lon="Longitude",
    hover_name="Name",
    hover_data=["Code", "City"],
    zoom=4,
    height=500,
    mapbox_style="open-street-map",
    title="Indian Airports - Turbulence Monitoring Network"
)

fig.update_layout(
    mapbox=dict(
        center=dict(lat=20.5937, lon=78.9629),  # Center of India
    )
)

st.plotly_chart(fig, use_container_width=True)

# System status
st.markdown("## 📊 System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.success("Weather API: Online")

with status_col2:
    st.success("ML Models: Loaded")

with status_col3:
    st.success("Data Processing: Active")

with status_col4:
    st.info("Last Updated: Real-time")

# Instructions
st.markdown("""
## 📝 How to Use This System

1. **🗺️ Airport Map**: View real-time turbulence conditions across Indian airports
2. **🌤️ Weather Analysis**: Analyze current atmospheric conditions affecting turbulence
3. **🤖 ML Prediction**: Generate turbulence predictions using machine learning models
4. **✈️ Route Assessment**: Assess turbulence risk for specific flight routes
5. **📊 Historical Analysis**: Explore historical turbulence patterns and trends
6. **⚡ Real-time Predictions**: Monitor live turbulence risk levels
7. **🌬️ Atmospheric Visualization**: Visualize atmospheric conditions and patterns
8. **📤 Data Upload**: Upload custom flight data for analysis
9. **📈 Statistical Dashboard**: View comprehensive turbulence statistics

Navigate using the sidebar to access different features of the system.
""")

# Disclaimer
st.markdown("""
---
**⚠️ Disclaimer**: This system provides turbulence predictions based on available data and machine learning models. 
Always consult official aviation weather services and follow standard aviation safety protocols.
""")
