import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils.severity_classification import TurbulenceSeverityClassifier

st.set_page_config(page_title="Severity Classification", page_icon="📊", layout="wide")

st.title("📊 Enhanced Turbulence Severity Classification")
st.markdown("Detailed turbulence intensity levels with comprehensive risk assessment")

# Initialize classifier
classifier = TurbulenceSeverityClassifier()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Severity Levels", "Risk Assessment", "Live Classification", "Comparison Tool"])

with tab1:
    st.header("Turbulence Severity Scale")
    st.markdown("8-level classification system from Smooth to Extreme conditions")
    
    # Get all severity levels
    levels = classifier.get_all_severity_levels()
    
    # Visual scale
    st.subheader("Severity Scale Visualization")
    
    fig = go.Figure()
    
    for i, level_info in enumerate(levels):
        fig.add_trace(go.Bar(
            x=[level_info['max'] - level_info['min']],
            y=[level_info['level']],
            orientation='h',
            name=level_info['level'],
            marker=dict(color=level_info['color']),
            text=f"{level_info['icon']} {level_info['level']}<br>{level_info['min']}-{level_info['max']}",
            textposition='auto',
            hovertemplate=f"<b>{level_info['level']}</b><br>" +
                         f"Range: {level_info['min']}-{level_info['max']}<br>" +
                         f"{level_info['description']}<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        height=500,
        xaxis_title="Turbulence Index (0-10)",
        yaxis_title="Severity Level",
        xaxis=dict(range=[0, 10])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed information table
    st.subheader("Detailed Severity Descriptions")
    
    for level_info in levels:
        with st.expander(f"{level_info['icon']} {level_info['level']} ({level_info['min']}-{level_info['max']})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Description:**")
                st.write(classifier.descriptions[level_info['level']])
                
                st.markdown(f"**Passenger Impact:**")
                st.write(classifier.passenger_impact[level_info['level']])
            
            with col2:
                st.markdown(f"**Pilot Actions:**")
                st.write(classifier.pilot_actions[level_info['level']])
                
                st.markdown(f"**Structural Concern:**")
                concern = classifier.structural_concern[level_info['level']]
                st.progress(concern / 100, text=f"{concern}% of design limits")

with tab2:
    st.header("Risk Assessment Tool")
    st.markdown("Calculate detailed risk assessment based on turbulence index and duration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        turbulence_input = st.slider(
            "Turbulence Index",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Select turbulence intensity (0=Smooth, 10=Extreme)"
        )
        
        duration_input = st.slider(
            "Expected Duration (minutes)",
            min_value=1,
            max_value=60,
            value=10,
            help="Duration affects risk assessment"
        )
    
    with col2:
        # Get risk assessment
        assessment = classifier.get_risk_assessment(turbulence_input, duration_input)
        
        # Display risk level
        st.markdown(f"""
        <div style="background-color: {assessment['risk_color']}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">Risk Level: {assessment['risk_level']}</h2>
            <p style="color: white; margin: 10px 0 0 0;">Adjusted Index: {assessment['adjusted_index']:.2f}/10</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Duration Factor:** {assessment['duration_factor']:.2f}x")
    
    # Severity information
    st.subheader("Severity Details")
    
    severity_info = assessment['severity_info']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**{severity_info['icon']} {severity_info['level']}**")
        st.write(severity_info['description'])
    
    with col2:
        st.markdown("**Passenger Impact:**")
        st.write(severity_info['passenger_impact'])
    
    with col3:
        st.markdown("**Structural Concern:**")
        st.progress(severity_info['structural_concern'] / 100, 
                   text=f"{severity_info['structural_concern']}%")
    
    # Pilot actions
    st.subheader("Required Pilot Actions")
    st.info(severity_info['pilot_action'])
    
    # Recommendations
    st.subheader("Detailed Recommendations")
    for i, rec in enumerate(assessment['recommendations'], 1):
        st.write(f"{i}. {rec}")

with tab3:
    st.header("Live Turbulence Classification")
    st.markdown("Real-time classification based on current weather conditions")
    
    from utils.airport_data import get_airports
    from utils.weather_api import WeatherAPI
    
    weather_api = WeatherAPI()
    airports = get_airports()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        airport_names = {f"{a['name']} ({a['code']})": a for a in airports}
        selected = st.selectbox("Select Airport", options=list(airport_names.keys()))
        airport = airport_names[selected]
        
        altitude_input = st.number_input(
            "Flight Altitude (feet)",
            min_value=0,
            max_value=50000,
            value=35000,
            step=1000
        )
        
        if st.button("Calculate Live Classification", type="primary"):
            with st.spinner("Analyzing conditions..."):
                # Get weather data
                weather = weather_api.get_current_weather(
                    airport['coordinates'][0],
                    airport['coordinates'][1]
                )
                
                if weather:
                    # Calculate turbulence index
                    weather_params = {
                        'wind_speed': weather['wind_speed'],
                        'wind_shear': weather['wind_speed'] * 0.3,  # Estimate
                        'temp_gradient': abs(weather['temperature'] - 15),
                        'pressure_change': abs(weather['pressure'] - 1013),
                        'altitude': altitude_input
                    }
                    
                    turb_index = classifier.calculate_turbulence_index(weather_params)
                    st.session_state.live_classification = {
                        'index': turb_index,
                        'weather': weather,
                        'airport': airport
                    }
    
    with col2:
        if 'live_classification' in st.session_state:
            data = st.session_state.live_classification
            severity_info = classifier.get_severity_info(data['index'])
            
            # Display result
            st.markdown(f"""
            <div style="background-color: {severity_info['color']}; padding: 30px; border-radius: 15px; text-align: center;">
                <h1 style="color: white; margin: 0;">{severity_info['icon']} {severity_info['level']}</h1>
                <h3 style="color: white; margin: 10px 0;">Turbulence Index: {severity_info['index']:.2f}/10</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Weather Conditions:**")
                st.write(f"Wind Speed: {data['weather']['wind_speed']:.1f} m/s")
                st.write(f"Temperature: {data['weather']['temperature']:.1f}°C")
                st.write(f"Pressure: {data['weather']['pressure']:.0f} hPa")
            
            with col_b:
                st.markdown("**Severity Details:**")
                st.write(severity_info['description'])
                st.markdown(f"**Action Required:**")
                st.write(severity_info['pilot_action'])

with tab4:
    st.header("Severity Comparison Tool")
    st.markdown("Compare two turbulence conditions side by side")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Condition A")
        index_a = st.slider("Turbulence Index A", 0.0, 10.0, 3.0, 0.1, key="index_a")
        severity_a = classifier.get_severity_info(index_a)
        
        st.markdown(f"""
        <div style="background-color: {severity_a['color']}; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: white;">{severity_a['icon']} {severity_a['level']}</h3>
            <p style="color: white;">Index: {severity_a['index']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Description:** {severity_a['description']}")
        st.write(f"**Passenger Impact:** {severity_a['passenger_impact']}")
    
    with col2:
        st.subheader("Condition B")
        index_b = st.slider("Turbulence Index B", 0.0, 10.0, 7.0, 0.1, key="index_b")
        severity_b = classifier.get_severity_info(index_b)
        
        st.markdown(f"""
        <div style="background-color: {severity_b['color']}; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: white;">{severity_b['icon']} {severity_b['level']}</h3>
            <p style="color: white;">Index: {severity_b['index']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Description:** {severity_b['description']}")
        st.write(f"**Passenger Impact:** {severity_b['passenger_impact']}")
    
    # Comparison metrics
    st.markdown("---")
    st.subheader("Comparison Analysis")
    
    comparison = classifier.compare_severities(index_a, index_b)
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Difference", f"{comparison['difference']:.2f}", 
                 delta=f"{comparison['percentage_change']:.1f}%")
    
    with col_b:
        if comparison['difference'] < 1:
            st.success("Similar severity levels")
        elif comparison['difference'] < 3:
            st.warning("Moderate difference")
        else:
            st.error("Significant difference")
    
    with col_c:
        if index_b > index_a:
            st.error("⬆️ Condition worsening")
        elif index_b < index_a:
            st.success("⬇️ Condition improving")
        else:
            st.info("➡️ Condition stable")

st.markdown("---")
st.caption("Enhanced classification system provides detailed turbulence analysis for better decision-making")
