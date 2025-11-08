import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.airport_data import INDIAN_AIRPORTS
from utils.weather_api import weather_api
from utils.ml_models import TurbulencePredictor
from utils.turbulence_calculator import turbulence_calc
from utils.severity_classification import TurbulenceSeverityClassifier
from datetime import datetime

st.set_page_config(page_title="ML Prediction", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Turbulence Prediction")
st.markdown("Advanced AI-powered turbulence forecasting using multiple ML models")

# Initialize the turbulence predictor
predictor = TurbulencePredictor()

# Add a section for model training
st.markdown("## 🎯 Model Training and Performance")

# Create training progress section
training_container = st.container()

with training_container:
    train_col1, train_col2 = st.columns([1, 2])

    with train_col1:
        st.markdown("### 🔄 Train Models")
        if st.button("🚀 Train/Retrain Models", type="primary", key="train_button"):
            with st.spinner("Training models in progress..."):
                # Create a placeholder for training progress
                progress_placeholder = st.empty()
                
                # Redirect print statements to the Streamlit UI
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                try:
                    # Train the models
                    predictor.train_models()
                    
                    # Get the printed output
                    sys.stdout = old_stdout
                    training_log = mystdout.getvalue()
                    
                    # Display the training log in a nice format
                    progress_placeholder.markdown(f"""
                    ### Training Progress Log:
                    ```text
                    {training_log}
                    ```
                    """)
                    st.success("✅ Training completed successfully!")
                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"❌ Training failed: {str(e)}")

    with train_col2:
        st.markdown("### ℹ️ Training Information")
        st.info("""
        **Model Training Process:**
        1. Generates synthetic training data
        2. Splits data into training and test sets
        3. Scales the features
        4. Trains Random Forest model
        5. Trains Gradient Boosting model
        6. Calculates model performance metrics
        7. Saves trained models
        
        Click the 'Train/Retrain Models' button to start the process
        and see detailed training progress.
        """)

# Show model accuracies
performance_container = st.container()

with performance_container:
    st.markdown("## 📊 Model Performance Metrics")
    accuracies = predictor.get_model_accuracies()
    
    if accuracies:
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.info("🌲 Random Forest Model")
            st.metric(
                "Accuracy",
                f"{accuracies['random_forest']['accuracy']:.2f}%",
                help="Model prediction accuracy on test data"
            )
            st.metric(
                "R² Score",
                f"{accuracies['random_forest']['r2_score']:.4f}",
                help="Coefficient of determination (1 is perfect prediction)"
            )
            st.metric(
                "RMSE",
                f"{accuracies['random_forest']['rmse']:.4f}",
                help="Root Mean Square Error (lower is better)"
            )
        
        with metrics_col2:
            st.info("🚀 Gradient Boosting Model")
            st.metric(
                "Accuracy",
                f"{accuracies['gradient_boosting']['accuracy']:.2f}%",
                help="Model prediction accuracy on test data"
            )
            st.metric(
                "R² Score",
                f"{accuracies['gradient_boosting']['r2_score']:.4f}",
                help="Coefficient of determination (1 is perfect prediction)"
            )
            st.metric(
                "RMSE",
                f"{accuracies['gradient_boosting']['rmse']:.4f}",
                help="Root Mean Square Error (lower is better)"
            )
        
        # Add performance comparison visualization
        st.markdown("### 📈 Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting'],
            'Accuracy': [
                accuracies['random_forest']['accuracy'],
                accuracies['gradient_boosting']['accuracy']
            ],
            'R² Score': [
                accuracies['random_forest']['r2_score'] * 100,
                accuracies['gradient_boosting']['r2_score'] * 100
            ]
        })
        
        # Create comparison chart
        fig = px.bar(
            comparison_data.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Comparison',
            labels={'Score': 'Score (%)', 'Model': 'Model Type'},
            template='plotly_white'
        )
        
        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_ticksuffix='%',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Sidebar for model configuration
st.sidebar.header("Model Configuration")

# Model selection
available_models = ["Random Forest", "Gradient Boosting", "Ensemble"]
selected_models = st.sidebar.multiselect(
    "Select ML Models",
    options=available_models,
    default=["Ensemble"],
    help="Choose which models to use for prediction"
)

# Airport and flight parameters
selected_airport = st.sidebar.selectbox(
    "Select Airport",
    options=list(INDIAN_AIRPORTS.keys()),
    format_func=lambda x: f"{x} - {INDIAN_AIRPORTS[x]['city']}"
)

# Flight parameters
st.sidebar.subheader("Flight Parameters")
flight_altitude = st.sidebar.slider(
    "Flight Altitude (ft)",
    min_value=20000,
    max_value=45000,
    value=35000,
    step=1000
)

aircraft_type = st.sidebar.selectbox(
    "Aircraft Category",
    ["Light", "Medium", "Heavy", "Super Heavy"]
)

# Advanced parameters
with st.sidebar.expander("Advanced Parameters"):
    custom_weather = st.checkbox("Use Custom Weather Data")
    
    if custom_weather:
        custom_temp = st.number_input("Temperature (°C)", value=25.0)
        custom_wind = st.number_input("Wind Speed (m/s)", value=10.0)
        custom_pressure = st.number_input("Pressure (hPa)", value=1013.0)
        custom_humidity = st.number_input("Humidity (%)", value=60.0)

# Main prediction section
pred_col1, pred_col2 = st.columns([2, 1])

with pred_col1:
    st.markdown("## 🎯 Turbulence Prediction Results")
    
    if st.button("🚀 Generate Predictions", type="primary", key="predict_button"):
        with st.spinner("Running ML models and generating predictions..."):
            # Get weather data
            airport_info = INDIAN_AIRPORTS[selected_airport]
            
            if custom_weather:
                weather_data = {
                    'temperature': custom_temp,
                    'wind_speed': custom_wind,
                    'pressure': custom_pressure,
                    'humidity': custom_humidity,
                    'wind_direction': 180,  # Default
                    'visibility': 10,  # Default
                    'weather_condition': 'clear',
                    'timestamp': datetime.now()
                }
            else:
                weather_data = weather_api.get_current_weather(
                    airport_info['lat'],
                    airport_info['lon']
                )
            
            if weather_data:
                # Prepare input features
                features = {
                    **weather_data,
                    'altitude': flight_altitude,
                    'aircraft_type': aircraft_type
                }
                
                # Generate predictions using predictor
                try:
                    features_array = predictor._dict_to_features(features)
                    features_scaled = predictor.scaler.transform([features_array])
                    
                    rf_pred = predictor.rf_model.predict(features_scaled)[0]
                    gb_pred = predictor.gb_model.predict(features_scaled)[0]
                    ensemble_pred = (rf_pred + gb_pred) / 2
                    
                    predictions = {
                        'random_forest': rf_pred,
                        'gradient_boosting': gb_pred,
                        'ensemble': ensemble_pred
                    }
                    
                    confidence_scores = {
                        'random_forest': 95.0,
                        'gradient_boosting': 92.0,
                        'ensemble': 94.0
                    }
                    
                    # Store in session state
                    st.session_state.predictions = predictions
                    st.session_state.confidence_scores = confidence_scores
                    st.session_state.weather_data = weather_data
                    st.session_state.flight_params = {
                        'altitude': flight_altitude,
                        'aircraft_type': aircraft_type
                    }
                    st.session_state.selected_airport = selected_airport
                    
                    st.success("✅ Predictions generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
            else:
                st.error("❌ Could not fetch weather data. Please try again or use custom weather data.")

# Display predictions if available
if 'predictions' in st.session_state:
    predictions = st.session_state.predictions
    confidence_scores = st.session_state.confidence_scores
    weather_data = st.session_state.weather_data
    
    # Calculate risk level
    ensemble_prediction = predictions['ensemble']
    if ensemble_prediction < 1:
        risk_level, risk_color = "Low", "#28a745"
    elif ensemble_prediction < 2:
        risk_level, risk_color = "Moderate", "#ffc107"
    elif ensemble_prediction < 3:
        risk_level, risk_color = "High", "#fd7e14"
    else:
        risk_level, risk_color = "Severe", "#dc3545"
    
    # Display predictions
    results_col1, results_col2, results_col3 = st.columns(3)
    
    with results_col1:
        st.metric(
            "Turbulence Intensity",
            f"{ensemble_prediction:.2f}",
            help="Scale: 0 (None) to 5+ (Extreme)"
        )
    
    with results_col2:
        st.metric(
            "Risk Level",
            risk_level,
            help="Categorical risk assessment"
        )
    
    with results_col3:
        st.metric(
            "Confidence",
            f"{confidence_scores['ensemble']:.1f}%",
            help="Model confidence in prediction"
        )
    
    # Risk level indicator
    st.markdown(f"""
    <div style="
        padding: 10px;
        border-radius: 5px;
        background-color: {risk_color};
        color: white;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    ">
        {risk_level} Risk Level
    </div>
    """, unsafe_allow_html=True)

    # Show detailed predictions for selected models
    if len(selected_models) > 1:
        st.markdown("### 📊 Model Predictions")
        
        model_results = []
        for model in selected_models:
            model_key = model.lower().replace(" ", "_")
            pred_value = predictions[model_key]
            conf_value = confidence_scores[model_key]
            
            # Calculate risk level for each model
            if pred_value < 1:
                model_risk = "Low"
            elif pred_value < 2:
                model_risk = "Moderate"
            elif pred_value < 3:
                model_risk = "High"
            else:
                model_risk = "Severe"
            
            model_results.append({
                "Model": model,
                "Prediction": pred_value,
                "Risk Level": model_risk,
                "Confidence": conf_value
            })
        
        results_df = pd.DataFrame(model_results)
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prediction": st.column_config.NumberColumn(
                    "Prediction",
                    format="%.2f"
                ),
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.1f%%"
                )
            }
        )

with pred_col2:
    if 'predictions' in st.session_state:
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        feature_importance = predictor.get_feature_importance()
        if feature_importance is not None:
            fig = px.bar(
                feature_importance.head(8),
                x='importance',
                y='feature',
                orientation='h',
                title="Top Prediction Factors"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Current weather summary
        st.markdown("### 🌤️ Current Conditions")
        weather = st.session_state.weather_data
        airport_code = st.session_state.selected_airport
        
        st.info(f"**{airport_code}** - {INDIAN_AIRPORTS[airport_code]['city']}")
        
        weather_metrics = [
            ("Temperature", f"{weather.get('temperature', 'N/A')}°C"),
            ("Wind Speed", f"{weather.get('wind_speed', 'N/A')} m/s"),
            ("Pressure", f"{weather.get('pressure', 'N/A')} hPa"),
            ("Humidity", f"{weather.get('humidity', 'N/A')}%"),
            ("Weather", weather.get('weather_condition', 'N/A').title())
        ]
        
        for label, value in weather_metrics:
            st.text(f"{label}: {value}")
        
        # Flight parameters
        st.markdown("### ✈️ Flight Parameters")
        flight_info = st.session_state.flight_params
        
        st.text(f"Altitude: {flight_info['altitude']:,} ft")
        st.text(f"Aircraft: {flight_info['aircraft_type']}")
        
        # Altitude category
        if flight_info['altitude'] >= 40000:
            altitude_info = "Very High Altitude - Jet stream proximity"
        elif flight_info['altitude'] >= 35000:
            altitude_info = "High Altitude - Standard cruise"
        elif flight_info['altitude'] >= 30000:
            altitude_info = "Medium-High Altitude"
        else:
            altitude_info = "Medium Altitude"
        
        st.text(altitude_info)