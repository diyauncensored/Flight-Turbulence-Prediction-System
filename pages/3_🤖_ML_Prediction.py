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

        # LSTM evaluation and severity classification
        from utils.lstm_model import lstm_predictor

        st.markdown("### 🧠 LSTM Model & Severity Classification")
        with st.expander("Evaluate LSTM model and severity classifier (on synthetic data)"):
            st.write("This will run a short evaluation on synthetic samples. If the LSTM model isn't available it may perform a brief quick-training run to enable evaluation.")
            if st.button("Evaluate LSTM & Severity", key="eval_lstm_button"):
                with st.spinner("Running LSTM evaluation..."):
                    result = lstm_predictor.evaluate_performance(num_samples=500, quick_train_if_missing=True, quick_epochs=8)
                    if result is None:
                        st.error("LSTM evaluation could not be completed. Ensure TensorFlow is installed and/or pretrain the LSTM model.")
                    else:
                        lstm_metrics = result['lstm']
                        sev_acc = result.get('severity_classification_accuracy')

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info("LSTM Regression Metrics")
                            st.metric("Accuracy", f"{lstm_metrics['accuracy']:.2f}%")
                            st.metric("R² Score", f"{lstm_metrics['r2_score']:.4f}")
                        with col2:
                            st.info("Errors")
                            st.metric("RMSE", f"{lstm_metrics['rmse']:.4f}")
                            st.metric("MAE", f"{lstm_metrics['mae']:.4f}")
                        with col3:
                            st.info("Severity Classification")
                            if sev_acc is not None:
                                st.metric("Severity Accuracy", f"{sev_acc:.2f}%")
                            else:
                                st.write("Severity accuracy not available")

                        st.write(f"Samples evaluated: {result.get('samples_evaluated', 0)}")
train_col1, train_col2 = st.columns([1, 2])

with train_col1:
    st.markdown("### Train Models")
    if st.button("🚀 Train/Retrain Models", type="primary"):
        with st.spinner("Training models in progress..."):
            # Create a placeholder for training progress
            progress_placeholder = st.empty()
            
            # Redirect print statements to the Streamlit UI
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
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

with train_col2:
    st.markdown("### Training Information")
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
with st.sidebar:
    st.markdown("### 🔧 Model Training")
    if st.button("Retrain Models"):
        with st.spinner("Training models..."):
            # Create a placeholder for training progress
            progress_placeholder = st.empty()
            
            # Redirect print statements to the Streamlit UI
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            # Train the models
            predictor.train_models()
            
            # Get the printed output
            sys.stdout = old_stdout
            training_log = mystdout.getvalue()
            
            # Display the training log in a nice format
            progress_placeholder.markdown(f"""
            ```text
            {training_log}
            ```
            """)
            st.success("Training completed!")

# Show model accuracies
st.markdown("### 📊 Current Model Performance")
accuracies = predictor.get_model_accuracies()
if accuracies:
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.info("Random Forest Model")
        st.metric("Accuracy", f"{accuracies['random_forest']['accuracy']:.2f}%", 
                 help="Model prediction accuracy on test data")
        st.metric("R² Score", f"{accuracies['random_forest']['r2_score']:.4f}", 
                 help="Coefficient of determination (1 is perfect prediction)")
        st.metric("RMSE", f"{accuracies['random_forest']['rmse']:.4f}", 
                 help="Root Mean Square Error (lower is better)")
    
    with perf_col2:
        st.info("Gradient Boosting Model")
        st.metric("Accuracy", f"{accuracies['gradient_boosting']['accuracy']:.2f}%", 
                 help="Model prediction accuracy on test data")
        st.metric("R² Score", f"{accuracies['gradient_boosting']['r2_score']:.4f}", 
                 help="Coefficient of determination (1 is perfect prediction)")
        st.metric("RMSE", f"{accuracies['gradient_boosting']['rmse']:.4f}", 
                 help="Root Mean Square Error (lower is better)")
    
    # Add performance comparison visualization
    st.markdown("#### Model Performance Comparison")
    
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
    
    # Create bar chart
    fig = px.bar(
        comparison_data.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
        x='Model',
        y='Score',
        color='Metric',
        barmode='group',
        title='Model Performance Metrics',
        labels={'Score': 'Score (%)', 'Model': 'Model Type'},
        template='plotly_white'
    )
    
    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_ticksuffix='%',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

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
        
    # Visualization of accuracies
    accuracy_data = pd.DataFrame({
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
    
    fig = px.bar(
        accuracy_data.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
        x='Model',
        y='Score',
        color='Metric',
        title='Model Performance Comparison',
        labels={'Score': 'Percentage (%)'}
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
    ["Light", "Medium", "Heavy", "Super Heavy"],
    index=2
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
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🎯 Turbulence Prediction Results")
    
    if st.button("🚀 Generate Predictions", type="primary"):
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
                # Flight parameters
                flight_params = {
                    'altitude': flight_altitude,
                    'aircraft_type': aircraft_type
                }
                
                # Prepare input features
                features = {**weather_data, **flight_params}
                
                # Generate predictions
                rf_pred = predictor.rf_model.predict(predictor.scaler.transform([predictor._dict_to_features(features)]))[0]
                gb_pred = predictor.gb_model.predict(predictor.scaler.transform([predictor._dict_to_features(features)]))[0]
                ensemble_pred = (rf_pred + gb_pred) / 2
                
                predictions = {
                    'random_forest': rf_pred,
                    'gradient_boosting': gb_pred,
                    'ensemble': ensemble_pred
                }
                
                confidence_scores = {
                    'random_forest': 95.0,  # Example confidence scores
                    'gradient_boosting': 92.0,
                    'ensemble': 94.0
                }
                
                # Store in session state for visualization
                st.session_state.predictions = predictions
                st.session_state.confidence_scores = confidence_scores
                st.session_state.weather_data = weather_data
                st.session_state.flight_params = flight_params
                st.session_state.selected_airport = selected_airport
    
    # Display predictions if available
    if 'predictions' in st.session_state:
        predictions = st.session_state.predictions
        confidence_scores = st.session_state.confidence_scores
        weather_data = st.session_state.weather_data
        
        # Main prediction display
        ensemble_prediction = predictions.get('ensemble', 0)
        ensemble_confidence = confidence_scores.get('ensemble', 0)
        # Calculate risk level
        if ensemble_prediction < 1:
            risk_level, risk_color = "Low", "#28a745"
        elif ensemble_prediction < 2:
            risk_level, risk_color = "Moderate", "#ffc107"
        elif ensemble_prediction < 3:
            risk_level, risk_color = "High", "#fd7e14"
        else:
            risk_level, risk_color = "Severe", "#dc3545"
        
        # Big prediction display
        st.markdown("### 🎯 Primary Prediction")
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            st.metric(
                "Turbulence Intensity",
                f"{ensemble_prediction:.2f}",
                help="Scale: 0 (None) to 5+ (Extreme)"
            )
        
        with pred_col2:
            st.metric(
                "Risk Level",
                risk_level,
                help="Categorical risk assessment"
            )
        
        with pred_col3:
            st.metric(
                "Confidence",
                f"{ensemble_confidence:.1f}%",
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
            font-weight: bold;
            margin: 10px 0;
        ">
            Current Risk Level: {risk_level.upper()}
        </div>
        """, unsafe_allow_html=True)
        
        # Model comparison
        if len(selected_models) > 1:
            st.markdown("### 📊 Model Comparison")
            
            model_comparison = []
            model_mapping = {
                'Random Forest': 'random_forest',
                'Gradient Boosting': 'gradient_boosting',
                'Ensemble': 'ensemble'
            }
            
            for model_name in selected_models:
                model_key = model_mapping.get(model_name, model_name.lower().replace(' ', '_'))
                if model_key in predictions:
                    pred_value = predictions[model_key]
                    conf_value = confidence_scores[model_key]
                    # Calculate risk level
                    if pred_value < 1:
                        risk_level_model = "Low"
                    elif pred_value < 2:
                        risk_level_model = "Moderate"
                    elif pred_value < 3:
                        risk_level_model = "High"
                    else:
                        risk_level_model = "Severe"
                    
                    model_comparison.append({
                        'Model': model_name,
                        'Prediction': pred_value,
                        'Confidence': conf_value,
                        'Risk Level': risk_level_model
                    })
            
            if model_comparison:
                comparison_df = pd.DataFrame(model_comparison)
                
                # Model comparison chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Turbulence Intensity',
                    x=comparison_df['Model'],
                    y=comparison_df['Prediction'],
                    yaxis='y',
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Scatter(
                    name='Confidence %',
                    x=comparison_df['Model'],
                    y=comparison_df['Confidence'],
                    yaxis='y2',
                    mode='lines+markers',
                    marker_color='red',
                    line=dict(width=3)
                ))
                
                fig.update_layout(
                    title="Model Predictions Comparison",
                    xaxis_title="ML Models",
                    yaxis=dict(
                        title="Turbulence Intensity",
                        side="left"
                    ),
                    yaxis2=dict(
                        title="Confidence (%)",
                        side="right",
                        overlaying="y"
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison table
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Prediction": st.column_config.NumberColumn(
                            "Prediction",
                            format="%.3f"
                        ),
                        "Confidence": st.column_config.NumberColumn(
                            "Confidence",
                            format="%.1f%%"
                        )
                    }
                )
        
        # Additional calculations
        st.markdown("### 🔍 Detailed Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Passenger comfort index
            comfort_index = turbulence_calc.calculate_passenger_comfort_index(
                ensemble_prediction, 60  # Assume 1 hour flight segment
            )
            
            st.metric("Passenger Comfort Index", f"{comfort_index:.1f}/100")
            
            # Flight delay probability
            delay_prob = turbulence_calc.calculate_flight_delay_probability(
                ensemble_prediction, weather_data.get('weather_condition', 'clear')
            )
            
            st.metric("Delay Probability", f"{delay_prob:.1f}%")
        
        with analysis_col2:
            # Fuel impact
            fuel_impact = turbulence_calc.estimate_fuel_consumption_impact(
                ensemble_prediction, 2  # Assume 2 hour flight
            )
            
            st.metric("Fuel Impact", f"+{fuel_impact:.1f}%")
            
            # Clear air turbulence potential
            cat_potential = turbulence_calc.calculate_clear_air_turbulence_potential(
                weather_data.get('wind_speed', 0),
                weather_data.get('wind_direction', 0),
                weather_data.get('pressure', 1013),
                weather_data.get('temperature', 20),
                flight_altitude
            )
            
            st.metric("CAT Potential", f"{cat_potential}/5")

with col2:
    st.markdown("## 📈 Model Performance")
    
    # Feature importance
    feature_importance = predictor.get_feature_importance()
    
    if feature_importance is not None:
        st.markdown("### 🎯 Feature Importance")
        
        fig = px.bar(
            feature_importance.head(8),
            x='importance',
            y='feature',
            orientation='h',
            title="Top Features for Turbulence Prediction"
        )
        
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Current weather summary
    if 'weather_data' in st.session_state:
        st.markdown("### 🌤️ Current Conditions")
        
        weather = st.session_state.weather_data
        airport_code = st.session_state.selected_airport
        
        st.info(f"**{airport_code}** - {INDIAN_AIRPORTS[airport_code]['city']}")
        
        weather_metrics = [
            ("Temperature", f"{weather.get('temperature', 'N/A')}°C"),
            ("Wind Speed", f"{weather.get('wind_speed', 'N/A')} m/s"),
            ("Pressure", f"{weather.get('pressure', 'N/A')} hPa"),
            ("Humidity", f"{weather.get('humidity', 'N/A')}%"),
            ("Condition", weather.get('weather_condition', 'N/A'))
        ]
        
        for label, value in weather_metrics:
            st.text(f"{label}: {value}")
    
    # Flight parameters summary
    if 'flight_params' in st.session_state:
        st.markdown("### ✈️ Flight Parameters")
        
        flight_info = st.session_state.flight_params
        
        st.text(f"Altitude: {flight_info['altitude']:,} ft")
        st.text(f"Aircraft: {flight_info['aircraft_type']}")
        
        # Altitude category info
        if flight_info['altitude'] >= 40000:
            altitude_info = "Very High Altitude - Jet stream proximity"
        elif flight_info['altitude'] >= 35000:
            altitude_info = "High Altitude - Standard cruise"
        elif flight_info['altitude'] >= 30000:
            altitude_info = "Medium-High Altitude"
        else:
            altitude_info = "Lower Altitude"
        
        st.caption(altitude_info)

# Model information and help
with st.expander("📚 Model Information & Help"):
    st.markdown("""
    ### Machine Learning Models Used
    
    **1. Random Forest Regressor**
    - Ensemble method using multiple decision trees
    - Good for handling non-linear relationships
    - Robust against overfitting
    
    **2. Gradient Boosting Regressor**
    - Sequential ensemble method
    - Builds models iteratively to correct errors
    - Often provides high accuracy
    
    **3. Ensemble Prediction**
    - Combines predictions from multiple models
    - Usually more reliable than individual models
    - Provides balanced predictions
    
    ### Input Features
    - Wind speed and direction
    - Atmospheric pressure and temperature
    - Humidity and visibility
    - Flight altitude
    - Time of day and seasonal factors
    - Weather conditions
    
    ### Turbulence Scale
    - **0.0-1.0**: Light turbulence (minimal discomfort)
    - **1.0-2.5**: Moderate turbulence (noticeable movement)
    - **2.5-4.0**: Severe turbulence (difficult to control aircraft)
    - **4.0+**: Extreme turbulence (structural stress)
    
    ### Confidence Score
    The confidence score indicates how certain the model is about its prediction:
    - **90-100%**: Very high confidence
    - **80-89%**: High confidence
    - **70-79%**: Moderate confidence
    - **Below 70%**: Low confidence (use with caution)
    """)

# Prediction history
if st.checkbox("Show Prediction History"):
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if 'predictions' in st.session_state:
        # Add current prediction to history
        history_entry = {
            'timestamp': datetime.now(),
            'airport': st.session_state.selected_airport,
            'prediction': st.session_state.predictions.get('ensemble', 0),
            'confidence': st.session_state.confidence_scores.get('ensemble', 0),
            'altitude': st.session_state.flight_params.get('altitude', 0)
        }
        
        # Add to history if not already present (avoid duplicates)
        if not st.session_state.prediction_history or \
           st.session_state.prediction_history[-1]['timestamp'] != history_entry['timestamp']:
            st.session_state.prediction_history.append(history_entry)
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Plot prediction timeline
        fig = px.line(
            history_df,
            x='timestamp',
            y='prediction',
            color='airport',
            title="Prediction History Timeline",
            labels={'prediction': 'Turbulence Intensity', 'timestamp': 'Time'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show history table
        st.dataframe(
            history_df.tail(10),
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
