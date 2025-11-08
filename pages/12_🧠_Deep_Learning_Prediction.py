import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.lstm_model import lstm_predictor
from utils.airport_data import get_airports
from utils.weather_api import WeatherAPI
from utils.severity_classification import TurbulenceSeverityClassifier
from datetime import datetime

st.set_page_config(page_title="Deep Learning Prediction", page_icon="🧠", layout="wide")

st.title("🧠 LSTM Deep Learning Turbulence Prediction")
st.markdown("Advanced neural network model for high-accuracy turbulence forecasting")

# Initialize systems
weather_api = WeatherAPI()
classifier = TurbulenceSeverityClassifier()

# Check TensorFlow availability
from utils.lstm_model import TF_AVAILABLE
if not TF_AVAILABLE:
    st.warning("⚠️ TensorFlow not available. Using fallback calculation method. Install TensorFlow for full LSTM functionality.")

# Sidebar
st.sidebar.header("Model Configuration")

airports = get_airports()
airport_names = {f"{a['name']} ({a['code']})": a for a in airports}

selected = st.sidebar.selectbox("Select Airport", options=list(airport_names.keys()))
airport = airport_names[selected]

altitude = st.sidebar.slider(
    "Flight Altitude (feet)",
    min_value=20000,
    max_value=45000,
    value=35000,
    step=1000
)

# Model info
with st.sidebar.expander("ℹ️ About LSTM Model"):
    st.markdown("""
    **LSTM (Long Short-Term Memory)**
    
    - Deep neural network architecture
    - Learns temporal patterns in weather data
    - Processes sequences of atmospheric conditions
    - Provides uncertainty estimates
    - Higher accuracy than traditional ML models
    
    **Architecture:**
    - 3 LSTM layers (128→64→32 units)
    - Dropout regularization
    - Dense output layers
    - Trained on weather sequences
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Training", "Performance"])

with tab1:
    st.header("Neural Network Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate LSTM Prediction", type="primary", use_container_width=True):
            with st.spinner("Running deep learning model..."):
                try:
                    # Get current weather using lat/lon from airport data
                    weather = weather_api.get_current_weather(
                        airport['lat'],  # Using lat instead of coordinates
                        airport['lon']   # Using lon instead of coordinates
                    )
                    
                    if weather:
                        # Prepare flight parameters
                        flight_params = {
                            'altitude': altitude,
                            'aircraft_type': 'Medium'
                        }
                        
                        # Get prediction
                        prediction = lstm_predictor.predict(weather, flight_params)
                        
                        # Store in session state
                        st.session_state.lstm_prediction = prediction
                        st.session_state.lstm_weather = weather
                        st.session_state.lstm_airport = airport
                        
                        st.success("✅ Prediction generated successfully!")
                    else:
                        st.error("❌ Could not fetch weather data. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error generating prediction: {str(e)}")
    
    with col2:
        st.markdown("**Current Conditions**")
        if 'lstm_weather' in st.session_state:
            weather = st.session_state.lstm_weather
            st.metric("Wind Speed", f"{weather['wind_speed']:.1f} m/s")
            st.metric("Temperature", f"{weather['temperature']:.1f}°C")
            st.metric("Pressure", f"{weather['pressure']:.0f} hPa")
    
    # Display prediction results
    if 'lstm_prediction' in st.session_state:
        pred = st.session_state.lstm_prediction
        severity_info = classifier.get_severity_info(pred['turbulence_index'])
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Main prediction display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {severity_info['color']}; padding: 30px; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">{severity_info['icon']} {severity_info['level']}</h2>
                <h3 style="color: white; margin: 10px 0 0 0;">Index: {pred['turbulence_index']:.2f}/10</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Model Confidence",
                f"{pred['confidence']*100:.1f}%",
                help="Higher confidence indicates more reliable prediction"
            )
            st.metric(
                "Uncertainty (σ)",
                f"{pred['std_dev']:.2f}",
                help="Standard deviation of predictions"
            )
        
        with col3:
            st.markdown("**Model Type**")
            st.info(pred['model_type'])
            st.markdown("**Structural Concern**")
            st.progress(severity_info['structural_concern'] / 100,
                       text=f"{severity_info['structural_concern']}%")
        
        # Detailed information
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Severity Analysis")
            st.write(f"**Description:** {severity_info['description']}")
            st.write(f"**Passenger Impact:** {severity_info['passenger_impact']}")
        
        with col2:
            st.subheader("Pilot Actions Required")
            st.warning(severity_info['pilot_action'])
        
        # Confidence visualization
        st.markdown("---")
        st.subheader("Prediction Confidence Analysis")
        
        # Create confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred['confidence'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Training & Improvement")
    st.markdown("Pre-train the LSTM model with synthetic data for improved accuracy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.number_input(
            "Training Samples",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of synthetic samples to generate for training"
        )
        
        epochs = st.slider(
            "Training Epochs",
            min_value=10,
            max_value=200,
            value=50,
            help="Number of training iterations"
        )
    
    with col2:
        st.info("""
        **Training Process:**
        1. Generate synthetic weather sequences
        2. Calculate turbulence indices
        3. Train LSTM network
        4. Save trained model
        
        Training improves model accuracy over time.
        """)
    
    if st.button("🎓 Train Model", type="primary"):
        with st.spinner(f"Training LSTM model with {num_samples} samples..."):
            import time
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Generate training data
            status_text.text("Generating training data...")
            progress_bar.progress(20)
            X_train, y_train = lstm_predictor.generate_training_data(num_samples)
            
            # Normalize data
            status_text.text("Normalizing data...")
            progress_bar.progress(40)
            X_train_normalized = np.array([lstm_predictor.normalize_data(seq) for seq in X_train])
            
            # Train model
            status_text.text("Training neural network...")
            progress_bar.progress(60)
            history = lstm_predictor.train_on_new_data(
                X_train_normalized,
                y_train,
                epochs=epochs,
                batch_size=64
            )
            
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            st.success(f"✅ Model trained successfully with {num_samples} samples over {epochs} epochs")
            
            # Display training results
            st.subheader("Training History")
            
            # Create loss chart
            loss_df = pd.DataFrame({
                'Epoch': range(1, len(history.history['loss']) + 1),
                'Training Loss': history.history['loss'],
                'Validation Loss': history.history['val_loss']
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=loss_df['Epoch'],
                y=loss_df['Training Loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=loss_df['Epoch'],
                y=loss_df['Validation Loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Model Training Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss (MSE)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final metrics
            col1, col2 = st.columns(2)
            with col1:
                final_loss = history.history['loss'][-1]
                st.metric("Final Training Loss", f"{final_loss:.4f}")
            with col2:
                final_val_loss = history.history['val_loss'][-1]
                st.metric("Final Validation Loss", f"{final_val_loss:.4f}")

with tab3:
    st.header("Model Performance Analysis")
    
    st.markdown("""
    ### LSTM Neural Network Architecture
    
    The deep learning model uses a multi-layer LSTM architecture specifically designed 
    for time-series turbulence prediction:
    """)
    
    # Architecture visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Input Layer:**
        - 10 timesteps of weather history
        - 8 features per timestep:
          - Wind speed & direction
          - Temperature
          - Pressure
          - Humidity
          - Visibility
          - Altitude
          - Time of day
        """)
        
        st.markdown("""
        **LSTM Layers:**
        - Layer 1: 128 units + Dropout (30%)
        - Layer 2: 64 units + Dropout (30%)
        - Layer 3: 32 units + Dropout (20%)
        """)
    
    with col2:
        st.markdown("""
        **Dense Layers:**
        - Dense 1: 64 units (ReLU)
        - Dropout: 20%
        - Dense 2: 32 units (ReLU)
        
        **Output:**
        - Single value: Turbulence Index (0-10)
        - Activation: Linear
        """)
    
    # Model comparison
    st.markdown("---")
    st.subheader("Model Comparison")
    
    comparison_data = {
        'Model': ['Random Forest', 'Gradient Boosting', 'LSTM Neural Network'],
        'Type': ['Ensemble', 'Ensemble', 'Deep Learning'],
        'Training Time': ['Fast', 'Medium', 'Slow'],
        'Accuracy': ['Good', 'Good', 'Excellent'],
        'Temporal Learning': ['No', 'Limited', 'Yes'],
        'Uncertainty Estimation': ['Limited', 'Limited', 'Yes']
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Advantages
    st.markdown("---")
    st.subheader("LSTM Advantages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Strengths:**
        - Learns temporal patterns
        - Captures weather evolution
        - Provides uncertainty estimates
        - Handles sequential data
        - Better long-term predictions
        """)
    
    with col2:
        st.warning("""
        **⚠️ Considerations:**
        - Requires more training data
        - Longer training time
        - Higher computational cost
        - Needs historical sequences
        """)

st.markdown("---")
st.caption("LSTM model provides state-of-the-art turbulence prediction using deep learning")
