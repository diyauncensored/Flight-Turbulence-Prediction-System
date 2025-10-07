import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import json
from utils.ml_models import turbulence_model
from utils.data_processing import data_processor
from utils.turbulence_calculator import turbulence_calc

st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")

st.title("📤 Data Upload and Custom Analysis")
st.markdown("Upload custom flight parameters and historical turbulence data for analysis")

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Sidebar for upload options
st.sidebar.header("Upload Options")

upload_type = st.sidebar.selectbox(
    "Data Type",
    ["Flight Parameters", "Historical Turbulence", "Weather Data", "Custom Dataset", "Pilot Reports"]
)

# File format selection
file_format = st.sidebar.selectbox(
    "File Format",
    ["CSV", "Excel", "JSON", "Text"]
)

# Analysis options
st.sidebar.subheader("Analysis Options")
perform_prediction = st.sidebar.checkbox("Generate Turbulence Predictions", value=True)
validate_data = st.sidebar.checkbox("Validate Data Quality", value=True)
generate_statistics = st.sidebar.checkbox("Generate Statistics", value=True)
create_visualizations = st.sidebar.checkbox("Create Visualizations", value=True)

# Main content based on upload type
if upload_type == "Flight Parameters":
    st.markdown("## ✈️ Flight Parameters Upload")
    st.info("Upload flight data including aircraft type, route, altitude, speed, and other flight parameters.")
    
    # Data format explanation
    with st.expander("📋 Required Data Format"):
        st.markdown("""
        **Required Columns:**
        - `flight_id`: Unique flight identifier
        - `aircraft_type`: Aircraft category (Light/Medium/Heavy/Super Heavy)
        - `origin_airport`: ICAO code of departure airport
        - `destination_airport`: ICAO code of arrival airport
        - `cruise_altitude`: Cruise altitude in feet
        - `flight_duration`: Duration in hours
        - `departure_time`: Departure timestamp
        
        **Optional Columns:**
        - `aircraft_weight`: Aircraft weight in kg
        - `fuel_load`: Fuel load in kg
        - `passenger_count`: Number of passengers
        - `cargo_weight`: Cargo weight in kg
        - `route_waypoints`: JSON string of waypoints
        """)
        
        # Sample data
        sample_data = pd.DataFrame({
            'flight_id': ['AI101', 'SG202', 'UK303'],
            'aircraft_type': ['Heavy', 'Medium', 'Heavy'],
            'origin_airport': ['DEL', 'BOM', 'BLR'],
            'destination_airport': ['BOM', 'BLR', 'HYD'],
            'cruise_altitude': [37000, 35000, 39000],
            'flight_duration': [2.5, 1.8, 1.2],
            'departure_time': ['2024-10-07 08:00:00', '2024-10-07 10:30:00', '2024-10-07 14:15:00']
        })
        
        st.markdown("**Sample Data:**")
        st.dataframe(sample_data)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Flight Parameters File",
        type=['csv', 'xlsx', 'json', 'txt'],
        help="Upload your flight parameters data file"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on format
            if file_format == "CSV" or uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif file_format == "Excel" or uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif file_format == "JSON" or uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} flight records")
            
            # Store in session state
            st.session_state.uploaded_data['flight_parameters'] = df
            
            # Display data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data validation
            if validate_data:
                st.markdown("### ✅ Data Validation")
                
                validation_results = []
                
                # Check required columns
                required_cols = ['flight_id', 'aircraft_type', 'origin_airport', 'destination_airport', 'cruise_altitude']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    validation_results.append(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    validation_results.append("✅ All required columns present")
                
                # Check data types and ranges
                if 'cruise_altitude' in df.columns:
                    invalid_altitudes = df[(df['cruise_altitude'] < 10000) | (df['cruise_altitude'] > 50000)]
                    if not invalid_altitudes.empty:
                        validation_results.append(f"⚠️ {len(invalid_altitudes)} records with invalid altitudes")
                    else:
                        validation_results.append("✅ Altitude values within valid range")
                
                # Check for duplicates
                duplicates = df.duplicated(subset=['flight_id']).sum()
                if duplicates > 0:
                    validation_results.append(f"⚠️ {duplicates} duplicate flight IDs found")
                else:
                    validation_results.append("✅ No duplicate flight IDs")
                
                for result in validation_results:
                    if "❌" in result:
                        st.error(result)
                    elif "⚠️" in result:
                        st.warning(result)
                    else:
                        st.success(result)
            
            # Generate predictions
            if perform_prediction and 'cruise_altitude' in df.columns:
                st.markdown("### 🤖 Turbulence Predictions")
                
                with st.spinner("Generating turbulence predictions..."):
                    predictions_data = []
                    
                    for _, row in df.iterrows():
                        # Simulate weather data (in real application, would fetch actual data)
                        weather_data = {
                            'temperature': 20,
                            'wind_speed': 15,
                            'wind_direction': 180,
                            'pressure': 1013,
                            'humidity': 60,
                            'visibility': 10,
                            'weather_condition': 'clear'
                        }
                        
                        flight_params = {
                            'altitude': row.get('cruise_altitude', 35000),
                            'aircraft_type': row.get('aircraft_type', 'Heavy')
                        }
                        
                        predictions, confidence = turbulence_model.predict_turbulence(
                            weather_data, flight_params
                        )
                        
                        turbulence_intensity = predictions.get('ensemble', 0)
                        risk_level, _ = turbulence_model.get_turbulence_risk_level(turbulence_intensity)
                        
                        predictions_data.append({
                            'flight_id': row.get('flight_id', f"Flight_{len(predictions_data)+1}"),
                            'turbulence_intensity': turbulence_intensity,
                            'risk_level': risk_level,
                            'confidence': confidence.get('ensemble', 0)
                        })
                    
                    predictions_df = pd.DataFrame(predictions_data)
                    
                    # Display predictions
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Predictions summary
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_turbulence = predictions_df['turbulence_intensity'].mean()
                        st.metric("Average Turbulence", f"{avg_turbulence:.3f}")
                    
                    with col2:
                        high_risk_count = len(predictions_df[predictions_df['risk_level'].isin(['High', 'Severe'])])
                        st.metric("High Risk Flights", high_risk_count)
                    
                    with col3:
                        avg_confidence = predictions_df['confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    
                    # Risk distribution
                    risk_dist = predictions_df['risk_level'].value_counts()
                    fig_risk = px.pie(
                        values=risk_dist.values,
                        names=risk_dist.index,
                        title="Flight Risk Level Distribution",
                        color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Severe': 'red'}
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please check the file format and try again.")

elif upload_type == "Historical Turbulence":
    st.markdown("## 📊 Historical Turbulence Data Upload")
    st.info("Upload historical turbulence encounter data for analysis and model improvement.")
    
    # Data format explanation
    with st.expander("📋 Required Data Format"):
        st.markdown("""
        **Required Columns:**
        - `timestamp`: Date and time of turbulence encounter
        - `airport_code`: Airport code (DEL, BOM, BLR, etc.)
        - `altitude`: Altitude in feet
        - `turbulence_intensity`: Numerical intensity (0-5 scale)
        - `latitude`: Latitude coordinates
        - `longitude`: Longitude coordinates
        
        **Optional Columns:**
        - `aircraft_type`: Type of aircraft
        - `weather_condition`: Weather conditions
        - `pilot_report`: Pilot description
        - `duration_minutes`: Duration of turbulence
        - `severity_category`: Light/Moderate/Severe/Extreme
        """)
    
    uploaded_file = st.file_uploader(
        "Upload Historical Turbulence Data",
        type=['csv', 'xlsx', 'json'],
        help="Upload your historical turbulence data"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} turbulence records")
            
            # Store in session state
            st.session_state.uploaded_data['historical_turbulence'] = df
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if generate_statistics:
                st.markdown("### 📈 Statistical Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Basic statistics
                    if 'turbulence_intensity' in df.columns:
                        st.subheader("Turbulence Intensity Statistics")
                        st.write(df['turbulence_intensity'].describe())
                        
                        # Intensity distribution
                        fig_hist = px.histogram(
                            df,
                            x='turbulence_intensity',
                            title="Turbulence Intensity Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Temporal analysis
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df['hour'] = df['timestamp'].dt.hour
                        
                        hourly_counts = df.groupby('hour').size()
                        
                        fig_hourly = px.bar(
                            x=hourly_counts.index,
                            y=hourly_counts.values,
                            title="Turbulence Reports by Hour",
                            labels={'x': 'Hour of Day', 'y': 'Number of Reports'}
                        )
                        st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Altitude analysis
                if 'altitude' in df.columns and 'turbulence_intensity' in df.columns:
                    st.subheader("Altitude vs Turbulence Analysis")
                    
                    fig_altitude = px.scatter(
                        df,
                        x='altitude',
                        y='turbulence_intensity',
                        title="Turbulence Intensity vs Altitude",
                        labels={'altitude': 'Altitude (ft)', 'turbulence_intensity': 'Turbulence Intensity'}
                    )
                    st.plotly_chart(fig_altitude, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

elif upload_type == "Weather Data":
    st.markdown("## 🌤️ Weather Data Upload")
    st.info("Upload weather observation data for correlation analysis with turbulence.")
    
    with st.expander("📋 Required Data Format"):
        st.markdown("""
        **Required Columns:**
        - `timestamp`: Observation timestamp
        - `station_id`: Weather station identifier
        - `latitude`: Station latitude
        - `longitude`: Station longitude
        - `temperature`: Temperature in Celsius
        - `pressure`: Atmospheric pressure in hPa
        - `wind_speed`: Wind speed in m/s
        - `wind_direction`: Wind direction in degrees
        
        **Optional Columns:**
        - `humidity`: Relative humidity percentage
        - `visibility`: Visibility in km
        - `weather_condition`: Weather description
        - `cloud_cover`: Cloud cover percentage
        """)
    
    uploaded_file = st.file_uploader(
        "Upload Weather Data",
        type=['csv', 'xlsx', 'json'],
        help="Upload weather observation data"
    )
    
    if uploaded_file is not None:
        try:
            # Read weather data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} weather observations")
            st.session_state.uploaded_data['weather_data'] = df
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if create_visualizations:
                st.markdown("### 🎨 Weather Visualizations")
                
                # Weather parameters correlation
                if all(col in df.columns for col in ['temperature', 'pressure', 'wind_speed', 'humidity']):
                    weather_cols = ['temperature', 'pressure', 'wind_speed', 'humidity']
                    corr_matrix = df[weather_cols].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        title="Weather Parameters Correlation",
                        aspect="auto",
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Time series of weather parameters
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    if 'temperature' in df.columns:
                        fig_temp = px.line(
                            df.sort_values('timestamp'),
                            x='timestamp',
                            y='temperature',
                            title="Temperature Time Series"
                        )
                        st.plotly_chart(fig_temp, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error reading weather file: {str(e)}")

elif upload_type == "Pilot Reports":
    st.markdown("## 👨‍✈️ Pilot Reports Upload")
    st.info("Upload pilot turbulence reports (PIREPs) for analysis and validation.")
    
    with st.expander("📋 Required Data Format"):
        st.markdown("""
        **Required Columns:**
        - `report_id`: Unique report identifier
        - `timestamp`: Report timestamp
        - `aircraft_type`: Type of reporting aircraft
        - `altitude`: Flight altitude in feet
        - `location`: Location description or coordinates
        - `turbulence_type`: Type of turbulence (CAT, convective, etc.)
        - `intensity`: Intensity description (light, moderate, severe)
        - `pilot_report`: Detailed pilot description
        
        **Optional Columns:**
        - `duration`: Duration of turbulence encounter
        - `weather_conditions`: Associated weather
        - `frequency`: VHF frequency used for report
        - `flight_phase`: Phase of flight (cruise, descent, etc.)
        """)
    
    uploaded_file = st.file_uploader(
        "Upload Pilot Reports",
        type=['csv', 'xlsx', 'json', 'txt'],
        help="Upload pilot turbulence reports"
    )
    
    if uploaded_file is not None:
        try:
            # Read pilot reports
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} pilot reports")
            st.session_state.uploaded_data['pilot_reports'] = df
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if generate_statistics:
                st.markdown("### 📊 Report Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Intensity distribution
                    if 'intensity' in df.columns:
                        intensity_counts = df['intensity'].value_counts()
                        fig_intensity = px.pie(
                            values=intensity_counts.values,
                            names=intensity_counts.index,
                            title="Reported Turbulence Intensity Distribution"
                        )
                        st.plotly_chart(fig_intensity, use_container_width=True)
                
                with col2:
                    # Turbulence type distribution
                    if 'turbulence_type' in df.columns:
                        type_counts = df['turbulence_type'].value_counts()
                        fig_type = px.bar(
                            x=type_counts.index,
                            y=type_counts.values,
                            title="Turbulence Type Distribution"
                        )
                        st.plotly_chart(fig_type, use_container_width=True)
                
                # Altitude analysis
                if 'altitude' in df.columns and 'intensity' in df.columns:
                    fig_alt_intensity = px.box(
                        df,
                        x='intensity',
                        y='altitude',
                        title="Altitude Distribution by Turbulence Intensity"
                    )
                    st.plotly_chart(fig_alt_intensity, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error reading pilot reports: {str(e)}")

elif upload_type == "Custom Dataset":
    st.markdown("## 🔧 Custom Dataset Upload")
    st.info("Upload any custom turbulence-related dataset for analysis.")
    
    uploaded_file = st.file_uploader(
        "Upload Custom Dataset",
        type=['csv', 'xlsx', 'json', 'txt'],
        help="Upload your custom dataset"
    )
    
    if uploaded_file is not None:
        try:
            # Attempt to read file with different methods
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                # Try to read as CSV by default
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} records with {len(df.columns)} columns")
            st.session_state.uploaded_data['custom_dataset'] = df
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            st.markdown("### ℹ️ Dataset Information")
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write("**Dataset Shape:**", df.shape)
                st.write("**Column Names:**")
                st.write(list(df.columns))
            
            with info_col2:
                st.write("**Data Types:**")
                st.write(df.dtypes)
                
                st.write("**Missing Values:**")
                missing_values = df.isnull().sum()
                st.write(missing_values[missing_values > 0])
            
            # Basic analysis
            if create_visualizations:
                st.markdown("### 🎨 Basic Visualizations")
                
                # Select numeric columns for visualization
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_columns) > 1:
                    # Correlation matrix
                    corr_matrix = df[numeric_columns].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        title="Correlation Matrix",
                        aspect="auto",
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Distribution plots for numeric columns
                for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                    fig_dist = px.histogram(
                        df,
                        x=col,
                        title=f"Distribution of {col}",
                        nbins=30
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error reading custom dataset: {str(e)}")
            st.info("Please ensure the file is in the correct format (CSV, Excel, or JSON)")

# Data management section
st.markdown("## 💾 Data Management")

if st.session_state.uploaded_data:
    st.markdown("### 📚 Uploaded Datasets")
    
    for data_type, data in st.session_state.uploaded_data.items():
        with st.expander(f"📊 {data_type.replace('_', ' ').title()} ({len(data)} records)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Shape:** {data.shape}")
                st.write(f"**Columns:** {len(data.columns)}")
            
            with col2:
                # Download processed data
                csv_data = data.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name=f"{data_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Remove dataset
                if st.button(f"🗑️ Remove", key=f"remove_{data_type}"):
                    del st.session_state.uploaded_data[data_type]
                    st.rerun()
            
            # Show sample data
            st.dataframe(data.head(3), use_container_width=True)

# Analysis tools
if st.session_state.uploaded_data:
    st.markdown("## 🔍 Advanced Analysis Tools")
    
    analysis_tool = st.selectbox(
        "Select Analysis Tool",
        ["Data Comparison", "Correlation Analysis", "Trend Analysis", "Quality Assessment"]
    )
    
    if analysis_tool == "Data Comparison" and len(st.session_state.uploaded_data) >= 2:
        st.markdown("### ⚖️ Dataset Comparison")
        
        dataset_names = list(st.session_state.uploaded_data.keys())
        dataset1 = st.selectbox("First Dataset", dataset_names, key="comp1")
        dataset2 = st.selectbox("Second Dataset", dataset_names, key="comp2")
        
        if dataset1 != dataset2:
            df1 = st.session_state.uploaded_data[dataset1]
            df2 = st.session_state.uploaded_data[dataset2]
            
            # Compare basic statistics
            comparison_data = {
                'Metric': ['Records', 'Columns', 'Numeric Columns', 'Missing Values'],
                dataset1: [len(df1), len(df1.columns), len(df1.select_dtypes(include=[np.number]).columns), df1.isnull().sum().sum()],
                dataset2: [len(df2), len(df2.columns), len(df2.select_dtypes(include=[np.number]).columns), df2.isnull().sum().sum()]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    elif analysis_tool == "Quality Assessment":
        st.markdown("### ✅ Data Quality Assessment")
        
        selected_dataset = st.selectbox(
            "Select Dataset for Quality Check",
            list(st.session_state.uploaded_data.keys())
        )
        
        df = st.session_state.uploaded_data[selected_dataset]
        
        # Quality metrics
        quality_metrics = {
            'Total Records': len(df),
            'Complete Records': len(df.dropna()),
            'Completeness %': (len(df.dropna()) / len(df)) * 100,
            'Duplicate Records': df.duplicated().sum(),
            'Unique Records': len(df.drop_duplicates()),
            'Columns with Missing Data': (df.isnull().sum() > 0).sum()
        }
        
        col1, col2, col3 = st.columns(3)
        
        metrics_items = list(quality_metrics.items())
        for i, (metric, value) in enumerate(metrics_items):
            col_idx = i % 3
            if col_idx == 0:
                col1.metric(metric, f"{value:.1f}" if isinstance(value, float) else str(value))
            elif col_idx == 1:
                col2.metric(metric, f"{value:.1f}" if isinstance(value, float) else str(value))
            else:
                col3.metric(metric, f"{value:.1f}" if isinstance(value, float) else str(value))
        
        # Data quality visualization
        if df.isnull().sum().sum() > 0:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Data by Column",
                labels={'x': 'Column', 'y': 'Missing Values Count'}
            )
            st.plotly_chart(fig_missing, use_container_width=True)

else:
    st.info("Upload data files to begin analysis. Use the sidebar to select data type and format.")

# Help section
with st.expander("❓ Help & Documentation"):
    st.markdown("""
    ## How to Use the Data Upload System
    
    ### 1. Select Data Type
    Choose the type of data you want to upload from the sidebar:
    - **Flight Parameters**: Aircraft and flight route information
    - **Historical Turbulence**: Past turbulence encounter records
    - **Weather Data**: Meteorological observations
    - **Pilot Reports**: PIREP turbulence reports
    - **Custom Dataset**: Any other relevant data
    
    ### 2. Choose File Format
    Supported formats:
    - **CSV**: Comma-separated values (recommended)
    - **Excel**: .xlsx or .xls files
    - **JSON**: JavaScript Object Notation
    - **Text**: Tab or other delimited text files
    
    ### 3. Upload and Analyze
    - Upload your file using the file uploader
    - Review data validation results
    - Generate predictions and visualizations
    - Download processed results
    
    ### 4. Data Requirements
    - Ensure your data includes required columns as specified
    - Use consistent date/time formats (YYYY-MM-DD HH:MM:SS)
    - Remove or handle missing values appropriately
    - Verify coordinate formats (decimal degrees)
    
    ### 5. Troubleshooting
    - **File Reading Errors**: Check file format and encoding
    - **Missing Columns**: Verify required columns are present
    - **Data Validation Issues**: Review data ranges and formats
    - **Large Files**: Consider splitting large datasets
    """)
