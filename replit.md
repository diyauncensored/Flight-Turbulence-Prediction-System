# Flight Turbulence Prediction System for Indian Airports

## Overview

A comprehensive machine learning-based turbulence prediction system for major Indian airports (Delhi, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata). The system integrates real-time weather data, multiple ML models, and historical analysis to provide accurate turbulence forecasting for flight safety and route planning.

Built with Streamlit for an interactive web interface, the system offers real-time predictions, route risk assessment, pilot reporting capabilities, deep learning-based forecasting using LSTM neural networks, automated alert system, and enhanced 8-level severity classification.

## Recent Changes (October 2025)

### Phase 2 Implementation Complete
- ✅ Real-time weather API integration with caching and error handling
- ✅ PostgreSQL database with demo mode fallback (session-state storage)
- ✅ Pilot reporting system with validation and persistence
- ✅ Automated alert system with configurable thresholds
- ✅ Enhanced 8-level turbulence severity classification (Smooth to Extreme)
- ✅ LSTM deep learning model with TensorFlow/Keras (graceful fallback when unavailable)

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit multi-page application
- **Visualization**: Plotly for interactive charts, maps, and 3D atmospheric models
- **Pages**: 12 dedicated pages for different functionalities (airport map, weather analysis, ML prediction, route assessment, historical analysis, real-time monitoring, atmospheric visualization, data upload, statistical dashboard, pilot reports, alert system, severity classification, deep learning prediction)
- **State Management**: Streamlit session state for caching data, predictions, and user interactions

### Backend Architecture
- **ML Models**: 
  - Random Forest Regressor for baseline predictions
  - Gradient Boosting Regressor for advanced forecasting
  - Ensemble approach combining multiple models
  - LSTM (Long Short-Term Memory) neural network for time-series prediction with TensorFlow/Keras (with fallback when unavailable)
- **Data Processing**: 
  - Synthetic training data generation (5000+ samples)
  - Feature engineering with weather parameters, flight data, and temporal features
  - StandardScaler for normalization
  - Richardson Number calculation for atmospheric stability
  - Eddy Dissipation Rate (EDR) calculation for turbulence intensity
- **Severity Classification**: 8-level turbulence scale (Smooth to Extreme) with detailed risk assessment, pilot action recommendations, and passenger impact analysis

### Data Storage
- **Primary Database**: PostgreSQL with psycopg2 connector (optional)
- **Demo Mode**: In-memory storage using Streamlit session state when DATABASE_URL not configured
- **Tables**: 
  - pilot_reports: Turbulence encounters reported by pilots
  - alerts: Automated turbulence alert system
  - encounters: Historical turbulence event tracking
- **Caching**: Weather API responses cached for 5 minutes to reduce API calls

### Weather Data Integration
- **API**: OpenWeatherMap API for current weather and forecasts
- **Parameters**: Temperature, pressure, humidity, wind speed/direction, visibility, cloud cover
- **Fallback**: Demo weather data generator when API key unavailable
- **Altitude Levels**: Multi-altitude analysis (20,000 - 45,000 feet)

### Core Features
1. **Turbulence Prediction Engine**: Multi-model ensemble combining Random Forest, Gradient Boosting, and LSTM with confidence scoring
2. **Route Risk Assessment**: Great circle distance calculation, waypoint-based turbulence analysis, and comprehensive route safety evaluation
3. **Alert System**: Automated threshold-based alerts (severe, moderate, light) with configurable sensitivity
4. **Real-time Monitoring**: Auto-refresh capabilities, prediction logging, and continuous airport monitoring
5. **Historical Analysis**: Time-series pattern detection, seasonal analysis, weather correlation, and risk trend identification
6. **Pilot Reporting**: Structured turbulence encounter reporting with severity classification and location tracking

### Geographic Data
- **Airports**: 6 major Indian airports with coordinates, elevation, and runway information
- **Routes**: Predefined flight routes between major city pairs
- **Distance Calculation**: Great circle distance using Haversine formula
- **Regional Analysis**: North, South, East, West, and All-India weather pattern analysis

## External Dependencies

### Third-party APIs
- **OpenWeatherMap API**: Current weather and forecast data (requires OPENWEATHERMAP_API_KEY environment variable)

### Python Libraries
- **Web Framework**: streamlit (multi-page application framework)
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (RandomForest, GradientBoosting, StandardScaler), joblib (model persistence)
- **Deep Learning**: tensorflow/keras (optional, LSTM model with graceful fallback)
- **Visualization**: plotly (express and graph_objects for interactive charts)
- **Database**: psycopg2 (PostgreSQL connector with RealDictCursor)
- **HTTP**: requests (weather API integration)

### Environment Variables
- `OPENWEATHERMAP_API_KEY`: API key for weather data (optional, falls back to demo mode)
- `DATABASE_URL`: PostgreSQL connection string (optional, falls back to session state storage)

### Model Persistence
- **Location**: `models/` directory
- **Files**: 
  - `lstm_turbulence.keras`: LSTM neural network model
  - `scaler.pkl`: Feature scaler for normalization
- **Format**: Keras native format for neural networks, pickle for scalers

### Data Flow
1. Weather API → Data Processing → Feature Engineering → ML Models → Predictions
2. User Input → Route Calculation → Waypoint Analysis → Risk Assessment
3. Pilot Reports → Database Storage → Historical Analysis → Statistical Insights
4. Real-time Monitoring → Alert System → Threshold Checking → Notification Generation