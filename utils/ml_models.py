"""
Machine Learning models for turbulence prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import streamlit as st

class TurbulencePredictionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_names = [
            'wind_speed', 'wind_direction', 'temperature', 'pressure', 
            'humidity', 'visibility', 'altitude', 'time_of_day', 
            'season', 'weather_condition_encoded'
        ]
        
    def generate_synthetic_training_data(self, n_samples=5000):
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        
        data = []
        
        for _ in range(n_samples):
            # Weather parameters
            wind_speed = np.random.exponential(10)  # Average wind speed
            wind_direction = np.random.uniform(0, 360)
            temperature = np.random.normal(15, 20)  # Temperature in Celsius
            pressure = np.random.normal(1013, 20)  # Pressure in hPa
            humidity = np.random.uniform(20, 95)  # Humidity percentage
            visibility = np.random.exponential(10)  # Visibility in km
            altitude = np.random.choice([25000, 30000, 35000, 37000, 39000, 41000])  # Flight altitude
            
            # Time factors
            time_of_day = np.random.uniform(0, 24)  # Hour of day
            season = np.random.choice([0, 1, 2, 3])  # 0=Spring, 1=Summer, 2=Monsoon, 3=Winter
            
            # Weather conditions (encoded)
            weather_conditions = ['clear', 'clouds', 'rain', 'thunderstorm', 'mist', 'fog']
            weather_condition = np.random.choice(range(len(weather_conditions)))
            
            # Calculate turbulence intensity (target variable)
            # This is a simplified model based on meteorological factors
            turbulence_base = 0
            
            # Wind speed effect
            turbulence_base += wind_speed * 0.1
            
            # Pressure effect (low pressure = more turbulence)
            if pressure < 1000:
                turbulence_base += 2
            elif pressure < 1010:
                turbulence_base += 1
            
            # Weather condition effect
            if weather_condition in [3, 5]:  # thunderstorm, fog
                turbulence_base += 3
            elif weather_condition in [2, 4]:  # rain, mist
                turbulence_base += 1.5
            elif weather_condition == 1:  # clouds
                turbulence_base += 0.5
            
            # Altitude effect (jet streams around 35000-40000 ft)
            if 35000 <= altitude <= 40000:
                turbulence_base += 1
            
            # Season effect (monsoon season has more turbulence)
            if season == 2:  # Monsoon
                turbulence_base += 1
            
            # Time of day effect (afternoon thermal activity)
            if 12 <= time_of_day <= 16:
                turbulence_base += 0.5
            
            # Add some randomness
            turbulence_base += np.random.normal(0, 0.5)
            
            # Ensure non-negative values
            turbulence_intensity = max(0, turbulence_base)
            
            data.append({
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'temperature': temperature,
                'pressure': pressure,
                'humidity': humidity,
                'visibility': visibility,
                'altitude': altitude,
                'time_of_day': time_of_day,
                'season': season,
                'weather_condition_encoded': weather_condition,
                'turbulence_intensity': turbulence_intensity
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, data):
        """Prepare features for model training/prediction"""
        features = data[self.feature_names].copy()
        return features
    
    def train_models(self):
        """Train multiple ML models for turbulence prediction"""
        st.info("Generating training data and training models...")
        
        # Generate synthetic training data
        training_data = self.generate_synthetic_training_data()
        
        # Prepare features and target
        X = self.prepare_features(training_data)
        y = training_data['turbulence_intensity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        self.models['gradient_boosting'].fit(X_train_scaled, y_train)
        
        # Evaluate models
        model_performance = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            model_performance[name] = {'mse': mse, 'r2': r2}
        
        self.is_trained = True
        st.success("Models trained successfully!")
        
        return model_performance
    
    def predict_turbulence(self, weather_data, flight_params):
        """Predict turbulence intensity based on weather and flight parameters"""
        if not self.is_trained:
            self.train_models()
        
        # Prepare input features
        features = self._prepare_prediction_features(weather_data, flight_params)
        features_scaled = self.scalers['standard'].transform([features])
        
        predictions = {}
        confidence_scores = {}
        
        for name, model in self.models.items():
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence score based on model performance
            # This is a simplified confidence calculation
            confidence = min(100, max(0, 100 - (prediction * 10)))
            
            predictions[name] = max(0, prediction)
            confidence_scores[name] = confidence
        
        # Ensemble prediction (average of models)
        ensemble_prediction = np.mean(list(predictions.values()))
        ensemble_confidence = np.mean(list(confidence_scores.values()))
        
        predictions['ensemble'] = ensemble_prediction
        confidence_scores['ensemble'] = ensemble_confidence
        
        return predictions, confidence_scores
    
    def _prepare_prediction_features(self, weather_data, flight_params):
        """Convert weather data and flight parameters to model features"""
        # Map weather conditions to encoded values
        weather_condition_map = {
            'clear': 0, 'clouds': 1, 'rain': 2, 'thunderstorm': 3, 'mist': 4, 'fog': 5
        }
        
        # Get current time features
        now = datetime.now()
        time_of_day = now.hour + now.minute / 60.0
        
        # Determine season (simplified for India)
        month = now.month
        if month in [3, 4, 5]:
            season = 0  # Spring/Summer
        elif month in [6, 7, 8, 9]:
            season = 2  # Monsoon
        elif month in [10, 11]:
            season = 1  # Post-monsoon
        else:
            season = 3  # Winter
        
        # Extract weather condition
        weather_condition = weather_data.get('weather_condition', 'clear').lower()
        weather_condition_encoded = weather_condition_map.get(weather_condition, 0)
        
        features = [
            weather_data.get('wind_speed', 0),
            weather_data.get('wind_direction', 0),
            weather_data.get('temperature', 20),
            weather_data.get('pressure', 1013),
            weather_data.get('humidity', 50),
            weather_data.get('visibility', 10),
            flight_params.get('altitude', 35000),
            time_of_day,
            season,
            weather_condition_encoded
        ]
        
        return features
    
    def get_turbulence_risk_level(self, intensity):
        """Convert turbulence intensity to risk level"""
        if intensity < 1:
            return "Low", "green"
        elif intensity < 2.5:
            return "Moderate", "yellow"
        elif intensity < 4:
            return "High", "orange"
        else:
            return "Severe", "red"
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance for model interpretation"""
        if not self.is_trained or model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_data = {
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }
            return pd.DataFrame(importance_data).sort_values('importance', ascending=False)
        
        return None

# Global model instance
turbulence_model = TurbulencePredictionModel()
