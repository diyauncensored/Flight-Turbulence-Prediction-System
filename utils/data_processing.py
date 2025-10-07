"""
Data processing utilities for turbulence prediction system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.processed_data = {}
        
    def process_weather_data(self, weather_data):
        """Process raw weather data for analysis"""
        if not weather_data:
            return None
            
        processed = weather_data.copy()
        
        # Add derived features
        processed['wind_speed_category'] = self._categorize_wind_speed(processed['wind_speed'])
        processed['pressure_category'] = self._categorize_pressure(processed['pressure'])
        processed['visibility_category'] = self._categorize_visibility(processed['visibility'])
        
        # Add risk indicators
        processed['weather_risk_score'] = self._calculate_weather_risk_score(processed)
        
        return processed
    
    def process_flight_data(self, flight_params):
        """Process flight parameters"""
        processed = flight_params.copy()
        
        # Add altitude category
        altitude = processed.get('altitude', 35000)
        processed['altitude_category'] = self._categorize_altitude(altitude)
        
        return processed
    
    def create_time_series_data(self, forecast_data):
        """Create time series data from forecast"""
        if forecast_data.empty:
            return pd.DataFrame()
            
        # Add time-based features
        forecast_data['hour'] = forecast_data['datetime'].dt.hour
        forecast_data['day_of_week'] = forecast_data['datetime'].dt.dayofweek
        forecast_data['is_weekend'] = forecast_data['day_of_week'].isin([5, 6])
        
        # Add weather risk scores
        forecast_data['risk_score'] = forecast_data.apply(
            lambda row: self._calculate_weather_risk_score(row), axis=1
        )
        
        return forecast_data
    
    def aggregate_historical_data(self, data, groupby_cols, agg_functions):
        """Aggregate historical data for analysis"""
        if data.empty:
            return pd.DataFrame()
            
        try:
            aggregated = data.groupby(groupby_cols).agg(agg_functions).reset_index()
            return aggregated
        except Exception as e:
            st.error(f"Error aggregating data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_turbulence_statistics(self, data, group_by='airport'):
        """Calculate turbulence statistics"""
        if data.empty:
            return {}
            
        stats = {}
        
        if group_by in data.columns:
            grouped = data.groupby(group_by)
            
            stats['mean_intensity'] = grouped['turbulence_intensity'].mean()
            stats['max_intensity'] = grouped['turbulence_intensity'].max()
            stats['frequency'] = grouped.size()
            stats['severe_incidents'] = grouped.apply(
                lambda x: (x['turbulence_intensity'] > 4).sum()
            )
        else:
            stats['mean_intensity'] = data['turbulence_intensity'].mean()
            stats['max_intensity'] = data['turbulence_intensity'].max()
            stats['frequency'] = len(data)
            stats['severe_incidents'] = (data['turbulence_intensity'] > 4).sum()
        
        return stats
    
    def _categorize_wind_speed(self, wind_speed):
        """Categorize wind speed"""
        if wind_speed < 5:
            return "Light"
        elif wind_speed < 15:
            return "Moderate"
        elif wind_speed < 25:
            return "Strong"
        else:
            return "Very Strong"
    
    def _categorize_pressure(self, pressure):
        """Categorize atmospheric pressure"""
        if pressure < 1000:
            return "Very Low"
        elif pressure < 1010:
            return "Low"
        elif pressure < 1020:
            return "Normal"
        else:
            return "High"
    
    def _categorize_visibility(self, visibility):
        """Categorize visibility"""
        if visibility < 1:
            return "Poor"
        elif visibility < 5:
            return "Moderate"
        elif visibility < 10:
            return "Good"
        else:
            return "Excellent"
    
    def _categorize_altitude(self, altitude):
        """Categorize flight altitude"""
        if altitude < 20000:
            return "Low"
        elif altitude < 30000:
            return "Medium"
        elif altitude < 40000:
            return "High"
        else:
            return "Very High"
    
    def _calculate_weather_risk_score(self, weather_data):
        """Calculate weather-based risk score"""
        risk_score = 0
        
        # Wind speed contribution
        wind_speed = weather_data.get('wind_speed', 0)
        if wind_speed > 25:
            risk_score += 4
        elif wind_speed > 15:
            risk_score += 3
        elif wind_speed > 10:
            risk_score += 2
        elif wind_speed > 5:
            risk_score += 1
        
        # Pressure contribution
        pressure = weather_data.get('pressure', 1013)
        if pressure < 1000:
            risk_score += 3
        elif pressure < 1010:
            risk_score += 2
        elif pressure < 1015:
            risk_score += 1
        
        # Weather condition contribution
        condition = str(weather_data.get('weather_condition', '')).lower()
        if 'thunderstorm' in condition:
            risk_score += 4
        elif 'rain' in condition:
            risk_score += 2
        elif 'cloud' in condition:
            risk_score += 1
        
        # Visibility contribution
        visibility = weather_data.get('visibility', 10)
        if visibility < 1:
            risk_score += 3
        elif visibility < 5:
            risk_score += 2
        elif visibility < 10:
            risk_score += 1
        
        return min(risk_score, 10)  # Cap at 10
    
    def generate_sample_historical_data(self, airport_code, days=30):
        """Generate sample historical data for demonstration"""
        np.random.seed(hash(airport_code) % 1000)  # Consistent seed per airport
        
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days * 8):  # 8 data points per day (every 3 hours)
            timestamp = start_date + timedelta(hours=i * 3)
            
            # Generate weather data with some seasonality
            base_temp = 25 + 10 * np.sin(2 * np.pi * timestamp.timetuple().tm_yday / 365)
            
            weather_data = {
                'timestamp': timestamp,
                'airport': airport_code,
                'temperature': base_temp + np.random.normal(0, 5),
                'pressure': 1013 + np.random.normal(0, 15),
                'humidity': np.random.uniform(30, 90),
                'wind_speed': np.random.exponential(8),
                'wind_direction': np.random.uniform(0, 360),
                'visibility': np.random.exponential(12),
                'weather_condition': np.random.choice([
                    'clear', 'clouds', 'rain', 'thunderstorm', 'mist'
                ], p=[0.4, 0.3, 0.15, 0.05, 0.1]),
                'altitude': 35000,  # Standard cruise altitude
                'turbulence_intensity': max(0, np.random.exponential(1.5))
            }
            
            data.append(weather_data)
        
        return pd.DataFrame(data)

# Global processor instance
data_processor = DataProcessor()
