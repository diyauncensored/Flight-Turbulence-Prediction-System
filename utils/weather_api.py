"""
Weather API integration for turbulence prediction
"""

import requests
import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

class WeatherAPI:
    def __init__(self):
        # Use API key from environment variable
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
    def get_current_weather(self, lat, lon):
        """Get current weather data for given coordinates"""
        # Check API key availability
        if not self.api_key:
            st.warning("⚠️ Weather API key not configured. Using demo mode.")
            return self._get_demo_weather(lat, lon)
        
        # Check cache
        cache_key = f"weather_{lat}_{lon}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                return cached_data
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data = {
                    "temperature": data["main"]["temp"],
                    "pressure": data["main"]["pressure"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data.get("wind", {}).get("speed", 0),
                    "wind_direction": data.get("wind", {}).get("deg", 0),
                    "visibility": data.get("visibility", 10000) / 1000,  # Convert to km
                    "weather_condition": data["weather"][0]["main"],
                    "description": data["weather"][0]["description"],
                    "timestamp": datetime.now()
                }
                
                # Cache the result
                self.cache[cache_key] = (weather_data, datetime.now())
                
                return weather_data
            else:
                st.error(f"Weather API Error: {response.status_code}")
                return self._get_demo_weather(lat, lon)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Weather API Request Failed: {str(e)}")
            return self._get_demo_weather(lat, lon)
        except Exception as e:
            st.error(f"Weather API Error: {str(e)}")
            return self._get_demo_weather(lat, lon)
    
    def _get_demo_weather(self, lat, lon):
        """Generate demo weather data as fallback"""
        import random
        import math
        
        # Generate realistic demo data based on location
        base_temp = 25 + 10 * math.sin(lat / 10)
        
        return {
            "temperature": base_temp + random.uniform(-5, 5),
            "pressure": 1013 + random.uniform(-10, 10),
            "humidity": 60 + random.uniform(-20, 20),
            "wind_speed": abs(random.gauss(10, 5)),
            "wind_direction": random.uniform(0, 360),
            "visibility": random.uniform(5, 15),
            "weather_condition": random.choice(["Clear", "Clouds", "Rain"]),
            "description": "demo data",
            "timestamp": datetime.now()
        }
    
    def get_forecast_data(self, lat, lon, hours=48):
        """Get forecast data for turbulence prediction"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "cnt": min(hours // 3, 40)  # API returns 3-hour intervals, max 40 intervals
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                forecast_list = []
                
                for item in data["list"]:
                    forecast_item = {
                        "datetime": datetime.fromtimestamp(item["dt"]),
                        "temperature": item["main"]["temp"],
                        "pressure": item["main"]["pressure"],
                        "humidity": item["main"]["humidity"],
                        "wind_speed": item.get("wind", {}).get("speed", 0),
                        "wind_direction": item.get("wind", {}).get("deg", 0),
                        "weather_condition": item["weather"][0]["main"],
                        "description": item["weather"][0]["description"]
                    }
                    forecast_list.append(forecast_item)
                
                return pd.DataFrame(forecast_list)
            else:
                st.error(f"Forecast API Error: {response.status_code}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            st.error(f"Forecast API Request Failed: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Forecast API Error: {str(e)}")
            return pd.DataFrame()
    
    def get_atmospheric_data(self, lat, lon):
        """Get detailed atmospheric data for turbulence analysis"""
        current_weather = self.get_current_weather(lat, lon)
        
        if current_weather:
            # Calculate additional atmospheric parameters
            atmospheric_data = current_weather.copy()
            
            # Calculate wind shear potential (simplified)
            wind_speed = atmospheric_data["wind_speed"]
            atmospheric_data["wind_shear_potential"] = self._calculate_wind_shear_potential(wind_speed)
            
            # Calculate atmospheric stability
            temp = atmospheric_data["temperature"]
            pressure = atmospheric_data["pressure"]
            atmospheric_data["atmospheric_stability"] = self._calculate_atmospheric_stability(temp, pressure)
            
            # Calculate turbulence intensity indicator
            atmospheric_data["turbulence_indicator"] = self._calculate_turbulence_indicator(atmospheric_data)
            
            return atmospheric_data
        
        return None
    
    def _calculate_wind_shear_potential(self, wind_speed):
        """Calculate wind shear potential based on wind speed"""
        if wind_speed < 5:
            return "Low"
        elif wind_speed < 15:
            return "Moderate"
        elif wind_speed < 25:
            return "High"
        else:
            return "Very High"
    
    def _calculate_atmospheric_stability(self, temperature, pressure):
        """Calculate atmospheric stability indicator"""
        # Simplified stability calculation
        stability_index = (temperature + 273.15) / pressure * 1000
        
        if stability_index < 2.8:
            return "Very Stable"
        elif stability_index < 3.2:
            return "Stable"
        elif stability_index < 3.6:
            return "Neutral"
        elif stability_index < 4.0:
            return "Unstable"
        else:
            return "Very Unstable"
    
    def _calculate_turbulence_indicator(self, weather_data):
        """Calculate overall turbulence risk indicator"""
        risk_score = 0
        
        # Wind speed contribution
        wind_speed = weather_data["wind_speed"]
        if wind_speed > 20:
            risk_score += 3
        elif wind_speed > 15:
            risk_score += 2
        elif wind_speed > 10:
            risk_score += 1
        
        # Weather condition contribution
        condition = weather_data["weather_condition"].lower()
        if condition in ["thunderstorm", "squall"]:
            risk_score += 4
        elif condition in ["rain", "drizzle"]:
            risk_score += 2
        elif condition in ["clouds", "mist"]:
            risk_score += 1
        
        # Pressure contribution (low pressure systems)
        pressure = weather_data["pressure"]
        if pressure < 1010:
            risk_score += 2
        elif pressure < 1015:
            risk_score += 1
        
        # Convert to risk level
        if risk_score >= 6:
            return "Severe"
        elif risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Moderate"
        else:
            return "Low"

# Singleton instance
weather_api = WeatherAPI()
