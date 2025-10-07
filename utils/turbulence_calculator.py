"""
Turbulence calculation utilities
"""

import numpy as np
import math
from datetime import datetime

class TurbulenceCalculator:
    def __init__(self):
        self.severity_thresholds = {
            'light': (0, 1.0),
            'moderate': (1.0, 2.5),
            'severe': (2.5, 4.0),
            'extreme': (4.0, float('inf'))
        }
    
    def calculate_richardson_number(self, wind_shear, temperature_gradient, altitude_diff=1000):
        """
        Calculate Richardson Number for atmospheric stability
        Ri = (g/T) * (dT/dz) / (du/dz)²
        """
        if wind_shear == 0:
            return float('inf')  # Very stable
        
        g = 9.81  # Gravity
        T = 288.15  # Reference temperature (K)
        
        # Convert temperature gradient per altitude difference
        dt_dz = temperature_gradient / altitude_diff
        du_dz = wind_shear / altitude_diff
        
        ri = (g / T) * dt_dz / (du_dz ** 2)
        return ri
    
    def calculate_eddy_dissipation_rate(self, wind_speed, wind_shear, atmospheric_stability):
        """
        Calculate Eddy Dissipation Rate (EDR) - a measure of turbulence intensity
        """
        # Simplified EDR calculation
        base_edr = 0.01 * (wind_speed ** 0.5) * (wind_shear ** 1.5)
        
        # Adjust for atmospheric stability
        stability_factor = {
            'Very Stable': 0.5,
            'Stable': 0.7,
            'Neutral': 1.0,
            'Unstable': 1.3,
            'Very Unstable': 1.5
        }.get(atmospheric_stability, 1.0)
        
        edr = base_edr * stability_factor
        return max(0, edr)
    
    def calculate_clear_air_turbulence_potential(self, wind_speed, wind_direction, 
                                                pressure, temperature, altitude):
        """
        Calculate Clear Air Turbulence (CAT) potential
        """
        cat_score = 0
        
        # High altitude factor (CAT more common above 20,000 ft)
        if altitude > 20000:
            cat_score += 1
            if altitude > 35000:
                cat_score += 1
        
        # Wind speed factor
        if wind_speed > 30:  # Strong winds
            cat_score += 2
        elif wind_speed > 20:
            cat_score += 1
        
        # Jet stream proximity (simplified check)
        if 30000 <= altitude <= 45000 and wind_speed > 25:
            cat_score += 2  # Likely near jet stream
        
        # Temperature and pressure gradients (simplified)
        # In reality, you'd need vertical profiles
        if pressure < 300:  # High altitude low pressure
            cat_score += 1
        
        return min(cat_score, 5)  # Max score of 5
    
    def calculate_convective_turbulence_potential(self, temperature, humidity, 
                                                  pressure, time_of_day, season):
        """
        Calculate convective turbulence potential
        """
        conv_score = 0
        
        # Temperature factor (higher temps increase convection)
        if temperature > 30:
            conv_score += 2
        elif temperature > 25:
            conv_score += 1
        
        # Humidity factor (high humidity supports convection)
        if humidity > 80:
            conv_score += 2
        elif humidity > 60:
            conv_score += 1
        
        # Low pressure systems
        if pressure < 1010:
            conv_score += 1
        
        # Time of day (afternoon heating)
        if 12 <= time_of_day <= 17:
            conv_score += 1
        
        # Season factor (monsoon season in India)
        if season == 2:  # Monsoon
            conv_score += 2
        
        return min(conv_score, 6)  # Max score of 6
    
    def calculate_mountain_wave_turbulence(self, wind_speed, wind_direction, 
                                         terrain_height, flight_altitude):
        """
        Calculate mountain wave turbulence potential
        """
        if flight_altitude < terrain_height + 5000:
            # Low-level turbulence near terrain
            mwt_score = min(wind_speed * 0.1, 3)
        elif flight_altitude < terrain_height + 15000:
            # Lee wave turbulence
            if wind_speed > 15:
                mwt_score = 2
            else:
                mwt_score = 1
        else:
            # High-level mountain wave effects
            if wind_speed > 25:
                mwt_score = 1
            else:
                mwt_score = 0
        
        return mwt_score
    
    def calculate_wake_turbulence_risk(self, aircraft_separation, aircraft_weights, 
                                     wind_speed, wind_direction):
        """
        Calculate wake turbulence risk from other aircraft
        """
        if aircraft_separation is None or aircraft_weights is None:
            return 0
        
        # Weight category difference
        weight_diff = max(aircraft_weights) - min(aircraft_weights)
        
        # Base risk from weight difference
        if weight_diff > 100000:  # Heavy vs light aircraft
            wake_risk = 3
        elif weight_diff > 50000:
            wake_risk = 2
        else:
            wake_risk = 1
        
        # Separation factor
        if aircraft_separation < 3:  # nm
            wake_risk += 2
        elif aircraft_separation < 5:
            wake_risk += 1
        
        # Wind factor (crosswinds help dissipate wake)
        if wind_speed > 10:
            wake_risk -= 1
        
        return max(0, wake_risk)
    
    def get_turbulence_severity(self, intensity):
        """
        Convert turbulence intensity to severity category
        """
        for severity, (min_val, max_val) in self.severity_thresholds.items():
            if min_val <= intensity < max_val:
                return severity
        return 'extreme'
    
    def get_severity_color(self, severity):
        """
        Get color code for severity level
        """
        colors = {
            'light': 'green',
            'moderate': 'yellow',
            'severe': 'orange',
            'extreme': 'red'
        }
        return colors.get(severity, 'gray')
    
    def calculate_passenger_comfort_index(self, turbulence_intensity, duration_minutes):
        """
        Calculate passenger comfort index based on turbulence intensity and duration
        """
        # Base discomfort from intensity
        base_discomfort = turbulence_intensity * 20
        
        # Duration factor (longer duration = more discomfort)
        duration_factor = math.log10(max(1, duration_minutes)) * 10
        
        # Total discomfort index (0-100 scale)
        discomfort_index = min(100, base_discomfort + duration_factor)
        
        # Convert to comfort index (inverse)
        comfort_index = 100 - discomfort_index
        
        return max(0, comfort_index)
    
    def estimate_fuel_consumption_impact(self, turbulence_intensity, flight_duration_hours):
        """
        Estimate additional fuel consumption due to turbulence
        """
        # Base fuel increase percentage per unit of turbulence intensity
        fuel_increase_base = turbulence_intensity * 2  # 2% per unit
        
        # Duration factor
        duration_factor = min(flight_duration_hours / 2, 1.5)  # Cap the factor
        
        # Total fuel increase percentage
        fuel_increase_percent = fuel_increase_base * duration_factor
        
        return min(fuel_increase_percent, 20)  # Cap at 20% increase
    
    def calculate_flight_delay_probability(self, turbulence_intensity, weather_conditions):
        """
        Calculate probability of flight delay due to turbulence
        """
        base_delay_prob = turbulence_intensity * 15  # Base 15% per unit
        
        # Weather condition adjustments
        weather_multipliers = {
            'clear': 1.0,
            'clouds': 1.1,
            'rain': 1.3,
            'thunderstorm': 1.8,
            'mist': 1.2,
            'fog': 1.5
        }
        
        weather_mult = weather_multipliers.get(weather_conditions.lower(), 1.0)
        delay_probability = min(95, base_delay_prob * weather_mult)  # Cap at 95%
        
        return delay_probability

# Global calculator instance
turbulence_calc = TurbulenceCalculator()
