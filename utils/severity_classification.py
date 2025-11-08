import numpy as np
import streamlit as st

class TurbulenceSeverityClassifier:
    """Enhanced turbulence severity classification with detailed intensity levels"""
    
    def __init__(self):
        # Enhanced severity scale (0-10)
        self.severity_levels = {
            'Smooth': {'min': 0.0, 'max': 0.5, 'color': '#00ff00', 'icon': '✅'},
            'Light Chop': {'min': 0.5, 'max': 1.5, 'color': '#88ff00', 'icon': '🟢'},
            'Light': {'min': 1.5, 'max': 2.5, 'color': '#ffff00', 'icon': '🟡'},
            'Light-Moderate': {'min': 2.5, 'max': 3.5, 'color': '#ffdd00', 'icon': '🟡'},
            'Moderate': {'min': 3.5, 'max': 5.0, 'color': '#ffaa00', 'icon': '🟠'},
            'Moderate-Severe': {'min': 5.0, 'max': 6.5, 'color': '#ff8800', 'icon': '🟠'},
            'Severe': {'min': 6.5, 'max': 8.0, 'color': '#ff4444', 'icon': '🔴'},
            'Extreme': {'min': 8.0, 'max': 10.0, 'color': '#cc0000', 'icon': '🔴'}
        }
        
        # Detailed descriptions
        self.descriptions = {
            'Smooth': 'No turbulence detected. Safe flight conditions.',
            'Light Chop': 'Minor bumps, minimal passenger discomfort. Beverage service safe.',
            'Light': 'Slight, erratic changes in altitude/attitude. Some passenger discomfort possible.',
            'Light-Moderate': 'Noticeable turbulence with occasional moderate intensity. Secure loose items.',
            'Moderate': 'Definite strains against seat belts. Walking difficult. Items shift.',
            'Moderate-Severe': 'Strong turbulence with abrupt changes. Walking impossible. Objects dislodged.',
            'Severe': 'Large, abrupt changes in altitude/attitude. Aircraft may be out of control momentarily.',
            'Extreme': 'Aircraft violently tossed about. Virtually impossible to control. Structural damage possible.'
        }
        
        # Pilot action recommendations
        self.pilot_actions = {
            'Smooth': 'Normal operations. Monitor conditions.',
            'Light Chop': 'Monitor conditions. Inform passengers if unexpected.',
            'Light': 'Fasten seat belt sign recommended. Monitor closely.',
            'Light-Moderate': 'Fasten seat belt sign ON. Suspend cabin service.',
            'Moderate': 'Seat belt sign ON. Reduce speed to turbulence penetration speed. Suspend all cabin service.',
            'Moderate-Severe': 'Immediate seat belt sign ON. Reduce to Va. Consider altitude/route change. Cabin crew seated.',
            'Severe': 'Emergency procedures. Reduce to Va. Immediate altitude/route change. All crew seated. Report to ATC.',
            'Extreme': 'EMERGENCY. Maintain aircraft control. Immediate diversion. Structural inspection required upon landing.'
        }
        
        # Passenger impact levels
        self.passenger_impact = {
            'Smooth': 'No discomfort',
            'Light Chop': 'Minimal discomfort, slight unease possible',
            'Light': 'Some discomfort, anxiety in nervous passengers',
            'Light-Moderate': 'Moderate discomfort, increased anxiety',
            'Moderate': 'Significant discomfort, fear in many passengers',
            'Moderate-Severe': 'High discomfort, widespread fear, possible minor injuries',
            'Severe': 'Extreme discomfort, panic, injuries likely without restraint',
            'Extreme': 'Traumatic experience, serious injury risk even with restraints'
        }
        
        # Aircraft structural limits (as percentage of design limits)
        self.structural_concern = {
            'Smooth': 0,
            'Light Chop': 5,   
            'Light': 10,
            'Light-Moderate': 20,
            'Moderate': 35,
            'Moderate-Severe': 50,
            'Severe': 75,
            'Extreme': 95
        }
    
    def classify_severity(self, turbulence_index):
        """Classify turbulence severity based on index (0-10)"""
        for level, ranges in self.severity_levels.items():
            if ranges['min'] <= turbulence_index < ranges['max']:
                return level
        
        # Handle edge case for maximum value
        if turbulence_index >= 10.0:
            return 'Extreme'
        
        return 'Smooth'
    
    def get_severity_info(self, turbulence_index):
        """Get complete severity information"""
        severity = self.classify_severity(turbulence_index)
        
        return {
            'level': severity,
            'index': turbulence_index,
            'color': self.severity_levels[severity]['color'],
            'icon': self.severity_levels[severity]['icon'],
            'description': self.descriptions[severity],
            'pilot_action': self.pilot_actions[severity],
            'passenger_impact': self.passenger_impact[severity],
            'structural_concern': self.structural_concern[severity],
            'range': {
                'min': self.severity_levels[severity]['min'],
                'max': self.severity_levels[severity]['max']
            }
        }
    
    @st.cache_data(ttl=300)  # Cache results for 5 minutes
    def calculate_turbulence_index(self, weather_params):
        """Calculate turbulence index from weather parameters with enhanced accuracy"""
        
        try:
            # Extract and validate parameters
            wind_speed = float(weather_params.get('wind_speed', 0))
            wind_shear = float(weather_params.get('wind_shear', 0))
            temperature_gradient = float(weather_params.get('temp_gradient', 0))
            pressure_change = float(weather_params.get('pressure_change', 0))
            altitude = float(weather_params.get('altitude', 35000))
            
            # Additional weather factors
            humidity = float(weather_params.get('humidity', 50))
            visibility = float(weather_params.get('visibility', 10000))
            weather_condition = weather_params.get('weather_condition', 'clear')
            
            # Validate ranges
            if not (0 <= wind_speed <= 150):  # max wind speed in knots
                raise ValueError("Wind speed out of valid range")
            if not (0 <= altitude <= 45000):  # max altitude in feet
                raise ValueError("Altitude out of valid range")
                
            # Weather condition impact factors
            weather_factors = {
                'clear': 1.0,
                'cloudy': 1.1,
                'rain': 1.3,
                'thunderstorm': 1.8,
                'snow': 1.4
            }
            weather_multiplier = weather_factors.get(weather_condition.lower(), 1.0)
            
            # Visibility impact (reduced visibility often indicates worse conditions)
            visibility_factor = 1.0 + max(0, (10000 - visibility) / 20000)
            
            # Humidity impact (high humidity can indicate unstable air)
            humidity_factor = 1.0 + (humidity - 50) / 200  # Small increase for higher humidity
            
        except ValueError as e:
            st.error(f"Invalid parameter: {str(e)}")
            return 0.0
        except Exception as e:
            st.error(f"Error calculating turbulence index: {str(e)}")
            return 0.0
        
        # Base turbulence from wind speed (0-3)
        wind_component = min(wind_speed / 25.0, 3.0)
        
        # Wind shear contribution (0-3)
        shear_component = min(wind_shear / 15.0, 3.0)
        
        # Temperature gradient contribution (0-2)
        temp_component = min(abs(temperature_gradient) / 10.0, 2.0)
        
        # Pressure change contribution (0-2)
        pressure_component = min(abs(pressure_change) / 5.0, 2.0)
        
        # Altitude factor (turbulence generally increases with altitude up to jet stream)
        if altitude < 25000:
            altitude_factor = 0.8
        elif altitude < 35000:
            altitude_factor = 1.2
        else:
            altitude_factor = 1.0
        
        # Calculate base index
        base_index = (wind_component + shear_component + temp_component + pressure_component)
        
        # Apply environmental factors
        environmental_factor = weather_multiplier * visibility_factor * humidity_factor
        
        # Calculate final index with all factors
        turbulence_index = base_index * altitude_factor * environmental_factor
        
        # Calculate confidence score (0-1)
        confidence_score = min(1.0, max(0.5, (
            (1.0 if wind_speed > 0 else 0.6) *  # Wind data reliability
            (1.0 if wind_shear > 0 else 0.7) *  # Wind shear data reliability
            (1.0 if temperature_gradient != 0 else 0.8) *  # Temperature data reliability
            (1.0 if visibility < 8000 else 0.9) *  # Visibility impact on confidence
            (0.9 if weather_condition == 'clear' else 1.0)  # Weather condition reliability
        )))
        
        # Store confidence score for later use
        self._last_confidence = confidence_score
        
        # Clamp to 0-10 range
        turbulence_index = max(0, min(10, turbulence_index))
        
        return turbulence_index

    def get_confidence_score(self):
        """Get the confidence score of the last prediction"""
        return getattr(self, '_last_confidence', 0.8)  # Default 0.8 if no prediction made
    
    def get_all_severity_levels(self):
        """Get all severity levels with their ranges"""
        levels = []
        for level, ranges in self.severity_levels.items():
            levels.append({
                'level': level,
                'min': ranges['min'],
                'max': ranges['max'],
                'color': ranges['color'],
                'icon': ranges['icon'],
                'description': self.descriptions[level]
            })
        return levels
    
    def get_risk_assessment(self, turbulence_index, duration_minutes=None):
        """Get detailed risk assessment"""
        severity_info = self.get_severity_info(turbulence_index)
        
        # Risk categories
        if turbulence_index < 1.5:
            risk_level = 'Low'
            risk_color = '#00ff00'
        elif turbulence_index < 3.5:
            risk_level = 'Low-Moderate'
            risk_color = '#ffff00'
        elif turbulence_index < 5.0:
            risk_level = 'Moderate'
            risk_color = '#ffaa00'
        elif turbulence_index < 6.5:
            risk_level = 'High'
            risk_color = '#ff8800'
        elif turbulence_index < 8.0:
            risk_level = 'Very High'
            risk_color = '#ff4444'
        else:
            risk_level = 'Extreme'
            risk_color = '#cc0000'
        
        # Duration impact
        duration_factor = 1.0
        if duration_minutes:
            if duration_minutes > 30:
                duration_factor = 1.3
            elif duration_minutes > 15:
                duration_factor = 1.15
            elif duration_minutes > 5:
                duration_factor = 1.05
        
        adjusted_risk = turbulence_index * duration_factor
        
        assessment = {
            'risk_level': risk_level,
            'risk_color': risk_color,
            'base_index': turbulence_index,
            'adjusted_index': min(10, adjusted_risk),
            'duration_factor': duration_factor,
            'severity_info': severity_info,
            'recommendations': self._get_recommendations(severity_info['level'])
        }
        
        return assessment
    
    def _get_recommendations(self, severity_level):
        """Get specific recommendations based on severity"""
        recommendations = {
            'Smooth': [
                'Maintain normal operations',
                'Continue monitoring weather conditions',
                'Keep passengers informed of smooth conditions'
            ],
            'Light Chop': [
                'Monitor conditions for changes',
                'Brief passengers on minor turbulence',
                'Maintain normal service'
            ],
            'Light': [
                'Activate seat belt sign as precaution',
                'Inform passengers to remain seated when possible',
                'Continue monitoring intensity'
            ],
            'Light-Moderate': [
                'Seat belt sign ON',
                'Suspend beverage service',
                'Consider altitude change if persistent'
            ],
            'Moderate': [
                'Seat belt sign ON immediately',
                'Reduce to turbulence penetration speed',
                'Suspend all cabin service',
                'Request weather updates from ATC'
            ],
            'Moderate-Severe': [
                'IMMEDIATE seat belt sign activation',
                'Reduce to Va (design maneuvering speed)',
                'All cabin crew to be seated',
                'Request altitude/route change from ATC'
            ],
            'Severe': [
                'EMERGENCY procedures initiated',
                'Maintain Va speed strictly',
                'Immediate altitude/route deviation',
                'Report to ATC as emergency',
                'Prepare for possible structural inspection'
            ],
            'Extreme': [
                'EMERGENCY - Focus on aircraft control',
                'Immediate diversion to nearest suitable airport',
                'Declare emergency with ATC',
                'Mandatory structural inspection upon landing',
                'Medical assistance may be required for passengers'
            ]
        }
        
        return recommendations.get(severity_level, [])
    
    def compare_severities(self, index1, index2):
        """Compare two turbulence indices"""
        severity1 = self.classify_severity(index1)
        severity2 = self.classify_severity(index2)
        
        return {
            'index1': index1,
            'index2': index2,
            'severity1': severity1,
            'severity2': severity2,
            'difference': abs(index1 - index2),
            'percentage_change': ((index2 - index1) / max(index1, 0.1)) * 100 if index1 > 0 else 0
        }
