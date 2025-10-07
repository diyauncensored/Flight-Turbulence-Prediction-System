import json
from datetime import datetime, timedelta
from utils.database import TurbulenceDatabase
from utils.ml_models import TurbulencePredictor
from utils.weather_api import WeatherAPI
from utils.airport_data import get_airports

class AlertSystem:
    def __init__(self):
        self.db = TurbulenceDatabase()
        self.predictor = TurbulencePredictor()
        self.weather_api = WeatherAPI()
        
        # Alert thresholds
        self.thresholds = {
            'severe': {'min_level': 7.0, 'min_confidence': 0.75},
            'moderate': {'min_level': 5.0, 'min_confidence': 0.70},
            'light': {'min_level': 3.0, 'min_confidence': 0.65}
        }
    
    def check_and_create_alerts(self, airport_code=None):
        """Check conditions and create alerts if thresholds are exceeded"""
        airports = get_airports()
        
        if airport_code:
            airports = [a for a in airports if a['code'] == airport_code]
        
        alerts_created = []
        
        for airport in airports:
            # Get weather data
            weather = self.weather_api.get_current_weather(
                airport['coordinates'][0],
                airport['coordinates'][1]
            )
            
            if not weather:
                continue
            
            # Prepare prediction input
            flight_params = {
                'altitude': 35000,
                'wind_speed': weather['wind_speed'],
                'wind_direction': weather['wind_direction'],
                'temperature': weather['temperature'],
                'pressure': weather['pressure'],
                'humidity': weather['humidity'],
                'visibility': weather['visibility']
            }
            
            # Get prediction
            prediction = self.predictor.predict_turbulence(flight_params)
            
            if prediction:
                turbulence_level = prediction['turbulence_index']
                confidence = prediction['confidence']
                
                # Determine alert type
                alert_type = None
                severity = None
                
                if turbulence_level >= self.thresholds['severe']['min_level'] and \
                   confidence >= self.thresholds['severe']['min_confidence']:
                    alert_type = 'SEVERE'
                    severity = 'Severe'
                elif turbulence_level >= self.thresholds['moderate']['min_level'] and \
                     confidence >= self.thresholds['moderate']['min_confidence']:
                    alert_type = 'MODERATE'
                    severity = 'Moderate'
                elif turbulence_level >= self.thresholds['light']['min_level'] and \
                     confidence >= self.thresholds['light']['min_confidence']:
                    alert_type = 'ADVISORY'
                    severity = 'Light'
                
                if alert_type:
                    # Check if similar alert already exists
                    existing_alerts = self.db.get_active_alerts(airport_code=airport['code'])
                    
                    # Only create alert if no similar active alert in last hour
                    should_create = True
                    for existing in existing_alerts:
                        if existing['alert_type'] == alert_type:
                            time_diff = datetime.now() - existing['alert_date']
                            if time_diff.total_seconds() < 3600:  # 1 hour
                                should_create = False
                                break
                    
                    if should_create:
                        alert_data = {
                            'alert_date': datetime.now(),
                            'airport_code': airport['code'],
                            'alert_type': alert_type,
                            'severity': severity,
                            'predicted_level': turbulence_level,
                            'confidence_score': confidence,
                            'valid_from': datetime.now(),
                            'valid_until': datetime.now() + timedelta(hours=3),
                            'weather_data': json.dumps(weather, default=str),
                            'is_active': True
                        }
                        
                        alert_id = self.db.add_alert(alert_data)
                        if alert_id:
                            alerts_created.append({
                                'id': alert_id,
                                'airport': airport['code'],
                                'type': alert_type,
                                'severity': severity,
                                'level': turbulence_level,
                                'confidence': confidence
                            })
        
        return alerts_created
    
    def get_alert_summary(self, airport_code=None):
        """Get summary of active alerts"""
        alerts = self.db.get_active_alerts(airport_code=airport_code)
        
        summary = {
            'total_alerts': len(alerts),
            'severe_count': 0,
            'moderate_count': 0,
            'advisory_count': 0,
            'airports_affected': set(),
            'alerts': []
        }
        
        for alert in alerts:
            summary['airports_affected'].add(alert['airport_code'])
            
            if alert['alert_type'] == 'SEVERE':
                summary['severe_count'] += 1
            elif alert['alert_type'] == 'MODERATE':
                summary['moderate_count'] += 1
            else:
                summary['advisory_count'] += 1
            
            summary['alerts'].append({
                'id': alert['id'],
                'airport': alert['airport_code'],
                'type': alert['alert_type'],
                'severity': alert['severity'],
                'level': alert.get('predicted_level'),
                'confidence': alert.get('confidence_score'),
                'valid_from': alert.get('valid_from'),
                'valid_until': alert.get('valid_until'),
                'created_at': alert.get('created_at')
            })
        
        summary['airports_affected'] = list(summary['airports_affected'])
        
        return summary
    
    def update_alert_thresholds(self, severity_level, min_level, min_confidence):
        """Update alert thresholds"""
        if severity_level in self.thresholds:
            self.thresholds[severity_level]['min_level'] = min_level
            self.thresholds[severity_level]['min_confidence'] = min_confidence
            return True
        return False
    
    def deactivate_expired_alerts(self):
        """Deactivate alerts that have expired"""
        alerts = self.db.get_active_alerts()
        deactivated = 0
        
        for alert in alerts:
            if alert.get('valid_until') and alert['valid_until'] < datetime.now():
                if self.db.deactivate_alert(alert['id']):
                    deactivated += 1
        
        return deactivated
