import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import streamlit as st

class TurbulenceDatabase:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.demo_mode = not bool(self.database_url)
        
        # In-memory storage for demo mode
        if self.demo_mode:
            if 'demo_pilot_reports' not in st.session_state:
                st.session_state.demo_pilot_reports = []
            if 'demo_alerts' not in st.session_state:
                st.session_state.demo_alerts = []
            if 'demo_encounters' not in st.session_state:
                st.session_state.demo_encounters = []
            if 'demo_id_counter' not in st.session_state:
                st.session_state.demo_id_counter = 1
        
    def get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None
    
    def initialize_schema(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Pilot reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pilot_reports (
                    id SERIAL PRIMARY KEY,
                    report_date TIMESTAMP NOT NULL,
                    airport_code VARCHAR(10) NOT NULL,
                    flight_number VARCHAR(20),
                    altitude INTEGER NOT NULL,
                    turbulence_level VARCHAR(20) NOT NULL,
                    severity_index FLOAT,
                    location_lat FLOAT,
                    location_lon FLOAT,
                    duration_minutes INTEGER,
                    weather_conditions TEXT,
                    pilot_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turbulence_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_date TIMESTAMP NOT NULL,
                    airport_code VARCHAR(10) NOT NULL,
                    alert_type VARCHAR(20) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    predicted_level FLOAT,
                    confidence_score FLOAT,
                    valid_from TIMESTAMP,
                    valid_until TIMESTAMP,
                    weather_data JSONB,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Turbulence encounters table (historical data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turbulence_encounters (
                    id SERIAL PRIMARY KEY,
                    encounter_date TIMESTAMP NOT NULL,
                    origin_airport VARCHAR(10),
                    destination_airport VARCHAR(10),
                    route_name VARCHAR(100),
                    altitude INTEGER,
                    location_lat FLOAT,
                    location_lon FLOAT,
                    turbulence_intensity FLOAT,
                    turbulence_type VARCHAR(50),
                    weather_wind_speed FLOAT,
                    weather_wind_direction FLOAT,
                    weather_pressure FLOAT,
                    weather_temperature FLOAT,
                    aircraft_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Database schema initialization error: {str(e)}")
            if conn:
                conn.close()
            return False
    
    def add_pilot_report(self, report_data):
        """Add a new pilot turbulence report"""
        # Demo mode - use session state
        if self.demo_mode:
            report_id = st.session_state.demo_id_counter
            st.session_state.demo_id_counter += 1
            
            report = {
                'id': report_id,
                'created_at': datetime.now(),
                **report_data
            }
            st.session_state.demo_pilot_reports.append(report)
            return report_id
        
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pilot_reports 
                (report_date, airport_code, flight_number, altitude, turbulence_level, 
                 severity_index, location_lat, location_lon, duration_minutes, 
                 weather_conditions, pilot_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                report_data['report_date'],
                report_data['airport_code'],
                report_data.get('flight_number'),
                report_data['altitude'],
                report_data['turbulence_level'],
                report_data.get('severity_index'),
                report_data.get('location_lat'),
                report_data.get('location_lon'),
                report_data.get('duration_minutes'),
                report_data.get('weather_conditions'),
                report_data.get('pilot_notes')
            ))
            
            report_id = cursor.fetchone()['id']
            conn.commit()
            cursor.close()
            conn.close()
            return report_id
            
        except Exception as e:
            st.error(f"Error adding pilot report: {str(e)}")
            if conn:
                conn.close()
            return None
    
    def get_pilot_reports(self, airport_code=None, start_date=None, end_date=None, limit=100):
        """Get pilot reports with optional filters"""
        # Demo mode - use session state
        if self.demo_mode:
            reports = st.session_state.demo_pilot_reports.copy()
            
            # Apply filters
            if airport_code:
                reports = [r for r in reports if r.get('airport_code') == airport_code]
            if start_date:
                reports = [r for r in reports if r.get('report_date') >= start_date]
            if end_date:
                reports = [r for r in reports if r.get('report_date') <= end_date]
            
            # Sort and limit
            reports = sorted(reports, key=lambda x: x.get('report_date', datetime.now()), reverse=True)
            return reports[:limit]
        
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            query = "SELECT * FROM pilot_reports WHERE 1=1"
            params = []
            
            if airport_code:
                query += " AND airport_code = %s"
                params.append(airport_code)
            
            if start_date:
                query += " AND report_date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND report_date <= %s"
                params.append(end_date)
            
            query += " ORDER BY report_date DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            reports = cursor.fetchall()
            
            cursor.close()
            conn.close()
            return reports
            
        except Exception as e:
            st.error(f"Error fetching pilot reports: {str(e)}")
            if conn:
                conn.close()
            return []
    
    def add_alert(self, alert_data):
        """Add a new turbulence alert"""
        # Demo mode - use session state
        if self.demo_mode:
            alert_id = st.session_state.demo_id_counter
            st.session_state.demo_id_counter += 1
            
            alert = {
                'id': alert_id,
                'created_at': datetime.now(),
                **alert_data
            }
            st.session_state.demo_alerts.append(alert)
            return alert_id
        
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO turbulence_alerts 
                (alert_date, airport_code, alert_type, severity, predicted_level, 
                 confidence_score, valid_from, valid_until, weather_data, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                alert_data['alert_date'],
                alert_data['airport_code'],
                alert_data['alert_type'],
                alert_data['severity'],
                alert_data.get('predicted_level'),
                alert_data.get('confidence_score'),
                alert_data.get('valid_from'),
                alert_data.get('valid_until'),
                alert_data.get('weather_data'),
                alert_data.get('is_active', True)
            ))
            
            alert_id = cursor.fetchone()['id']
            conn.commit()
            cursor.close()
            conn.close()
            return alert_id
            
        except Exception as e:
            st.error(f"Error adding alert: {str(e)}")
            if conn:
                conn.close()
            return None
    
    def get_active_alerts(self, airport_code=None):
        """Get active turbulence alerts"""
        # Demo mode - use session state
        if self.demo_mode:
            alerts = [a for a in st.session_state.demo_alerts if a.get('is_active', True)]
            
            # Filter by expiry
            alerts = [a for a in alerts if not a.get('valid_until') or a['valid_until'] > datetime.now()]
            
            # Filter by airport
            if airport_code:
                alerts = [a for a in alerts if a.get('airport_code') == airport_code]
            
            return sorted(alerts, key=lambda x: x.get('alert_date', datetime.now()), reverse=True)
        
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM turbulence_alerts 
                WHERE is_active = TRUE 
                AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
            """
            params = []
            
            if airport_code:
                query += " AND airport_code = %s"
                params.append(airport_code)
            
            query += " ORDER BY alert_date DESC"
            
            cursor.execute(query, params)
            alerts = cursor.fetchall()
            
            cursor.close()
            conn.close()
            return alerts
            
        except Exception as e:
            st.error(f"Error fetching alerts: {str(e)}")
            if conn:
                conn.close()
            return []
    
    def deactivate_alert(self, alert_id):
        """Deactivate an alert"""
        # Demo mode - use session state
        if self.demo_mode:
            for alert in st.session_state.demo_alerts:
                if alert.get('id') == alert_id:
                    alert['is_active'] = False
                    return True
            return False
        
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE turbulence_alerts 
                SET is_active = FALSE 
                WHERE id = %s
            """, (alert_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error deactivating alert: {str(e)}")
            if conn:
                conn.close()
            return False
    
    def add_turbulence_encounter(self, encounter_data):
        """Add turbulence encounter for historical data"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO turbulence_encounters 
                (encounter_date, origin_airport, destination_airport, route_name, altitude,
                 location_lat, location_lon, turbulence_intensity, turbulence_type,
                 weather_wind_speed, weather_wind_direction, weather_pressure, 
                 weather_temperature, aircraft_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                encounter_data['encounter_date'],
                encounter_data.get('origin_airport'),
                encounter_data.get('destination_airport'),
                encounter_data.get('route_name'),
                encounter_data.get('altitude'),
                encounter_data.get('location_lat'),
                encounter_data.get('location_lon'),
                encounter_data.get('turbulence_intensity'),
                encounter_data.get('turbulence_type'),
                encounter_data.get('weather_wind_speed'),
                encounter_data.get('weather_wind_direction'),
                encounter_data.get('weather_pressure'),
                encounter_data.get('weather_temperature'),
                encounter_data.get('aircraft_type')
            ))
            
            encounter_id = cursor.fetchone()['id']
            conn.commit()
            cursor.close()
            conn.close()
            return encounter_id
            
        except Exception as e:
            st.error(f"Error adding encounter: {str(e)}")
            if conn:
                conn.close()
            return None
    
    def get_turbulence_statistics(self, airport_code=None, days=30):
        """Get turbulence statistics for analysis"""
        conn = self.get_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Get report counts by severity
            query = """
                SELECT turbulence_level, COUNT(*) as count
                FROM pilot_reports
                WHERE report_date >= CURRENT_TIMESTAMP - INTERVAL '%s days'
            """
            params = [days]
            
            if airport_code:
                query += " AND airport_code = %s"
                params.append(airport_code)
            
            query += " GROUP BY turbulence_level"
            
            cursor.execute(query, params)
            severity_counts = {row['turbulence_level']: row['count'] for row in cursor.fetchall()}
            
            # Get average severity by altitude
            query = """
                SELECT 
                    FLOOR(altitude / 5000) * 5000 as altitude_range,
                    AVG(severity_index) as avg_severity,
                    COUNT(*) as count
                FROM pilot_reports
                WHERE report_date >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND severity_index IS NOT NULL
            """
            params = [days]
            
            if airport_code:
                query += " AND airport_code = %s"
                params.append(airport_code)
            
            query += " GROUP BY altitude_range ORDER BY altitude_range"
            
            cursor.execute(query, params)
            altitude_stats = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'severity_counts': severity_counts,
                'altitude_stats': altitude_stats
            }
            
        except Exception as e:
            st.error(f"Error fetching statistics: {str(e)}")
            if conn:
                conn.close()
            return {}
