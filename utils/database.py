import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3
import os
import json
from datetime import datetime, timedelta
import streamlit as st

class TurbulenceDatabase:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.connection_error = None
        self.use_sqlite = False
        self.sqlite_db_path = os.path.join("models", "turbulence_database.db")
        
        # Test connecting to PostgreSQL if configured
        if self.database_url:
            try:
                conn = psycopg2.connect(
                    self.database_url,
                    cursor_factory=RealDictCursor,
                    connect_timeout=5,
                )
                conn.close()
                self.use_sqlite = False
            except Exception as e:
                self.connection_error = str(e)
                self.use_sqlite = True
        else:
            self.use_sqlite = True
            
        # Initialize the active database
        if self.use_sqlite:
            # Ensure database directory exists
            os.makedirs("models", exist_ok=True)
            self.initialize_sqlite_schema()
            
    def get_connection(self):
        """Get database connection (PostgreSQL)"""
        try:
            return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None
            
    def initialize_schema(self):
        """Create PostgreSQL database tables if they don't exist"""
        if self.use_sqlite:
            return True
            
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
            
    def initialize_sqlite_schema(self):
        """Create SQLite database tables and pre-seed with realistic data if empty"""
        conn = sqlite3.connect(self.sqlite_db_path)
        try:
            cursor = conn.cursor()
            
            # Pilot reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pilot_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date TEXT NOT NULL,
                    airport_code TEXT NOT NULL,
                    flight_number TEXT,
                    altitude INTEGER NOT NULL,
                    turbulence_level TEXT NOT NULL,
                    severity_index REAL,
                    location_lat REAL,
                    location_lon REAL,
                    duration_minutes INTEGER,
                    weather_conditions TEXT,
                    pilot_notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turbulence_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_date TEXT NOT NULL,
                    airport_code TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    predicted_level REAL,
                    confidence_score REAL,
                    valid_from TEXT,
                    valid_until TEXT,
                    weather_data TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Turbulence encounters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turbulence_encounters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encounter_date TEXT NOT NULL,
                    origin_airport TEXT,
                    destination_airport TEXT,
                    route_name TEXT,
                    altitude INTEGER,
                    location_lat REAL,
                    location_lon REAL,
                    turbulence_intensity REAL,
                    turbulence_type TEXT,
                    weather_wind_speed REAL,
                    weather_wind_direction REAL,
                    weather_pressure REAL,
                    weather_temperature REAL,
                    aircraft_type TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if seeding is needed
            cursor.execute("SELECT COUNT(*) FROM pilot_reports")
            count = cursor.fetchone()[0]
            if count == 0:
                self.pre_seed_sqlite_data(cursor)
                
            conn.commit()
        except Exception as e:
            print(f"Error initializing SQLite schema: {e}")
        finally:
            conn.close()
            
    def pre_seed_sqlite_data(self, cursor):
        """Pre-seed SQLite database with realistic initial operations data"""
        now = datetime.now()
        
        # Mock Pilot Reports
        reports = [
            (str(now - timedelta(hours=2)), 'DEL', 'AI102', 32000, 'Moderate', 4.5, 28.567, 77.100, 15, 'Cloudy, some convection', 'Encountered moderate chop during descent.'),
            (str(now - timedelta(hours=5)), 'BOM', '6E234', 36000, 'Light Chop', 1.2, 19.090, 72.863, 10, 'Clear', 'Light bumps at flight level 360.'),
            (str(now - timedelta(days=1, hours=3)), 'BLR', 'I5432', 28000, 'Severe', 7.2, 13.198, 77.706, 20, 'Thunderstorm vicinity', 'Severe turbulence, autopilot disconnected briefly.'),
            (str(now - timedelta(days=2)), 'HYD', '6E501', 34000, 'Light', 2.0, 17.240, 78.430, 8, 'Clear', 'Smooth flight with occasional light chop.'),
            (str(now - timedelta(days=3, hours=1)), 'CCU', 'SG123', 38000, 'Moderate-Severe', 5.8, 22.654, 88.446, 12, 'Monsoon clouds', 'Significant shifting of loose cabin items.')
        ]
        
        cursor.executemany("""
            INSERT INTO pilot_reports 
            (report_date, airport_code, flight_number, altitude, turbulence_level, 
             severity_index, location_lat, location_lon, duration_minutes, 
             weather_conditions, pilot_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, reports)
        
        # Mock Active Alerts
        alerts = [
            (str(now - timedelta(minutes=15)), 'DEL', 'SEVERE', 'Severe', 7.5, 0.85, str(now), str(now + timedelta(hours=2)), json.dumps({'temp': 32.5, 'wind': 25.0}), 1),
            (str(now - timedelta(hours=1)), 'BOM', 'MODERATE', 'Moderate', 4.2, 0.78, str(now), str(now + timedelta(hours=3)), json.dumps({'temp': 28.0, 'wind': 18.0}), 1),
            (str(now - timedelta(hours=4)), 'MAA', 'LIGHT', 'Light', 2.1, 0.70, str(now), str(now + timedelta(hours=1)), json.dumps({'temp': 30.2, 'wind': 12.0}), 1)
        ]
        
        cursor.executemany("""
            INSERT INTO turbulence_alerts 
            (alert_date, airport_code, alert_type, severity, predicted_level, 
             confidence_score, valid_from, valid_until, weather_data, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, alerts)
        
        # Mock Turbulence Encounters
        encounters = [
            (str(now - timedelta(hours=1)), 'DEL', 'BOM', 'DEL-BOM', 35000, 24.0, 75.0, 3.2, 'convective', 15.0, 240.0, 1010.0, 24.0, 'Boeing 737'),
            (str(now - timedelta(hours=4)), 'BOM', 'DEL', 'BOM-DEL', 37000, 22.5, 74.2, 1.5, 'clear_air', 12.0, 230.0, 1011.0, 20.0, 'Airbus A320'),
            (str(now - timedelta(days=1, hours=2)), 'DEL', 'BLR', 'DEL-BLR', 36000, 20.1, 76.5, 5.5, 'jet_stream', 35.0, 270.0, 1005.0, -10.0, 'Boeing 777'),
            (str(now - timedelta(days=2)), 'BLR', 'BOM', 'BLR-BOM', 32000, 16.0, 75.2, 0.8, 'clear_air', 8.0, 180.0, 1013.0, 15.0, 'ATR-72'),
            (str(now - timedelta(days=3)), 'MAA', 'DEL', 'MAA-DEL', 34000, 21.0, 79.0, 2.5, 'convective', 18.0, 210.0, 1008.0, 26.0, 'Airbus A321'),
            (str(now - timedelta(days=4)), 'DEL', 'CCU', 'DEL-CCU', 38000, 25.2, 83.5, 4.8, 'convective', 22.0, 190.0, 1007.0, 22.0, 'Boeing 787'),
            (str(now - timedelta(days=5)), 'CCU', 'BOM', 'CCU-BOM', 36000, 21.0, 81.0, 1.2, 'clear_air', 10.0, 160.0, 1012.0, 28.0, 'Airbus A320'),
            (str(now - timedelta(days=6)), 'BOM', 'HYD', 'BOM-HYD', 30000, 18.2, 75.8, 3.8, 'convective', 20.0, 200.0, 1009.0, 25.0, 'Boeing 737'),
            (str(now - timedelta(days=7)), 'HYD', 'DEL', 'HYD-DEL', 33000, 23.0, 77.9, 0.5, 'clear_air', 6.0, 90.0, 1014.0, 18.0, 'Airbus A320neo'),
            (str(now - timedelta(days=10)), 'DEL', 'MAA', 'DEL-MAA', 35000, 19.8, 78.5, 6.2, 'jet_stream', 42.0, 280.0, 1002.0, -15.0, 'Boeing 777')
        ]
        
        cursor.executemany("""
            INSERT INTO turbulence_encounters 
            (encounter_date, origin_airport, destination_airport, route_name, altitude,
             location_lat, location_lon, turbulence_intensity, turbulence_type,
             weather_wind_speed, weather_wind_direction, weather_pressure, 
             weather_temperature, aircraft_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, encounters)

    def add_pilot_report(self, report_data):
        """Add a new pilot turbulence report"""
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pilot_reports 
                    (report_date, airport_code, flight_number, altitude, turbulence_level, 
                     severity_index, location_lat, location_lon, duration_minutes, 
                     weather_conditions, pilot_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(report_data['report_date']),
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
                report_id = cursor.lastrowid
                conn.commit()
                return report_id
            except Exception as e:
                st.error(f"Error adding pilot report (SQLite): {str(e)}")
                return None
            finally:
                conn.close()
                
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
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM pilot_reports WHERE 1=1"
                params = []
                
                if airport_code:
                    query += " AND airport_code = ?"
                    params.append(airport_code)
                if start_date:
                    query += " AND report_date >= ?"
                    params.append(str(start_date))
                if end_date:
                    query += " AND report_date <= ?"
                    params.append(str(end_date))
                    
                query += " ORDER BY report_date DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                reports = []
                for r in rows:
                    d = dict(r)
                    for field in ['report_date', 'created_at']:
                        if isinstance(d.get(field), str):
                            try:
                                d[field] = datetime.fromisoformat(d[field])
                            except:
                                pass
                    reports.append(d)
                return reports
            except Exception as e:
                st.error(f"Error fetching pilot reports (SQLite): {str(e)}")
                return []
            finally:
                conn.close()
                
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
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            try:
                cursor = conn.cursor()
                weather_str = None
                if alert_data.get('weather_data'):
                    weather_str = json.dumps(alert_data['weather_data'])
                    
                cursor.execute("""
                    INSERT INTO turbulence_alerts 
                    (alert_date, airport_code, alert_type, severity, predicted_level, 
                     confidence_score, valid_from, valid_until, weather_data, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(alert_data['alert_date']),
                    alert_data['airport_code'],
                    alert_data['alert_type'],
                    alert_data['severity'],
                    alert_data.get('predicted_level'),
                    alert_data.get('confidence_score'),
                    str(alert_data['valid_from']) if alert_data.get('valid_from') else None,
                    str(alert_data['valid_until']) if alert_data.get('valid_until') else None,
                    weather_str,
                    1 if alert_data.get('is_active', True) else 0
                ))
                alert_id = cursor.lastrowid
                conn.commit()
                return alert_id
            except Exception as e:
                st.error(f"Error adding alert (SQLite): {str(e)}")
                return None
            finally:
                conn.close()
                
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
                json.dumps(alert_data.get('weather_data')) if alert_data.get('weather_data') else None,
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
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                query = """
                    SELECT * FROM turbulence_alerts 
                    WHERE is_active = 1 
                    AND (valid_until IS NULL OR datetime(valid_until) > datetime('now'))
                """
                params = []
                if airport_code:
                    query += " AND airport_code = ?"
                    params.append(airport_code)
                query += " ORDER BY alert_date DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                alerts = []
                for r in rows:
                    d = dict(r)
                    d['is_active'] = bool(d['is_active'])
                    for field in ['alert_date', 'valid_from', 'valid_until', 'created_at']:
                        if isinstance(d.get(field), str):
                            try:
                                d[field] = datetime.fromisoformat(d[field])
                            except:
                                pass
                    if d.get('weather_data'):
                        try:
                            d['weather_data'] = json.loads(d['weather_data'])
                        except:
                            pass
                    alerts.append(d)
                return alerts
            except Exception as e:
                st.error(f"Error fetching active alerts (SQLite): {str(e)}")
                return []
            finally:
                conn.close()
                
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
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE turbulence_alerts 
                    SET is_active = 0 
                    WHERE id = ?
                """, (alert_id,))
                conn.commit()
                return True
            except Exception as e:
                st.error(f"Error deactivating alert (SQLite): {str(e)}")
                return False
            finally:
                conn.close()
                
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
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO turbulence_encounters 
                    (encounter_date, origin_airport, destination_airport, route_name, altitude,
                     location_lat, location_lon, turbulence_intensity, turbulence_type,
                     weather_wind_speed, weather_wind_direction, weather_pressure, 
                     weather_temperature, aircraft_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(encounter_data['encounter_date']),
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
                encounter_id = cursor.lastrowid
                conn.commit()
                return encounter_id
            except Exception as e:
                st.error(f"Error adding encounter (SQLite): {str(e)}")
                return None
            finally:
                conn.close()
                
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
            
    def get_turbulence_encounters(self, airports=None, days=30):
        """Get historical turbulence encounters"""
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cutoff = datetime.now() - timedelta(days=days)
                query = "SELECT * FROM turbulence_encounters WHERE datetime(encounter_date) >= datetime(?)"
                params = [str(cutoff)]
                
                if airports:
                    placeholders = ','.join(['?'] * len(airports))
                    query += f" AND (origin_airport IN ({placeholders}) OR destination_airport IN ({placeholders}))"
                    params.extend(airports)
                    params.extend(airports)
                    
                query += " ORDER BY encounter_date DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                encounters = []
                for r in rows:
                    d = dict(r)
                    for field in ['encounter_date', 'created_at']:
                        if isinstance(d.get(field), str):
                            try:
                                d[field] = datetime.fromisoformat(d[field])
                            except:
                                pass
                    encounters.append(d)
                return encounters
            except Exception as e:
                st.error(f"Error fetching encounters (SQLite): {str(e)}")
                return []
            finally:
                conn.close()
                
        conn = self.get_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            query = "SELECT * FROM turbulence_encounters WHERE encounter_date >= CURRENT_TIMESTAMP - INTERVAL '%s days'"
            params = [days]
            
            if airports:
                placeholders = ','.join(['%s'] * len(airports))
                query += f" AND (origin_airport IN ({placeholders}) OR destination_airport IN ({placeholders}))"
                params.extend(airports)
                params.extend(airports)
                
            query += " ORDER BY encounter_date DESC"
            
            cursor.execute(query, params)
            encounters = cursor.fetchall()
            cursor.close()
            conn.close()
            return encounters
            
        except Exception as e:
            st.error(f"Error fetching encounters: {str(e)}")
            if conn:
                conn.close()
            return []
            
    def get_turbulence_statistics(self, airport_code=None, days=30):
        """Get turbulence statistics for analysis"""
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cutoff = datetime.now() - timedelta(days=days)
                
                # Get report counts by severity
                query = """
                    SELECT turbulence_level, COUNT(*) as count
                    FROM pilot_reports
                    WHERE datetime(report_date) >= datetime(?)
                """
                params = [str(cutoff)]
                if airport_code:
                    query += " AND airport_code = ?"
                    params.append(airport_code)
                query += " GROUP BY turbulence_level"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                severity_counts = {r['turbulence_level']: r['count'] for r in rows}
                
                # Get average severity by altitude
                query = """
                    SELECT 
                        (altitude / 5000) * 5000 as altitude_range,
                        AVG(severity_index) as avg_severity,
                        COUNT(*) as count
                    FROM pilot_reports
                    WHERE datetime(report_date) >= datetime(?)
                    AND severity_index IS NOT NULL
                """
                params = [str(cutoff)]
                if airport_code:
                    query += " AND airport_code = ?"
                    params.append(airport_code)
                query += " GROUP BY altitude_range ORDER BY altitude_range"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                altitude_stats = []
                for r in rows:
                    altitude_stats.append({
                        'altitude_range': r['altitude_range'],
                        'avg_severity': r['avg_severity'],
                        'count': r['count']
                    })
                    
                return {
                    'severity_counts': severity_counts,
                    'altitude_stats': altitude_stats
                }
            except Exception as e:
                st.error(f"Error fetching statistics (SQLite): {str(e)}")
                return {}
            finally:
                conn.close()
                
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
