import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from utils.database import TurbulenceDatabase
from utils.airport_data import get_airports

st.set_page_config(page_title="Pilot Reports", page_icon="📝", layout="wide")

st.title("📝 Pilot Turbulence Reporting System")
st.markdown("Submit and view actual turbulence encounters reported by pilots")

# Initialize database
db = TurbulenceDatabase()

# Show demo mode warning if applicable
if db.demo_mode:
    st.warning("⚠️ Running in demo mode. Data is stored in session and will be lost when you refresh. Configure DATABASE_URL for persistence.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Submit Report", "View Reports", "Statistics"])

with tab1:
    st.header("Submit Turbulence Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic information
        st.subheader("Flight Information")
        
        airports = get_airports()
        airport_names = {f"{a['name']} ({a['code']})": a['code'] for a in airports}
        
        selected_airport = st.selectbox(
            "Airport",
            options=list(airport_names.keys())
        )
        airport_code = airport_names[selected_airport]
        
        flight_number = st.text_input("Flight Number (Optional)", placeholder="e.g., AI101")
        
        report_date = st.date_input("Date of Encounter", value=datetime.now())
        report_time = st.time_input("Time of Encounter", value=datetime.now().time())
        
        altitude = st.number_input("Altitude (feet)", min_value=0, max_value=50000, value=35000, step=1000)
        
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=120, value=5)
    
    with col2:
        # Turbulence details
        st.subheader("Turbulence Details")
        
        turbulence_level = st.selectbox(
            "Turbulence Level",
            options=["Light", "Light-Moderate", "Moderate", "Moderate-Severe", "Severe", "Extreme"]
        )
        
        # Severity index (0-10 scale)
        severity_index = st.slider(
            "Severity Index (0-10)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="0 = Negligible, 10 = Extreme"
        )
        
        # Location coordinates (optional)
        use_custom_location = st.checkbox("Specify Custom Location Coordinates", value=False)
        
        location_lat = None
        location_lon = None
        
        if use_custom_location:
            col_lat, col_lon = st.columns(2)
            with col_lat:
                location_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f")
            with col_lon:
                location_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, format="%.6f")
        
        weather_conditions = st.text_area(
            "Weather Conditions",
            placeholder="e.g., Clear air, convective clouds, jet stream..."
        )
        
        pilot_notes = st.text_area(
            "Additional Notes",
            placeholder="Any additional observations or comments..."
        )
    
    if st.button("Submit Report", type="primary", use_container_width=True):
        # Combine date and time
        report_datetime = datetime.combine(report_date, report_time)
        
        report_data = {
            'report_date': report_datetime,
            'airport_code': airport_code,
            'flight_number': flight_number if flight_number else None,
            'altitude': altitude,
            'turbulence_level': turbulence_level,
            'severity_index': severity_index,
            'location_lat': location_lat,
            'location_lon': location_lon,
            'duration_minutes': duration,
            'weather_conditions': weather_conditions if weather_conditions else None,
            'pilot_notes': pilot_notes if pilot_notes else None
        }
        
        report_id = db.add_pilot_report(report_data)
        
        if report_id:
            st.success(f"✅ Report submitted successfully! Report ID: {report_id}")
            st.balloons()
        else:
            st.error("Failed to submit report. Please try again.")

with tab2:
    st.header("Recent Pilot Reports")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_airport = st.selectbox(
            "Filter by Airport",
            options=["All Airports"] + list(airport_names.keys())
        )
    
    with col2:
        days_back = st.selectbox(
            "Time Period",
            options=[7, 14, 30, 60, 90],
            format_func=lambda x: f"Last {x} days"
        )
    
    with col3:
        max_reports = st.selectbox(
            "Number of Reports",
            options=[10, 25, 50, 100],
            index=1
        )
    
    # Get reports
    start_date = datetime.now() - timedelta(days=days_back)
    filter_code = airport_names.get(filter_airport) if filter_airport != "All Airports" else None
    
    reports = db.get_pilot_reports(
        airport_code=filter_code,
        start_date=start_date,
        limit=max_reports
    )
    
    if reports:
        # Convert to DataFrame
        df = pd.DataFrame(reports)
        df['report_date'] = pd.to_datetime(df['report_date'])
        
        st.write(f"Showing {len(reports)} reports")
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reports", len(reports))
        
        with col2:
            avg_severity = df['severity_index'].mean() if 'severity_index' in df and not df['severity_index'].isna().all() else 0
            st.metric("Avg Severity", f"{avg_severity:.1f}/10")
        
        with col3:
            severe_count = len(df[df['turbulence_level'].isin(['Moderate-Severe', 'Severe', 'Extreme'])])
            st.metric("Severe Events", severe_count)
        
        with col4:
            avg_duration = df['duration_minutes'].mean() if 'duration_minutes' in df else 0
            st.metric("Avg Duration", f"{avg_duration:.0f} min")
        
        # Display reports
        for idx, report in df.iterrows():
            with st.expander(
                f"{report['report_date'].strftime('%Y-%m-%d %H:%M')} - {report['airport_code']} - {report['turbulence_level']}",
                expanded=False
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Airport:** {report['airport_code']}")
                    if report.get('flight_number'):
                        st.write(f"**Flight:** {report['flight_number']}")
                    st.write(f"**Altitude:** {report['altitude']:,} ft")
                    st.write(f"**Duration:** {report.get('duration_minutes', 'N/A')} minutes")
                
                with col2:
                    st.write(f"**Turbulence Level:** {report['turbulence_level']}")
                    if report.get('severity_index'):
                        st.write(f"**Severity Index:** {report['severity_index']}/10")
                    if report.get('location_lat') and report.get('location_lon'):
                        st.write(f"**Location:** {report['location_lat']:.4f}, {report['location_lon']:.4f}")
                
                if report.get('weather_conditions'):
                    st.write(f"**Weather Conditions:** {report['weather_conditions']}")
                
                if report.get('pilot_notes'):
                    st.write(f"**Notes:** {report['pilot_notes']}")
    else:
        st.info("No reports found for the selected criteria.")

with tab3:
    st.header("Turbulence Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_airport = st.selectbox(
            "Select Airport for Statistics",
            options=["All Airports"] + list(airport_names.keys()),
            key="stats_airport"
        )
    
    with col2:
        stats_days = st.selectbox(
            "Analysis Period",
            options=[7, 14, 30, 60, 90],
            format_func=lambda x: f"Last {x} days",
            key="stats_days"
        )
    
    stats_code = airport_names.get(stats_airport) if stats_airport != "All Airports" else None
    stats = db.get_turbulence_statistics(airport_code=stats_code, days=stats_days)
    
    if stats and stats.get('severity_counts'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reports by Turbulence Level")
            severity_df = pd.DataFrame(
                list(stats['severity_counts'].items()),
                columns=['Level', 'Count']
            )
            st.bar_chart(severity_df.set_index('Level'))
        
        with col2:
            st.subheader("Severity by Altitude")
            if stats.get('altitude_stats'):
                altitude_df = pd.DataFrame(stats['altitude_stats'])
                altitude_df['altitude_range'] = altitude_df['altitude_range'].astype(int)
                st.line_chart(
                    altitude_df.set_index('altitude_range')['avg_severity'],
                    use_container_width=True
                )
        
        # Detailed statistics table
        st.subheader("Detailed Breakdown")
        st.dataframe(severity_df, use_container_width=True)
    else:
        st.info("No data available for statistics in the selected period.")

st.markdown("---")
st.caption("Pilot reports are stored securely and used to improve turbulence prediction models.")
