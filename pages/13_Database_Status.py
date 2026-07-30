import streamlit as st
import os
from datetime import datetime
from utils.database import TurbulenceDatabase

st.set_page_config(page_title="Database Status", page_icon="🗄️", layout="wide")

st.title("Database Status Console")
st.markdown("Monitor real-time storage metrics, active connection configurations, and schema status.")

# Initialize database
db = TurbulenceDatabase()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## Connection Status")
    
    if db.use_sqlite:
        # SQLite active status card
        st.success("🟢 Active Storage Engine: **Local Persistent SQLite**")
        st.markdown(f"""
        The system is running on a self-contained local SQLite database because the remote production database is currently not configured or unreachable.
        All data (pilot reports, warnings, encounters) is saved persistently to your local project folder.
        
        * **Database Path:** `{db.sqlite_db_path}`
        * **Persisted Status:** Persistent (across application restarts)
        * **Storage Driver:** Python `sqlite3`
        """)
        
        # Display postgres error if one was attempted
        if db.connection_error:
            with st.expander("PostgreSQL Connection Diagnostics Logs"):
                st.code(db.connection_error, language="text")
                st.caption("To connect to PostgreSQL, please verify your credentials and make sure that your Supabase project is active (resumed) in your Supabase dashboard.")
    else:
        # PostgreSQL active status card
        st.success("🟢 Active Storage Engine: **Production PostgreSQL**")
        
        # Parse database host details securely for display
        db_host = "Unknown Host"
        if db.database_url:
            try:
                # Extract host from postgres://user:password@host:port/dbname
                parts = db.database_url.split("@")
                if len(parts) > 1:
                    db_host = parts[1].split("/")[0]
            except:
                pass
                
        st.markdown(f"""
        The system is connected directly to the production cloud database.
        
        * **Database Host:** `{db_host}`
        * **Connection Status:** Connected & Schema Initialized
        * **Storage Driver:** `psycopg2`
        """)

    # Statistics Section
    st.markdown("## Storage Metrics")
    
    # Query database counts
    with st.spinner("Fetching database statistics..."):
        try:
            reports_count = len(db.get_pilot_reports(limit=9999))
            active_alerts_count = len(db.get_active_alerts())
            encounters_count = len(db.get_turbulence_encounters(days=365))
        except Exception:
            reports_count = 0
            active_alerts_count = 0
            encounters_count = 0
            
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric(label="Total Pilot Reports", value=reports_count)
        
    with stat_col2:
        st.metric(label="Active Warnings", value=active_alerts_count)
        
    with stat_col3:
        st.metric(label="Historical Encounters (365d)", value=encounters_count)

with col2:
    st.markdown("## Configuration Overview")
    
    # Display file metadata for SQLite
    if db.use_sqlite and os.path.exists(db.sqlite_db_path):
        size_bytes = os.path.getsize(db.sqlite_db_path)
        size_kb = size_bytes / 1024
        mod_time = os.path.getmtime(db.sqlite_db_path)
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        st.info(f"""
        **File Properties:**
        * **File Size:** {size_kb:.1f} KB
        * **Last Modified:** {mod_date}
        """)
        
    # Environment status
    st.markdown("### Environment Variables")
    has_db_url = "Set" if os.getenv("DATABASE_URL") else "Not Set"
    st.text(f"DATABASE_URL: {has_db_url}")
    
    has_weather_key = "Set (Demo mode fallback active if invalid)" if os.getenv("OPENWEATHERMAP_API_KEY") else "Not Set"
    st.text(f"OPENWEATHERMAP_API_KEY: {has_weather_key}")
    
    # Help guide
    st.markdown("""
    ### Setup Instructions
    To transition from local SQLite storage to PostgreSQL in production:
    1. Activate or resume your database server (e.g. Supabase, RDS).
    2. Retrieve your database connection string URI:
       `postgresql://<username>:<password>@<host>:<port>/<dbname>`
    3. Update the `DATABASE_URL` value in your project `.env` file.
    4. Restart this Streamlit application.
    """)
