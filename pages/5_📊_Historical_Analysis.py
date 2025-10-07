import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.airport_data import INDIAN_AIRPORTS
from utils.data_processing import data_processor
from utils.ml_models import turbulence_model
from utils.turbulence_calculator import turbulence_calc

st.set_page_config(page_title="Historical Analysis", page_icon="📊", layout="wide")

st.title("📊 Historical Turbulence Data Analysis")
st.markdown("Time-series analysis and pattern detection in historical turbulence data")

# Sidebar controls
st.sidebar.header("Analysis Parameters")

# Time range selection
analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year"],
    index=1
)

# Convert period to days
period_mapping = {
    "Last 7 Days": 7,
    "Last 30 Days": 30,
    "Last 90 Days": 90,
    "Last 6 Months": 180,
    "Last Year": 365
}
days_back = period_mapping[analysis_period]

# Airport selection
selected_airports = st.sidebar.multiselect(
    "Select Airports",
    options=list(INDIAN_AIRPORTS.keys()),
    default=["DEL", "BOM", "BLR", "HYD"],
    format_func=lambda x: f"{x} - {INDIAN_AIRPORTS[x]['city']}"
)

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Focus",
    ["Temporal Patterns", "Seasonal Analysis", "Weather Correlation", "Risk Trends"]
)

# Data aggregation level
aggregation_level = st.sidebar.selectbox(
    "Data Aggregation",
    ["Hourly", "Daily", "Weekly", "Monthly"],
    index=1
)

if not selected_airports:
    st.warning("Please select at least one airport for analysis.")
    st.stop()

# Generate or load historical data
@st.cache_data(ttl=3600)
def load_historical_data(airports, days):
    """Load historical turbulence data for selected airports"""
    all_data = []
    
    for airport in airports:
        airport_data = data_processor.generate_sample_historical_data(airport, days)
        all_data.append(airport_data)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    return pd.DataFrame()

# Load data
with st.spinner(f"Loading {analysis_period.lower()} of historical data..."):
    historical_data = load_historical_data(selected_airports, days_back)

if historical_data.empty:
    st.error("No historical data available for the selected parameters.")
    st.stop()

# Data preprocessing
historical_data['date'] = historical_data['timestamp'].dt.date
historical_data['hour'] = historical_data['timestamp'].dt.hour
historical_data['day_of_week'] = historical_data['timestamp'].dt.dayofweek
historical_data['month'] = historical_data['timestamp'].dt.month
historical_data['is_weekend'] = historical_data['day_of_week'].isin([5, 6])

# Calculate risk levels
historical_data['risk_level'] = historical_data['turbulence_intensity'].apply(
    lambda x: turbulence_model.get_turbulence_risk_level(x)[0]
)

st.markdown("## 📈 Historical Data Overview")

# Summary statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_records = len(historical_data)
    st.metric("Total Records", f"{total_records:,}")

with col2:
    avg_turbulence = historical_data['turbulence_intensity'].mean()
    st.metric("Average Turbulence", f"{avg_turbulence:.2f}")

with col3:
    severe_incidents = len(historical_data[historical_data['turbulence_intensity'] >= 2.5])
    severe_percentage = (severe_incidents / total_records) * 100
    st.metric("Severe Incidents", f"{severe_incidents} ({severe_percentage:.1f}%)")

with col4:
    date_range = historical_data['timestamp'].max() - historical_data['timestamp'].min()
    st.metric("Date Range", f"{date_range.days} days")

# Main analysis based on selected type
if analysis_type == "Temporal Patterns":
    st.markdown("## ⏰ Temporal Pattern Analysis")
    
    # Time series plot
    if aggregation_level == "Hourly":
        time_series = historical_data.groupby([historical_data['timestamp'].dt.floor('H'), 'airport']).agg({
            'turbulence_intensity': 'mean',
            'temperature': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        time_column = 'timestamp'
    elif aggregation_level == "Daily":
        time_series = historical_data.groupby([historical_data['timestamp'].dt.date, 'airport']).agg({
            'turbulence_intensity': 'mean',
            'temperature': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        time_column = 'timestamp'
    elif aggregation_level == "Weekly":
        historical_data['week'] = historical_data['timestamp'].dt.to_period('W').dt.start_time
        time_series = historical_data.groupby(['week', 'airport']).agg({
            'turbulence_intensity': 'mean',
            'temperature': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        time_column = 'week'
    else:  # Monthly
        historical_data['month_year'] = historical_data['timestamp'].dt.to_period('M').dt.start_time
        time_series = historical_data.groupby(['month_year', 'airport']).agg({
            'turbulence_intensity': 'mean',
            'temperature': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        time_column = 'month_year'
    
    # Turbulence intensity over time
    fig = px.line(
        time_series,
        x=time_column,
        y='turbulence_intensity',
        color='airport',
        title=f"{aggregation_level} Turbulence Intensity Trends",
        labels={'turbulence_intensity': 'Average Turbulence Intensity'}
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly pattern analysis
    st.markdown("### ⏰ Daily Pattern Analysis")
    
    hourly_pattern = historical_data.groupby(['hour', 'airport']).agg({
        'turbulence_intensity': ['mean', 'std', 'count']
    }).reset_index()
    
    hourly_pattern.columns = ['hour', 'airport', 'mean_turbulence', 'std_turbulence', 'count']
    
    fig_hourly = px.line(
        hourly_pattern,
        x='hour',
        y='mean_turbulence',
        color='airport',
        title="Average Turbulence by Hour of Day",
        labels={'hour': 'Hour of Day', 'mean_turbulence': 'Average Turbulence Intensity'}
    )
    
    fig_hourly.update_layout(height=400)
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Day of week analysis
    st.markdown("### 📅 Weekly Pattern Analysis")
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pattern = historical_data.groupby(['day_of_week', 'airport'])['turbulence_intensity'].mean().reset_index()
    weekly_pattern['day_name'] = weekly_pattern['day_of_week'].map(lambda x: day_names[x])
    
    fig_weekly = px.bar(
        weekly_pattern,
        x='day_name',
        y='turbulence_intensity',
        color='airport',
        title="Average Turbulence by Day of Week",
        labels={'day_name': 'Day of Week', 'turbulence_intensity': 'Average Turbulence Intensity'}
    )
    
    fig_weekly.update_layout(height=400)
    st.plotly_chart(fig_weekly, use_container_width=True)

elif analysis_type == "Seasonal Analysis":
    st.markdown("## 🌡️ Seasonal Pattern Analysis")
    
    # Month-wise analysis
    monthly_stats = historical_data.groupby(['month', 'airport']).agg({
        'turbulence_intensity': ['mean', 'std', 'max'],
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    monthly_stats.columns = ['month', 'airport', 'mean_turbulence', 'std_turbulence', 
                           'max_turbulence', 'mean_temp', 'mean_humidity', 'mean_wind']
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_stats['month_name'] = monthly_stats['month'].map(lambda x: month_names[x-1])
    
    # Seasonal turbulence patterns
    fig_seasonal = px.line(
        monthly_stats,
        x='month_name',
        y='mean_turbulence',
        color='airport',
        title="Seasonal Turbulence Patterns",
        labels={'month_name': 'Month', 'mean_turbulence': 'Average Turbulence Intensity'}
    )
    
    fig_seasonal.update_layout(height=500)
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Temperature vs Turbulence correlation
    st.markdown("### 🌡️ Temperature-Turbulence Relationship")
    
    fig_temp_turb = px.scatter(
        monthly_stats,
        x='mean_temp',
        y='mean_turbulence',
        color='airport',
        size='mean_wind',
        hover_data=['month_name'],
        title="Temperature vs Turbulence Correlation",
        labels={'mean_temp': 'Average Temperature (°C)', 'mean_turbulence': 'Average Turbulence'}
    )
    
    fig_temp_turb.update_layout(height=500)
    st.plotly_chart(fig_temp_turb, use_container_width=True)
    
    # Seasonal risk distribution
    st.markdown("### 📊 Seasonal Risk Distribution")
    
    # Define seasons for India
    def get_season(month):
        if month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        elif month in [10, 11]:
            return "Post-Monsoon"
        else:
            return "Winter"
    
    historical_data['season'] = historical_data['month'].apply(get_season)
    
    seasonal_risk = historical_data.groupby(['season', 'airport', 'risk_level']).size().reset_index(name='count')
    
    fig_risk_dist = px.bar(
        seasonal_risk,
        x='season',
        y='count',
        color='risk_level',
        facet_col='airport',
        facet_col_wrap=3,
        title="Risk Level Distribution by Season and Airport",
        color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Severe': 'red'}
    )
    
    fig_risk_dist.update_layout(height=600)
    st.plotly_chart(fig_risk_dist, use_container_width=True)

elif analysis_type == "Weather Correlation":
    st.markdown("## 🌤️ Weather Parameter Correlation Analysis")
    
    # Correlation matrix
    weather_columns = ['turbulence_intensity', 'temperature', 'pressure', 'humidity', 
                      'wind_speed', 'visibility']
    
    correlation_matrix = historical_data[weather_columns].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Weather Parameters Correlation Matrix",
        color_continuous_scale='RdBu_r'
    )
    
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Weather condition impact
    st.markdown("### 🌦️ Weather Condition Impact")
    
    weather_impact = historical_data.groupby(['weather_condition', 'airport']).agg({
        'turbulence_intensity': ['mean', 'std', 'count']
    }).reset_index()
    
    weather_impact.columns = ['weather_condition', 'airport', 'mean_turbulence', 'std_turbulence', 'count']
    
    fig_weather_impact = px.box(
        historical_data,
        x='weather_condition',
        y='turbulence_intensity',
        color='airport',
        title="Turbulence Distribution by Weather Condition"
    )
    
    fig_weather_impact.update_layout(height=500)
    st.plotly_chart(fig_weather_impact, use_container_width=True)
    
    # Wind speed analysis
    st.markdown("### 💨 Wind Speed Analysis")
    
    # Create wind speed categories
    historical_data['wind_category'] = pd.cut(
        historical_data['wind_speed'],
        bins=[0, 5, 10, 15, 25, float('inf')],
        labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong']
    )
    
    wind_analysis = historical_data.groupby(['wind_category', 'airport'])['turbulence_intensity'].mean().reset_index()
    
    fig_wind = px.bar(
        wind_analysis,
        x='wind_category',
        y='turbulence_intensity',
        color='airport',
        title="Average Turbulence by Wind Speed Category",
        labels={'wind_category': 'Wind Speed Category', 'turbulence_intensity': 'Average Turbulence'}
    )
    
    fig_wind.update_layout(height=400)
    st.plotly_chart(fig_wind, use_container_width=True)

elif analysis_type == "Risk Trends":
    st.markdown("## 📈 Risk Trend Analysis")
    
    # Risk level trends over time
    daily_risk = historical_data.groupby([historical_data['timestamp'].dt.date, 'airport', 'risk_level']).size().reset_index(name='count')
    daily_risk_pct = daily_risk.groupby(['timestamp', 'airport']).apply(
        lambda x: x.assign(percentage=x['count'] / x['count'].sum() * 100)
    ).reset_index(drop=True)
    
    fig_risk_trend = px.area(
        daily_risk_pct[daily_risk_pct['risk_level'].isin(['High', 'Severe'])],
        x='timestamp',
        y='percentage',
        color='risk_level',
        facet_col='airport',
        facet_col_wrap=3,
        title="High/Severe Risk Trends Over Time (%)",
        color_discrete_map={'High': 'orange', 'Severe': 'red'}
    )
    
    fig_risk_trend.update_layout(height=600)
    st.plotly_chart(fig_risk_trend, use_container_width=True)
    
    # Risk escalation analysis
    st.markdown("### ⚠️ Risk Escalation Patterns")
    
    # Calculate moving averages
    for airport in selected_airports:
        airport_data = historical_data[historical_data['airport'] == airport].copy()
        airport_data = airport_data.sort_values('timestamp')
        airport_data['turbulence_ma_7'] = airport_data['turbulence_intensity'].rolling(window=7*8, min_periods=1).mean()  # 7 days
        airport_data['turbulence_ma_30'] = airport_data['turbulence_intensity'].rolling(window=30*8, min_periods=1).mean()  # 30 days
        
        historical_data.loc[historical_data['airport'] == airport, 'turbulence_ma_7'] = airport_data['turbulence_ma_7'].values
        historical_data.loc[historical_data['airport'] == airport, 'turbulence_ma_30'] = airport_data['turbulence_ma_30'].values
    
    # Plot moving averages
    ma_data = historical_data.groupby([historical_data['timestamp'].dt.date, 'airport']).agg({
        'turbulence_intensity': 'mean',
        'turbulence_ma_7': 'mean',
        'turbulence_ma_30': 'mean'
    }).reset_index()
    
    fig_ma = go.Figure()
    
    for airport in selected_airports:
        airport_ma = ma_data[ma_data['airport'] == airport]
        
        fig_ma.add_trace(go.Scatter(
            x=airport_ma['timestamp'],
            y=airport_ma['turbulence_ma_7'],
            mode='lines',
            name=f'{airport} - 7 Day MA',
            line=dict(width=2)
        ))
        
        fig_ma.add_trace(go.Scatter(
            x=airport_ma['timestamp'],
            y=airport_ma['turbulence_ma_30'],
            mode='lines',
            name=f'{airport} - 30 Day MA',
            line=dict(width=2, dash='dash')
        ))
    
    fig_ma.update_layout(
        title="Turbulence Moving Averages Trend",
        xaxis_title="Date",
        yaxis_title="Turbulence Intensity",
        height=500
    )
    
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Peak incident analysis
    st.markdown("### 🔴 Peak Incident Analysis")
    
    severe_incidents = historical_data[historical_data['turbulence_intensity'] >= 2.5].copy()
    
    if not severe_incidents.empty:
        # Time distribution of severe incidents
        severe_incidents['hour_category'] = pd.cut(
            severe_incidents['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )
        
        severe_time_dist = severe_incidents.groupby(['hour_category', 'airport']).size().reset_index(name='count')
        
        fig_severe_time = px.bar(
            severe_time_dist,
            x='hour_category',
            y='count',
            color='airport',
            title="Severe Incidents by Time of Day",
            labels={'hour_category': 'Time Period', 'count': 'Number of Incidents'}
        )
        
        fig_severe_time.update_layout(height=400)
        st.plotly_chart(fig_severe_time, use_container_width=True)
        
        # Top 10 most severe incidents
        st.markdown("#### 🔍 Most Severe Incidents")
        
        top_incidents = severe_incidents.nlargest(10, 'turbulence_intensity')[
            ['timestamp', 'airport', 'turbulence_intensity', 'temperature', 'wind_speed', 'weather_condition']
        ].copy()
        
        top_incidents['timestamp'] = top_incidents['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            top_incidents,
            use_container_width=True,
            hide_index=True,
            column_config={
                "turbulence_intensity": st.column_config.NumberColumn(
                    "Turbulence",
                    format="%.3f"
                ),
                "temperature": st.column_config.NumberColumn(
                    "Temp (°C)",
                    format="%.1f"
                ),
                "wind_speed": st.column_config.NumberColumn(
                    "Wind (m/s)",
                    format="%.1f"
                )
            }
        )
    else:
        st.info("No severe turbulence incidents found in the selected time period.")

# Data export functionality
st.markdown("## 💾 Data Export")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Analysis Data"):
        export_data = historical_data[[
            'timestamp', 'airport', 'turbulence_intensity', 'risk_level',
            'temperature', 'pressure', 'humidity', 'wind_speed', 'weather_condition'
        ]].copy()
        
        csv_data = export_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"turbulence_analysis_{analysis_period.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Summary Stats"):
        summary_stats = historical_data.groupby('airport').agg({
            'turbulence_intensity': ['count', 'mean', 'std', 'min', 'max'],
            'temperature': 'mean',
            'wind_speed': 'mean',
            'pressure': 'mean'
        }).round(3)
        
        csv_summary = summary_stats.to_csv()
        st.download_button(
            label="Download Summary CSV",
            data=csv_summary,
            file_name=f"turbulence_summary_{analysis_period.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

# Key insights
st.markdown("## 💡 Key Insights")

insights = []

# Calculate overall statistics
overall_avg = historical_data['turbulence_intensity'].mean()
overall_severe_pct = (len(historical_data[historical_data['turbulence_intensity'] >= 2.5]) / len(historical_data)) * 100

if overall_avg < 1.0:
    insights.append("✅ Overall turbulence levels are low across selected airports")
elif overall_avg > 2.0:
    insights.append("⚠️ High average turbulence intensity detected - monitor closely")

if overall_severe_pct > 10:
    insights.append(f"🔴 High percentage of severe turbulence incidents: {overall_severe_pct:.1f}%")
elif overall_severe_pct < 2:
    insights.append(f"✅ Low severe turbulence rate: {overall_severe_pct:.1f}%")

# Airport-specific insights
airport_stats = historical_data.groupby('airport')['turbulence_intensity'].mean()
highest_risk_airport = airport_stats.idxmax()
lowest_risk_airport = airport_stats.idxmin()

insights.append(f"📊 Highest average turbulence: {highest_risk_airport} ({airport_stats[highest_risk_airport]:.2f})")
insights.append(f"📊 Lowest average turbulence: {lowest_risk_airport} ({airport_stats[lowest_risk_airport]:.2f})")

# Seasonal insights
if 'season' in historical_data.columns:
    seasonal_avg = historical_data.groupby('season')['turbulence_intensity'].mean()
    peak_season = seasonal_avg.idxmax()
    insights.append(f"🌪️ Peak turbulence season: {peak_season}")

for insight in insights:
    st.info(insight)
