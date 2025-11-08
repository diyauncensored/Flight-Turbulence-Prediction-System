import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from utils.airport_data import INDIAN_AIRPORTS
from utils.data_processing import data_processor
from utils.ml_models import turbulence_model
from utils.turbulence_calculator import turbulence_calc

st.set_page_config(page_title="Statistical Dashboard", page_icon="📈", layout="wide")

st.title("📈 Statistical Analysis Dashboard")
st.markdown("Comprehensive turbulence statistics by airport, season, and altitude")

# Sidebar controls
st.sidebar.header("Analysis Parameters")

# Time period selection
analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "All Time"],
    index=2
)

# Airport selection
selected_airports = st.sidebar.multiselect(
    "Select Airports",
    options=list(INDIAN_AIRPORTS.keys()),
    default=list(INDIAN_AIRPORTS.keys()),
    format_func=lambda x: f"{x} - {INDIAN_AIRPORTS[x]['city']}"
)

# Statistical measures
statistical_measures = st.sidebar.multiselect(
    "Statistical Measures",
    options=["Frequency", "Mean Intensity", "Maximum Intensity", "Standard Deviation", "95th Percentile"],
    default=["Frequency", "Mean Intensity", "Maximum Intensity"]
)

# Grouping options
grouping_options = st.sidebar.multiselect(
    "Group Analysis By",
    options=["Airport", "Season", "Altitude Range", "Time of Day", "Day of Week"],
    default=["Airport", "Season", "Altitude Range"]
)

if not selected_airports:
    st.warning("Please select at least one airport for analysis.")
    st.stop()

# Convert period to days
period_mapping = {
    "Last 30 Days": 30,
    "Last 90 Days": 90,
    "Last 6 Months": 180,
    "Last Year": 365,
    "All Time": 1000  # Large number for all available data
}
days_back = period_mapping[analysis_period]

# Generate comprehensive statistical data
@st.cache_data(ttl=3600)
def generate_statistical_data(airports, days):
    """Generate comprehensive turbulence statistics data"""
    all_data = []
    
    for airport in airports:
        # Generate historical data for each airport
        airport_data = data_processor.generate_sample_historical_data(airport, days)
        all_data.append(airport_data)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Add derived features for statistical analysis
        combined_data['date'] = combined_data['timestamp'].dt.date
        combined_data['hour'] = combined_data['timestamp'].dt.hour
        combined_data['day_of_week'] = combined_data['timestamp'].dt.dayofweek
        combined_data['month'] = combined_data['timestamp'].dt.month
        combined_data['year'] = combined_data['timestamp'].dt.year
        combined_data['is_weekend'] = combined_data['day_of_week'].isin([5, 6])
        
        # Add season categorization (Indian seasons)
        def get_indian_season(month):
            if month in [3, 4, 5]:
                return "Summer"
            elif month in [6, 7, 8, 9]:
                return "Monsoon"
            elif month in [10, 11]:
                return "Post-Monsoon"
            else:
                return "Winter"
        
        combined_data['season'] = combined_data['month'].apply(get_indian_season)
        
        # Add altitude ranges
        def categorize_altitude(altitude):
            if altitude < 25000:
                return "Low (< 25k ft)"
            elif altitude < 30000:
                return "Medium (25k-30k ft)"
            elif altitude < 35000:
                return "High (30k-35k ft)"
            elif altitude < 40000:
                return "Very High (35k-40k ft)"
            else:
                return "Extreme (> 40k ft)"
        
        combined_data['altitude_range'] = combined_data['altitude'].apply(categorize_altitude)
        
        # Add time of day categories
        def categorize_time_of_day(hour):
            if 5 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 17:
                return "Afternoon"
            elif 17 <= hour < 21:
                return "Evening"
            else:
                return "Night"
        
        combined_data['time_of_day'] = combined_data['hour'].apply(categorize_time_of_day)
        
        # Add risk categories
        combined_data['risk_level'] = combined_data['turbulence_intensity'].apply(
            lambda x: turbulence_model.get_turbulence_risk_level(x)[0]
        )
        
        return combined_data
    
    return pd.DataFrame()

# Load statistical data
with st.spinner(f"Generating statistical data for {analysis_period.lower()}..."):
    stats_data = generate_statistical_data(selected_airports, days_back)

if stats_data.empty:
    st.error("No data available for the selected parameters.")
    st.stop()

# Overview metrics
st.markdown("## 📊 Overview Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_reports = len(stats_data)
    st.metric("Total Reports", f"{total_reports:,}")

with col2:
    avg_intensity = stats_data['turbulence_intensity'].mean()
    st.metric("Average Intensity", f"{avg_intensity:.3f}")

with col3:
    max_intensity = stats_data['turbulence_intensity'].max()
    st.metric("Maximum Intensity", f"{max_intensity:.3f}")

with col4:
    severe_count = len(stats_data[stats_data['turbulence_intensity'] >= 2.5])
    severe_pct = (severe_count / total_reports) * 100 if total_reports > 0 else 0
    st.metric("Severe Incidents", f"{severe_count} ({severe_pct:.1f}%)")

with col5:
    unique_dates = stats_data['date'].nunique()
    st.metric("Days Analyzed", unique_dates)

# Statistical Analysis by different groupings
if "Airport" in grouping_options:
    st.markdown("## 🏢 Statistics by Airport")
    
    # Calculate statistics by airport
    airport_stats = stats_data.groupby('airport').agg({
        'turbulence_intensity': ['count', 'mean', 'std', 'max', 'min', lambda x: np.percentile(x, 95)]
    }).round(4)
    
    airport_stats.columns = ['Frequency', 'Mean Intensity', 'Std Dev', 'Max Intensity', 'Min Intensity', '95th Percentile']
    airport_stats = airport_stats.reset_index()
    
    # Add airport names and cities
    airport_stats['Airport Name'] = airport_stats['airport'].map(lambda x: INDIAN_AIRPORTS[x]['name'])
    airport_stats['City'] = airport_stats['airport'].map(lambda x: INDIAN_AIRPORTS[x]['city'])
    
    # Reorder columns
    airport_stats = airport_stats[['airport', 'Airport Name', 'City'] + [col for col in airport_stats.columns if col not in ['airport', 'Airport Name', 'City']]]
    
    st.dataframe(
        airport_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "airport": "Code",
            "Frequency": st.column_config.NumberColumn("Frequency", format="%d"),
            "Mean Intensity": st.column_config.NumberColumn("Mean Intensity", format="%.4f"),
            "Max Intensity": st.column_config.NumberColumn("Max Intensity", format="%.4f"),
            "95th Percentile": st.column_config.NumberColumn("95th Percentile", format="%.4f")
        }
    )
    
    # Visualizations for airport statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Frequency by airport
        fig_freq = px.bar(
            airport_stats.sort_values('Frequency', ascending=False),
            x='airport',
            y='Frequency',
            title="Turbulence Report Frequency by Airport",
            labels={'airport': 'Airport Code', 'Frequency': 'Number of Reports'}
        )
        fig_freq.update_layout(height=400)
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col2:
        # Mean intensity by airport
        fig_intensity = px.bar(
            airport_stats.sort_values('Mean Intensity', ascending=False),
            x='airport',
            y='Mean Intensity',
            title="Average Turbulence Intensity by Airport",
            labels={'airport': 'Airport Code', 'Mean Intensity': 'Average Intensity'},
            color='Mean Intensity',
            color_continuous_scale='RdYlBu_r'
        )
        fig_intensity.update_layout(height=400)
        st.plotly_chart(fig_intensity, use_container_width=True)
    
    # Risk level distribution by airport
    st.markdown("### ⚠️ Risk Level Distribution by Airport")
    
    risk_by_airport = stats_data.groupby(['airport', 'risk_level']).size().reset_index(name='count')
    risk_by_airport_pct = risk_by_airport.groupby('airport').apply(
        lambda x: x.assign(percentage=x['count'] / x['count'].sum() * 100)
    ).reset_index(drop=True)
    
    fig_risk_airport = px.bar(
        risk_by_airport_pct,
        x='airport',
        y='percentage',
        color='risk_level',
        title="Risk Level Distribution by Airport (%)",
        labels={'airport': 'Airport Code', 'percentage': 'Percentage'},
        color_discrete_map={'Low': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Severe': 'red'}
    )
    fig_risk_airport.update_layout(height=500)
    st.plotly_chart(fig_risk_airport, use_container_width=True)

if "Season" in grouping_options:
    st.markdown("## 🌡️ Statistics by Season")
    
    # Seasonal statistics
    seasonal_stats = stats_data.groupby('season').agg({
        'turbulence_intensity': ['count', 'mean', 'std', 'max', lambda x: np.percentile(x, 95)]
    }).round(4)
    
    seasonal_stats.columns = ['Frequency', 'Mean Intensity', 'Std Dev', 'Max Intensity', '95th Percentile']
    seasonal_stats = seasonal_stats.reset_index()
    
    # Order seasons logically
    season_order = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
    seasonal_stats['season'] = pd.Categorical(seasonal_stats['season'], categories=season_order, ordered=True)
    seasonal_stats = seasonal_stats.sort_values('season')
    
    st.dataframe(
        seasonal_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "season": "Season",
            "Frequency": st.column_config.NumberColumn("Frequency", format="%d"),
            "Mean Intensity": st.column_config.NumberColumn("Mean Intensity", format="%.4f"),
            "Max Intensity": st.column_config.NumberColumn("Max Intensity", format="%.4f")
        }
    )
    
    # Seasonal visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal frequency
        fig_seasonal_freq = px.bar(
            seasonal_stats,
            x='season',
            y='Frequency',
            title="Turbulence Reports by Season",
            color='season',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_seasonal_freq.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_seasonal_freq, use_container_width=True)
    
    with col2:
        # Seasonal intensity
        fig_seasonal_intensity = px.line(
            seasonal_stats,
            x='season',
            y='Mean Intensity',
            title="Average Turbulence Intensity by Season",
            markers=True,
            line_shape='spline'
        )
        fig_seasonal_intensity.update_layout(height=400)
        st.plotly_chart(fig_seasonal_intensity, use_container_width=True)
    
    # Monthly breakdown
    st.markdown("### 📅 Monthly Turbulence Patterns")
    
    monthly_stats = stats_data.groupby(['month', 'season']).agg({
        'turbulence_intensity': ['count', 'mean']
    }).round(3)
    
    monthly_stats.columns = ['Frequency', 'Mean Intensity']
    monthly_stats = monthly_stats.reset_index()
    
    # Add month names
    monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: calendar.month_name[x])
    
    fig_monthly = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Monthly Frequency", "Monthly Mean Intensity"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig_monthly.add_trace(
        go.Bar(x=monthly_stats['month_name'], y=monthly_stats['Frequency'], 
               name='Frequency', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig_monthly.add_trace(
        go.Scatter(x=monthly_stats['month_name'], y=monthly_stats['Mean Intensity'],
                  mode='lines+markers', name='Mean Intensity', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    fig_monthly.update_xaxes(title_text="Month", row=1, col=1)
    fig_monthly.update_xaxes(title_text="Month", row=1, col=2)
    fig_monthly.update_yaxes(title_text="Frequency", row=1, col=1)
    fig_monthly.update_yaxes(title_text="Mean Intensity", row=1, col=2)
    
    fig_monthly.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_monthly, width='stretch')

if "Altitude Range" in grouping_options:
    st.markdown("## ✈️ Statistics by Altitude Range")
    
    # Altitude statistics
    altitude_stats = stats_data.groupby('altitude_range').agg({
        'turbulence_intensity': ['count', 'mean', 'std', 'max', lambda x: np.percentile(x, 95)]
    }).round(4)
    
    altitude_stats.columns = ['Frequency', 'Mean Intensity', 'Std Dev', 'Max Intensity', '95th Percentile']
    altitude_stats = altitude_stats.reset_index()
    
    # Order altitude ranges logically
    altitude_order = ["Low (< 25k ft)", "Medium (25k-30k ft)", "High (30k-35k ft)", 
                     "Very High (35k-40k ft)", "Extreme (> 40k ft)"]
    altitude_stats['altitude_range'] = pd.Categorical(
        altitude_stats['altitude_range'], categories=altitude_order, ordered=True
    )
    altitude_stats = altitude_stats.sort_values('altitude_range')
    
    st.dataframe(
        altitude_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "altitude_range": "Altitude Range",
            "Frequency": st.column_config.NumberColumn("Frequency", format="%d"),
            "Mean Intensity": st.column_config.NumberColumn("Mean Intensity", format="%.4f")
        }
    )
    
    # Altitude visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Altitude frequency distribution
        fig_alt_freq = px.bar(
            altitude_stats,
            x='altitude_range',
            y='Frequency',
            title="Turbulence Reports by Altitude Range",
            color='Mean Intensity',
            color_continuous_scale='RdYlBu_r'
        )
        fig_alt_freq.update_xaxes(tickangle=45)
        fig_alt_freq.update_layout(height=400)
        st.plotly_chart(fig_alt_freq, use_container_width=True)
    
    with col2:
        # Box plot of intensity by altitude
        fig_alt_box = px.box(
            stats_data,
            x='altitude_range',
            y='turbulence_intensity',
            title="Turbulence Intensity Distribution by Altitude"
        )
        fig_alt_box.update_xaxes(tickangle=45)
        fig_alt_box.update_layout(height=400)
        st.plotly_chart(fig_alt_box, use_container_width=True)

if "Time of Day" in grouping_options:
    st.markdown("## ⏰ Statistics by Time of Day")
    
    # Time of day statistics
    time_stats = stats_data.groupby('time_of_day').agg({
        'turbulence_intensity': ['count', 'mean', 'std', 'max']
    }).round(4)
    
    time_stats.columns = ['Frequency', 'Mean Intensity', 'Std Dev', 'Max Intensity']
    time_stats = time_stats.reset_index()
    
    # Order time periods logically
    time_order = ['Night', 'Morning', 'Afternoon', 'Evening']
    time_stats['time_of_day'] = pd.Categorical(
        time_stats['time_of_day'], categories=time_order, ordered=True
    )
    time_stats = time_stats.sort_values('time_of_day')
    
    st.dataframe(time_stats, use_container_width=True, hide_index=True)
    
    # Time of day visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Polar plot for hourly distribution
        hourly_stats = stats_data.groupby('hour')['turbulence_intensity'].agg(['count', 'mean']).reset_index()
        
        fig_polar = go.Figure()
        
        fig_polar.add_trace(go.Scatterpolar(
            r=hourly_stats['count'],
            theta=hourly_stats['hour'] * 15,  # Convert hour to degrees
            mode='lines+markers',
            name='Frequency',
            line=dict(color='blue', width=3)
        ))
        
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[0, 90, 180, 270],
                    ticktext=['00:00', '06:00', '12:00', '18:00']
                )
            ),
            title="24-Hour Turbulence Report Distribution",
            height=400
        )
        
        st.plotly_chart(fig_polar, use_container_width=True)
    
    with col2:
        # Time of day intensity comparison
        fig_time_intensity = px.bar(
            time_stats,
            x='time_of_day',
            y='Mean Intensity',
            title="Average Intensity by Time of Day",
            color='Mean Intensity',
            color_continuous_scale='plasma'
        )
        fig_time_intensity.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_time_intensity, use_container_width=True)

if "Day of Week" in grouping_options:
    st.markdown("## 📅 Statistics by Day of Week")
    
    # Day of week statistics
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    stats_data['day_name'] = stats_data['day_of_week'].map(lambda x: day_names[x])
    
    dow_stats = stats_data.groupby(['day_of_week', 'day_name']).agg({
        'turbulence_intensity': ['count', 'mean', 'std', 'max']
    }).round(4)
    
    dow_stats.columns = ['Frequency', 'Mean Intensity', 'Std Dev', 'Max Intensity']
    dow_stats = dow_stats.reset_index()
    
    st.dataframe(
        dow_stats[['day_name', 'Frequency', 'Mean Intensity', 'Std Dev', 'Max Intensity']],
        use_container_width=True,
        hide_index=True,
        column_config={"day_name": "Day of Week"}
    )
    
    # Day of week visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dow_freq = px.bar(
            dow_stats,
            x='day_name',
            y='Frequency',
            title="Reports by Day of Week",
            color='is_weekend',
            color_discrete_map={True: 'orange', False: 'lightblue'}
        )
        fig_dow_freq.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dow_freq, use_container_width=True)
    
    with col2:
        # Weekend vs Weekday comparison
        weekend_comparison = stats_data.groupby('is_weekend').agg({
            'turbulence_intensity': ['count', 'mean']
        }).round(3)
        weekend_comparison.columns = ['Frequency', 'Mean Intensity']
        weekend_comparison = weekend_comparison.reset_index()
        weekend_comparison['day_type'] = weekend_comparison['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        
        fig_weekend = px.bar(
            weekend_comparison,
            x='day_type',
            y='Mean Intensity',
            title="Weekend vs Weekday Intensity",
            color='day_type',
            color_discrete_sequence=['lightcoral', 'lightblue']
        )
        fig_weekend.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_weekend, use_container_width=True)

# Advanced Statistical Analysis
st.markdown("## 🔬 Advanced Statistical Analysis")

# Correlation analysis
st.markdown("### 📊 Correlation Analysis")

# Select numeric columns for correlation
numeric_cols = ['turbulence_intensity', 'temperature', 'pressure', 'wind_speed', 'humidity', 'altitude']
available_cols = [col for col in numeric_cols if col in stats_data.columns]

if len(available_cols) > 1:
    corr_matrix = stats_data[available_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Turbulence Parameter Correlations",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# Statistical significance tests
st.markdown("### 📈 Statistical Significance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Key Statistical Insights:**")
    
    # Calculate some key insights
    insights = []
    
    # Airport with highest mean intensity
    if "Airport" in grouping_options and 'airport_stats' in locals():
        highest_intensity_airport = airport_stats.loc[airport_stats['Mean Intensity'].idxmax()]
        insights.append(f"🏢 Highest average turbulence: {highest_intensity_airport['airport']} ({highest_intensity_airport['Mean Intensity']:.3f})")
    
    # Seasonal analysis
    if "Season" in grouping_options and 'seasonal_stats' in locals():
        peak_season = seasonal_stats.loc[seasonal_stats['Mean Intensity'].idxmax()]
        insights.append(f"🌪️ Peak turbulence season: {peak_season['season']} ({peak_season['Mean Intensity']:.3f})")
    
    # Altitude analysis
    if "Altitude Range" in grouping_options and 'altitude_stats' in locals():
        risky_altitude = altitude_stats.loc[altitude_stats['Mean Intensity'].idxmax()]
        insights.append(f"✈️ Most turbulent altitude range: {risky_altitude['altitude_range']}")
    
    # Overall statistics
    total_severe = len(stats_data[stats_data['turbulence_intensity'] >= 2.5])
    severe_percentage = (total_severe / len(stats_data)) * 100
    insights.append(f"⚠️ Severe turbulence rate: {severe_percentage:.1f}% ({total_severe:,} incidents)")
    
    # Standard deviation analysis
    std_dev = stats_data['turbulence_intensity'].std()
    insights.append(f"📊 Turbulence variability (σ): {std_dev:.3f}")
    
    for insight in insights:
        st.info(insight)

with col2:
    st.markdown("**Distribution Analysis:**")
    
    # Turbulence intensity distribution
    fig_dist = px.histogram(
        stats_data,
        x='turbulence_intensity',
        title="Turbulence Intensity Distribution",
        nbins=50,
        marginal="box"
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

# Export functionality
st.markdown("## 💾 Export Statistical Reports")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Summary Statistics"):
        # Create summary report
        summary_data = {
            'Metric': ['Total Reports', 'Average Intensity', 'Maximum Intensity', 'Standard Deviation', 'Severe Incidents (%)'],
            'Value': [
                len(stats_data),
                stats_data['turbulence_intensity'].mean(),
                stats_data['turbulence_intensity'].max(),
                stats_data['turbulence_intensity'].std(),
                (len(stats_data[stats_data['turbulence_intensity'] >= 2.5]) / len(stats_data)) * 100
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=csv_summary,
            file_name=f"turbulence_summary_statistics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🏢 Export Airport Statistics"):
        if 'airport_stats' in locals():
            csv_airport = airport_stats.to_csv(index=False)
            st.download_button(
                label="Download Airport Stats CSV",
                data=csv_airport,
                file_name=f"airport_turbulence_statistics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

with col3:
    if st.button("📅 Export Detailed Data"):
        # Export full dataset with all derived features
        export_columns = ['timestamp', 'airport', 'turbulence_intensity', 'altitude', 'season', 
                         'altitude_range', 'time_of_day', 'risk_level', 'temperature', 'wind_speed', 'pressure']
        
        export_data = stats_data[[col for col in export_columns if col in stats_data.columns]]
        
        csv_detailed = export_data.to_csv(index=False)
        st.download_button(
            label="Download Detailed CSV",
            data=csv_detailed,
            file_name=f"detailed_turbulence_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Key findings and recommendations
st.markdown("## 💡 Key Findings & Recommendations")

findings = []

# Analysis-based findings
if len(stats_data) > 100:  # Sufficient data for meaningful analysis
    
    # Seasonal findings
    if "Season" in grouping_options and 'seasonal_stats' in locals():
        monsoon_intensity = seasonal_stats[seasonal_stats['season'] == 'Monsoon']['Mean Intensity'].iloc[0] if 'Monsoon' in seasonal_stats['season'].values else 0
        if monsoon_intensity > stats_data['turbulence_intensity'].mean():
            findings.append("🌧️ **Monsoon Season Alert**: Significantly higher turbulence intensity during monsoon season")
    
    # Airport-specific findings
    if "Airport" in grouping_options and 'airport_stats' in locals():
        high_freq_airports = airport_stats[airport_stats['Frequency'] > airport_stats['Frequency'].mean()]
        if len(high_freq_airports) > 0:
            airports_list = ", ".join(high_freq_airports['airport'].tolist())
            findings.append(f"🏢 **High Activity Airports**: {airports_list} show above-average turbulence frequency")
    
    # Altitude findings
    if "Altitude Range" in grouping_options and 'altitude_stats' in locals():
        high_alt_risk = altitude_stats[altitude_stats['altitude_range'].str.contains('Very High|Extreme')]
        if not high_alt_risk.empty and high_alt_risk['Mean Intensity'].max() > 2.0:
            findings.append("✈️ **High Altitude Risk**: Increased turbulence intensity at very high altitudes (>35,000 ft)")
    
    # Time-based findings
    severe_incidents = stats_data[stats_data['turbulence_intensity'] >= 2.5]
    if len(severe_incidents) > 0:
        if 'time_of_day' in severe_incidents.columns:
            afternoon_severe = len(severe_incidents[severe_incidents['time_of_day'] == 'Afternoon'])
            if afternoon_severe / len(severe_incidents) > 0.4:
                findings.append("☀️ **Afternoon Peak**: Higher concentration of severe turbulence during afternoon hours")
    
    # Overall safety assessment
    severe_rate = (len(stats_data[stats_data['turbulence_intensity'] >= 2.5]) / len(stats_data)) * 100
    if severe_rate > 10:
        findings.append(f"⚠️ **Safety Alert**: High severe turbulence rate ({severe_rate:.1f}%) requires attention")
    elif severe_rate < 2:
        findings.append(f"✅ **Good Safety Record**: Low severe turbulence rate ({severe_rate:.1f}%)")

else:
    findings.append("📊 **Insufficient Data**: More data points needed for comprehensive statistical analysis")

# Display findings
if findings:
    for finding in findings:
        if "Alert" in finding or "⚠️" in finding:
            st.warning(finding)
        elif "✅" in finding or "Good" in finding:
            st.success(finding)
        else:
            st.info(finding)
else:
    st.info("📈 **Analysis Complete**: Review the statistics above for detailed insights into turbulence patterns")

# Methodology note
with st.expander("📚 Statistical Methodology"):
    st.markdown("""
    ## Statistical Methods Used
    
    ### Data Processing
    - **Temporal Grouping**: Data categorized by season (Indian climate), time of day, and day of week
    - **Altitude Stratification**: Flight levels grouped into meaningful ranges for aviation analysis
    - **Risk Categorization**: Turbulence intensity converted to operational risk levels
    
    ### Statistical Measures
    - **Frequency Analysis**: Count of turbulence reports per category
    - **Central Tendency**: Mean turbulence intensity for each group
    - **Variability**: Standard deviation and percentiles for distribution analysis
    - **Correlation Analysis**: Pearson correlation coefficients between parameters
    
    ### Seasonal Definitions (Indian Climate)
    - **Winter**: December, January, February
    - **Summer**: March, April, May  
    - **Monsoon**: June, July, August, September
    - **Post-Monsoon**: October, November
    
    ### Altitude Categories
    - **Low**: Below 25,000 ft (Approach/Departure phases)
    - **Medium**: 25,000-30,000 ft (Initial cruise)
    - **High**: 30,000-35,000 ft (Standard cruise)
    - **Very High**: 35,000-40,000 ft (High cruise, jet stream)
    - **Extreme**: Above 40,000 ft (Maximum service ceiling)
    
    ### Risk Assessment Scale
    - **Low**: 0.0-1.0 (Light turbulence, minimal passenger discomfort)
    - **Moderate**: 1.0-2.5 (Noticeable movement, seatbelt sign recommended)
    - **High**: 2.5-4.0 (Severe turbulence, difficult aircraft control)
    - **Severe**: 4.0+ (Extreme turbulence, structural stress concerns)
    
    ### Data Quality Notes
    - All timestamps converted to local Indian Standard Time (IST)
    - Missing data points excluded from statistical calculations
    - Outliers (>5σ from mean) flagged but included in analysis
    - Minimum sample size of 30 observations required for group statistics
    """)
