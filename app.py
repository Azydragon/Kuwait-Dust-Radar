import streamlit as st
import xgboost as xgb
import pandas as pd
import requests
from datetime import datetime, timedelta

# 1. Page Configuration & Premium UI/UX Styling
st.set_page_config(
    page_title="Kuwait Atmospheric & Dust Radar", 
    page_icon="🌤️", 
    layout="wide"
)

st.markdown("""
<style>
    /* Global Deep Blue Atmospheric Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e3a8a 0%, #0f172a 60%, #020617 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Floating Cloud Animated Header */
    @keyframes floatClouds {
        0% { transform: translateY(0px) translateX(0px); }
        50% { transform: translateY(-6px) translateX(10px); }
        100% { transform: translateY(0px) translateX(0px); }
    }
    .floating-title {
        animation: floatClouds 5s ease-in-out infinite;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #38bdf8, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 10px 30px rgba(56, 189, 248, 0.4);
        margin-bottom: 0px;
    }

    /* Crisp White Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Glassmorphism Cards with Hover Animation */
    div[data-testid="stMetric"], .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.8);
        background: rgba(30, 41, 59, 0.9);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.4);
    }
    
    /* Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
        color: #cbd5e1;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# 2. Load Trained XGBoost Model
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('kuwait_dust_radar.json')
    return model

model = load_model()

# 3. Fetch Live, Forecast & Past Data for Salmiya, Kuwait
@st.cache_data(ttl=600)
def fetch_all_weather_data():
    try:
        # Current & 48h Forecast
        url_forecast = "https://api.open-meteo.com/v1/forecast?latitude=29.3375&longitude=48.0759&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
        res_forecast = requests.get(url_forecast, timeout=5).json()
        
        # Past Week Historical Data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url_past = f"https://archive-api.open-meteo.com/v1/archive?latitude=29.3375&longitude=48.0759&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,wind_speed_10m"
        res_past = requests.get(url_past, timeout=5).json()
        
        return res_forecast, res_past
    except Exception:
        return None, None

forecast_data, past_data = fetch_all_weather_data()

# Header Section
st.markdown('<p class="floating-title">☁️ Kuwait Atmospheric & Dust Radar</p>', unsafe_allow_html=True)
st.markdown("Precision meteorological intelligence and machine learning particulate matter forecasting for Hawalli Governorate.")
st.divider()

# 4. Multi-Tab Professional Layout
tab_current, tab_future, tab_past_table, tab_sim = st.tabs([
    "⚡ Current Telemetry", 
    "🤖 48-Hour AI Dust Forecast", 
    "📅 Past Week Log",
    "🕹️ Interactive Simulation"
])

# --- TAB 1: CURRENT TELEMETRY ---
with tab_current:
    st.subheader("Live Atmospheric Feed — Salmiya, Kuwait")
    if forecast_data and 'current' in forecast_data:
        curr = forecast_data['current']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f"{curr['temperature_2m']} °C", "Ambient")
        c2.metric("Wind Speed", f"{curr['wind_speed_10m']} km/h", "Surface")
        c3.metric("Humidity", f"{curr['relative_humidity_2m']}%", "Relative")
        c4.metric("Wind Direction", f"{curr['wind_direction_10m']}°", "Vector")
    else:
        st.warning("⚠️ Unable to reach live weather feed. Check connection.")

# --- TAB 2: 48-HOUR AI DUST FORECAST ---
with tab_future:
    st.subheader("Next 48-Hour Automated Dust Storm Prediction")
    st.markdown("Feeding Open-Meteo's future meteorological forecast into your trained XGBoost model to predict upcoming particulate matter trends.")

    if forecast_data and 'hourly' in forecast_data:
        hourly = forecast_data['hourly']
        df_future = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'temperature_2m': hourly['temperature_2m'],
            'relative_humidity_2m': hourly['relative_humidity_2m'],
            'wind_speed_10m': hourly['wind_speed_10m'],
            'wind_direction_10m': hourly['wind_direction_10m']
        })

        now = datetime.now()
        df_future = df_future[df_future['time'] >= now].head(48).copy()

        df_future['month'] = df_future['time'].dt.month
        df_future['hour'] = df_future['time'].dt.hour
        df_future['day_of_year'] = df_future['time'].dt.dayofyear

        for col in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m']:
            df_future[f'{col}_lag_1'] = df_future[col].shift(1)
            df_future[f'{col}_lag_2'] = df_future[col].shift(2)
            df_future[f'{col}_lag_3'] = df_future[col].shift(3)

        df_future['wind_speed_10m_rolling_mean_6h'] = df_future['wind_speed_10m'].rolling(6).mean()
        df_future['wind_speed_10m_rolling_std_6h'] = df_future['wind_speed_10m'].rolling(6).std().fillna(0)
        df_future['wind_speed_10m_rolling_mean_12h'] = df_future['wind_speed_10m'].rolling(12).mean()
        df_future['wind_speed_10m_rolling_std_12h'] = df_future['wind_speed_10m'].rolling(12).std().fillna(0)

        df_future = df_future.dropna().reset_index(drop=True)

        model_features = [
            'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m',
            'month', 'hour', 'day_of_year',
            'wind_speed_10m_lag_1', 'wind_direction_10m_lag_1', 'relative_humidity_2m_lag_1', 'temperature_2m_lag_1',
            'wind_speed_10m_lag_2', 'wind_direction_10m_lag_2', 'relative_humidity_2m_lag_2', 'temperature_2m_lag_2',
            'wind_speed_10m_lag_3', 'wind_direction_10m_lag_3', 'relative_humidity_2m_lag_3', 'temperature_2m_lag_3',
            'wind_speed_10m_rolling_mean_6h', 'wind_speed_10m_rolling_std_6h',
            'wind_speed_10m_rolling_mean_12h', 'wind_speed_10m_rolling_std_12h'
        ]

        X_pred = df_future[model_features]
        df_future['Predicted_PM10'] = model.predict(X_pred)

        chart_data = df_future.set_index('time')[['Predicted_PM10', 'wind_speed_10m']]
        st.line_chart(chart_data, color=["#38bdf8", "#94a3b8"])

        peak_pm10 = df_future['Predicted_PM10'].max()
        if peak_pm10 > 300:
            st.error(f"🚨 WARNING: Peak PM10 expected to reach {peak_pm10:.0f} µg/m³ within the next 48 hours.")
        else:
            st.success("✅ Atmospheric outlook stable over the next 48 hours. No severe dust spikes forecasted.")
    else:
        st.info("Loading 48-hour forecast telemetry...")

# --- TAB 3: PAST WEEK TABLE LOG ---
with tab_past_table:
    st.subheader("Historical Weather Log (Past 7 Days)")
    st.markdown("Archived hourly temperature and wind speed records for Kuwait.")
    
    if past_data and 'hourly' in past_data:
        df_table = pd.DataFrame({
            'Timestamp': pd.to_datetime(past_data['hourly']['time']),
            'Temperature (°C)': past_data['hourly']['temperature_2m'],
            'Wind Speed (km/h)': past_data['hourly']['wind_speed_10m']
        })
        # Sort descending so the most recent hours appear first
        df_table = df_table.sort_values(by='Timestamp', ascending=False).reset_index(drop=True)
        
        st.dataframe(df_table, use_container_width=True, height=450)
    else:
        st.info("Loading past week historical records...")

# --- TAB 4: INTERACTIVE SIMULATION ---
with tab_sim:
    st.subheader("Manual Scenario Simulation Sandbox")
    st.markdown("Override atmospheric parameters to simulate extreme wind fronts and test model response thresholds.")

    col_ctrl, col_res = st.columns([1, 1])
    with col_ctrl:
        sim_temp = st.slider("Temperature (°C)", 10.0, 55.0, 38.0)
        sim_hum = st.slider("Relative Humidity (%)", 5.0, 100.0, 12.0)
        sim_wind = st.slider("Wind Speed (km/h)", 0.0, 90.0, 32.0)
        sim_dir = st.slider("Wind Direction (°)", 0, 360, 310)

    now = datetime.now()
    sim_df = pd.DataFrame({
        'temperature_2m': [sim_temp], 'relative_humidity_2m': [sim_hum],
        'wind_speed_10m': [sim_wind], 'wind_direction_10m': [sim_dir],
        'month': [now.month], 'hour': [now.hour], 'day_of_year': [now.timetuple().tm_yday],
        'wind_speed_10m_lag_1': [sim_wind], 'wind_direction_10m_lag_1': [sim_dir],
        'relative_humidity_2m_lag_1': [sim_hum], 'temperature_2m_lag_1': [sim_temp],
        'wind_speed_10m_lag_2': [sim_wind], 'wind_direction_10m_lag_2': [sim_dir],
        'relative_humidity_2m_lag_2': [sim_hum], 'temperature_2m_lag_2': [sim_temp],
        'wind_speed_10m_lag_3': [sim_wind], 'wind_direction_10m_lag_3': [sim_dir],
        'relative_humidity_2m_lag_3': [sim_hum], 'temperature_2m_lag_3': [sim_temp],
        'wind_speed_10m_rolling_mean_6h': [sim_wind], 'wind_speed_10m_rolling_std_6h': [0.0],
        'wind_speed_10m_rolling_mean_12h': [sim_wind], 'wind_speed_10m_rolling_std_12h': [0.0]
    })

    sim_pred = float(model.predict(sim_df)[0])

    with col_res:
        st.markdown("### Simulation Results")
        st.metric("Simulated PM10 Output", f"{sim_pred:.0f} µg/m³")
        if sim_pred > 300:
            st.error("🚨 HAZARDOUS THRESHOLD REACHED: Severe dust storm conditions.")
            st.progress(float(min(sim_pred / 1000, 1.0)), text="Hazard Intensity")
        elif sim_pred > 150:
            st.warning("⚠️ MODERATE RISK: Elevated dust levels.")
            st.progress(float(min(sim_pred / 1000, 1.0)), text="Hazard Intensity")
        else:
            st.success("✅ SAFE: Normal air quality.")
            st.progress(float(max(sim_pred / 1000, 0.05)), text="Hazard Intensity")
