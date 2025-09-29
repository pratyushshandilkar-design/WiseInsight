# ==============================================================================
# UNIFIED WFM PLATFORM - WISE INSIGHTS (v4.0 - Database Integration)
# ==============================================================================
# This application provides Forecasting and Capacity Planning tools,
# secured with a granular, role-based access control system.
# ==============================================================================

# --- SECTION 1: UNIVERSAL LIBRARY IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import math
import os
import time
import zipfile
import sqlite3
import bcrypt
import json
from io import BytesIO
from datetime import datetime, timedelta
from contextlib import contextmanager
import json

# Machine Learning and Time Series Libraries
from prophet import Prophet
import plotly.graph_objects as go
from holidays import CountryHoliday
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from ortools.sat.python import cp_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# --- SECTION 2: GLOBAL CONFIGURATION & UTILITIES ---

st.set_page_config(
    page_title="WiseInsights",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTT1PWyhOO0xEnxEPJ5ReTNTreJpAoOEJo6Tg&s",
    layout="wide"
)

# Define a constant for the archive directory path
SHRINK_ARCHIVE_DIR = 'shrinkage_archive'

warnings.filterwarnings("ignore")
# MODIFIED: Add a new table to your database creation function
def create_db_tables():
    """Creates all necessary tables for users, runs, and capacity plans."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # ... (your existing users and runs tables are unchanged) ...
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL,
                can_view_shrinkage INTEGER NOT NULL, can_view_volume INTEGER NOT NULL,
                can_view_capacity INTEGER NOT NULL, can_manage_schedules INTEGER NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_type TEXT NOT NULL,
                job_run_by TEXT NOT NULL, timestamp TEXT NOT NULL,
                status TEXT NOT NULL, details TEXT
            )
        ''')
        
        # NEW: Table to store saved capacity plans
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capacity_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_name TEXT NOT NULL UNIQUE,
                queues TEXT NOT NULL,
                start_month TEXT NOT NULL,
                saved_by TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                plan_data TEXT NOT NULL -- Stores input data as a JSON string
            )
        ''')
        conn.commit()

def to_excel_bytes(data, index=True):
    """Converts a DataFrame into Excel format as bytes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=index, sheet_name="Sheet1")
    return output.getvalue()

def get_significance_rating(metric_value, metric_type='wmape'):
    """Returns a rating string based on an error metric's value."""
    if metric_type.lower() == 'wmape':
        if metric_value <= 5: return "⭐⭐⭐ Excellent"
        elif metric_value <= 10: return "⭐⭐ Good"
        elif metric_value <= 15: return "⭐ Fair"
        else: return "⚠️ Needs Review"
    elif metric_type.lower() == 'mae':
        if metric_value <= 2: return "⭐⭐⭐ Excellent"
        elif metric_value <= 5: return "⭐⭐ Good"
        elif metric_value <= 10: return "⭐ Fair"
        else: return "⚠️ Needs Review"
    return "N/A"

# --- SECTION 3: DATABASE MANAGEMENT & AUTHENTICATION HELPER FUNCTIONS ---
DATABASE_NAME = 'wiseinsights_db.db'

@contextmanager
def get_db_connection():
    """Provides a context-managed database connection."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def create_db_tables():
    """Creates all necessary tables for users, runs, and capacity plans."""
    print("Attempting to create database tables...")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL,
                    can_view_shrinkage INTEGER NOT NULL, can_view_volume INTEGER NOT NULL,
                    can_view_capacity INTEGER NOT NULL, can_manage_schedules INTEGER NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_type TEXT NOT NULL,
                    job_run_by TEXT NOT NULL, timestamp TEXT NOT NULL,
                    status TEXT NOT NULL, details TEXT
                )
            ''')
            
            # Table to store saved capacity plans
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS capacity_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_name TEXT NOT NULL UNIQUE,
                    queues TEXT NOT NULL,
                    start_month TEXT NOT NULL,
                    saved_by TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    plan_data TEXT NOT NULL -- Stores input data as a JSON string
                )
            ''')
            conn.commit()
            print("Database tables created successfully.")
    except Exception as e:
        print(f"Error creating database tables: {e}")

def add_user(username, password, role, permissions):
    """Adds a new user to the database with a hashed password."""
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, can_view_shrinkage, can_view_volume, can_view_capacity, can_manage_schedules)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, role, permissions['can_view_shrinkage'], permissions['can_view_volume'], permissions['can_view_capacity'], permissions['can_manage_schedules']))
            conn.commit()
            print(f"User '{username}' added successfully.")
            return True
        except sqlite3.IntegrityError:
            print(f"Error: Username '{username}' already exists.")
            return False
def bulk_add_users(df):
    """Adds multiple users from a DataFrame, skipping duplicates."""
    success_count = 0
    fail_count = 0
    for _, row in df.iterrows():
        permissions = {
            'can_view_shrinkage': 1 if str(row['can_view_shrinkage']).lower() == 'yes' else 0,
            'can_view_volume': 1 if str(row['can_view_volume']).lower() == 'yes' else 0,
            'can_view_capacity': 1 if str(row['can_view_capacity']).lower() == 'yes' else 0,
            'can_manage_schedules': 1 if str(row['can_manage_schedules']).lower() == 'yes' else 0,
        }
        if add_user(row['username'], row['password'], row['role'], permissions):
            success_count += 1
        else:
            fail_count += 1
    return success_count, fail_count

def update_user(username, new_password, role, permissions):
    """Updates an existing user's details and optionally their password."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if new_password:
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute('''
                UPDATE users SET password_hash=?, role=?, can_view_shrinkage=?, can_view_volume=?, can_view_capacity=?, can_manage_schedules=?
                WHERE username=?
            ''', (password_hash, role, permissions['can_view_shrinkage'], permissions['can_view_volume'], permissions['can_view_capacity'], permissions['can_manage_schedules'], username))
        else:
            cursor.execute('''
                UPDATE users SET role=?, can_view_shrinkage=?, can_view_volume=?, can_view_capacity=?, can_manage_schedules=?
                WHERE username=?
            ''', (role, permissions['can_view_shrinkage'], permissions['can_view_volume'], permissions['can_view_capacity'], permissions['can_manage_schedules'], username))
        conn.commit()

def delete_user(username):
    """Deletes a user from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE username=?', (username,))
        conn.commit()

@st.cache_data(ttl=600)
def get_all_users():
    """Fetches all users from the database."""
    with get_db_connection() as conn:
        df = pd.read_sql_query('SELECT username, role, can_view_shrinkage, can_view_volume, can_view_capacity, can_manage_schedules FROM users', conn)
        return df

def get_user_by_username(username):
    """Fetches a single user's record for authentication."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        return cursor.fetchone()

def check_password_hash(password, password_hash):
    """Checks a plaintext password against a hash."""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def log_job_run(job_type, status, error_code, time_took, details):
    """Appends a record to the unified run history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO runs (job_type, job_run_by, timestamp, status, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (job_type, st.session_state.get("username", "N/A"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, str(details)))
        conn.commit()
    get_run_history.clear()

@st.cache_data
def get_run_history():
    """Fetches run history from the database."""
    with get_db_connection() as conn:
        df = pd.read_sql_query('SELECT job_type, job_run_by, timestamp, status, details FROM runs ORDER BY timestamp DESC', conn)
    return df
def save_capacity_plan(plan_name, queues, start_month, plan_data):
    """Saves a capacity plan's input data to the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO capacity_plans (plan_name, queues, start_month, saved_by, timestamp, plan_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                plan_name, 
                ",".join(queues), 
                start_month, 
                st.session_state.get("username", "N/A"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps(plan_data)
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return "A plan with this name already exists."
        except Exception as e:
            return f"An error occurred: {e}"

@st.cache_data(ttl=60)
def get_saved_plan_names():
    """Fetches a list of all saved capacity plan names."""
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT plan_name FROM capacity_plans ORDER BY timestamp DESC", conn)
        return df['plan_name'].tolist()

def load_capacity_plan(plan_name):
    """Loads a saved capacity plan's data from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT queues, start_month, plan_data FROM capacity_plans WHERE plan_name=?", (plan_name,))
        result = cursor.fetchone()
        if result:
            return {
                "queues": result['queues'].split(','),
                "start_month": result['start_month'],
                "plan_data": json.loads(result['plan_data'])
            }
        return None
    
def initialize_session_state():
    if 'password_correct' not in st.session_state: st.session_state.password_correct = False
    if 'username' not in st.session_state: st.session_state.username = ""
    if 'run_history' not in st.session_state: st.session_state.run_history = []
    if 'shrink_show_graphs' not in st.session_state: st.session_state.shrink_show_graphs = False
    if 'shrinkage_results' not in st.session_state: st.session_state.shrinkage_results = None
    if 'manual_shrinkage_results' not in st.session_state: st.session_state.manual_shrinkage_results = None
    if 'volume_monthly_results' not in st.session_state: st.session_state.volume_monthly_results = None
    if 'volume_daily_results' not in st.session_state: st.session_state.volume_daily_results = None
    if 'manual_volume_results' not in st.session_state: st.session_state.manual_volume_results = None
    if 'backtest_volume_results' not in st.session_state: st.session_state.backtest_volume_results = None
    if 'capacity_model_results' not in st.session_state: st.session_state.capacity_model_results = None
    # Add keys for the new monthly planner
    if 'capacity_plan_inputs' not in st.session_state: st.session_state.capacity_plan_inputs = {}
    if 'loaded_plan' not in st.session_state: st.session_state.loaded_plan = None

def save_capacity_plan(plan_name, queues, start_month, plan_data):
    """Saves a capacity plan's input data to the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO capacity_plans (plan_name, queues, start_month, saved_by, timestamp, plan_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                plan_name, 
                ",".join(queues), 
                start_month, 
                st.session_state.get("username", "N/A"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps(plan_data)
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return "A plan with this name already exists."
        except Exception as e:
            return f"An error occurred: {e}"

@st.cache_data(ttl=60)
def get_saved_plan_names():
    """Fetches a list of all saved capacity plan names."""
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT plan_name FROM capacity_plans ORDER BY timestamp DESC", conn)
        return df['plan_name'].tolist()

def load_capacity_plan(plan_name):
    """Loads a saved capacity plan's data from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT queues, start_month, plan_data FROM capacity_plans WHERE plan_name=?", (plan_name,))
        result = cursor.fetchone()
        if result:
            return {
                "queues": result['queues'].split(','),
                "start_month": result['start_month'],
                "plan_data": json.loads(result['plan_data'])
            }
        return None


# --- SECTION 4: SHRINKAGE FORECAST ENGINE ---

def shrink_archive_run_results(run_ts, results_dict):
    """Saves all result DataFrames from a shrinkage forecast into a timestamped folder."""
    run_dir = os.path.join(SHRINK_ARCHIVE_DIR, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    
    for forecast_type, data in results_dict['forecasts'].items():
        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                file_path = os.path.join(run_dir, f"{forecast_type.lower()}_{key}.xlsx")
                df.to_excel(file_path)

def shrink_create_zip_for_run(run_ts):
    """Creates a ZIP archive of a given shrinkage forecast run folder."""
    run_dir = os.path.join(SHRINK_ARCHIVE_DIR, run_ts)
    if not os.path.isdir(run_dir): return None
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for file in files:
                zf.write(os.path.join(root, file), arcname=file)
    return zip_buffer.getvalue()

def shrink_forecast_moving_average(ts, steps, window=7, freq='D'):
    """Generates a moving average forecast."""
    if len(ts) < window: window = max(1, len(ts))
    val = ts.rolling(window=window, min_periods=1).mean().iloc[-1]
    future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.Series([val] * steps, index=future_idx).clip(0, 1)

def shrink_forecast_naive(ts, steps, freq='D'):
    """Generates a naive forecast (repeats the last value)."""
    last_val = ts.iloc[-1] if not ts.empty else 0
    future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.Series([last_val] * steps, index=future_idx).clip(0, 1)

def shrink_forecast_seasonal_naive(ts, steps, freq='D', seasonal_periods=7):
    """Generates a seasonal naive forecast (repeats the last season's pattern)."""
    if len(ts) < seasonal_periods: return shrink_forecast_naive(ts, steps, freq)
    future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    seasonal_pattern = [ts.iloc[-seasonal_periods:].iloc[i % seasonal_periods] for i in range(steps)]
    return pd.Series(seasonal_pattern, index=future_idx).clip(0, 1)

def shrink_forecast_prophet(ts, steps, holidays_df=None, prophet_params=None):
    """Generates a Prophet forecast."""
    if prophet_params is None: prophet_params = {}
    if len(ts) < 5: return shrink_forecast_moving_average(ts, steps, window=len(ts), freq=ts.index.freq)
    df = ts.reset_index(); df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    model = Prophet(holidays=holidays_df, **prophet_params).fit(df)
    future = model.make_future_dataframe(periods=steps, freq=ts.index.freq)
    forecast = model.predict(future)
    preds = forecast[['ds', 'yhat']].tail(steps).set_index('ds')
    return preds['yhat'].clip(0, 1)

def shrink_create_forecast_plot(historical_ts, forecast_series, queue_name, shrinkage_type="Total"):
    """Creates a Plotly graph comparing historical data and forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_ts.index, y=historical_ts.values, mode='lines', name=f'Historical {shrinkage_type}', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines', name=f'Forecasted {shrinkage_type}', line=dict(color='crimson', dash='dash')))
    fig.update_layout(title=f'📈 {shrinkage_type} Shrinkage Forecast: {queue_name}', xaxis_title='Date', yaxis_title='Shrinkage %', yaxis=dict(tickformat=".1%"))
    return fig

def shrink_create_aggregated_plot(historical_df, forecast_df, aggregation, shrinkage_type="Total"):
    """Creates a Plotly graph for aggregated (weekly/monthly) forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df.mean(axis=1).index, y=historical_df.mean(axis=1).values, mode='lines', name='Historical Avg', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=forecast_df.mean(axis=1).index, y=forecast_df.mean(axis=1).values, mode='lines', name='Forecasted Avg', line=dict(color='crimson', dash='dash')))
    fig.update_layout(title=f'📊 {aggregation} Shrinkage Forecast vs. Historical (Aggregated)', xaxis_title='Date', yaxis_title='Shrinkage %', yaxis=dict(tickformat=".1%"))
    return fig

def shrink_backtest_forecast(ts, horizon, method):
    """Performs backtesting to evaluate a model's historical accuracy."""
    forecasts, actuals = pd.Series(dtype=float), ts.copy()
    for i in range(len(ts) - horizon, 0, -horizon):
        train = ts.iloc[:i]
        preds = pd.Series(dtype=float)
        try:
            if method == 'Prophet': preds = shrink_forecast_prophet(train, horizon)
            elif 'Moving Average' in method: preds = shrink_forecast_moving_average(train, horizon)
            elif 'Seasonal Naive' in method: preds = shrink_forecast_seasonal_naive(train, horizon)
            if not preds.empty: forecasts = pd.concat([forecasts, preds])
        except Exception: continue
    return forecasts.reindex(actuals.index), actuals

@st.cache_data
def shrink_process_raw_data(raw_df):
    """Processes the raw uploaded Excel data for shrinkage analysis."""
    df = raw_df.copy()
    rename_map = {'Activity End Time (UTC) Date': 'Date', 'Activity Start Time (UTC) Hour of Day': 'Hour', 'Site Name': 'Queue', 'Scheduled Paid Time (h)': 'Scheduled_Hours', 'Absence Time [Planned] (h)': 'Planned_Absence_Hours', 'Absence Time [Unplanned] (h)': 'Unplanned_Absence_Hours'}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    required_cols = ['Date', 'Hour', 'Queue', 'Scheduled_Hours', 'Planned_Absence_Hours', 'Unplanned_Absence_Hours']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Shrinkage file missing required columns: {', '.join(missing)}")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Timestamp'] = df.apply(lambda row: row['Date'].replace(hour=int(row['Hour'])), axis=1)
    df.set_index('Timestamp', inplace=True)
    df['Planned_Shrinkage'] = np.where(df['Scheduled_Hours'] > 0, df['Planned_Absence_Hours'] / df['Scheduled_Hours'], 0).clip(0, 1)
    df['Unplanned_Shrinkage'] = np.where(df['Scheduled_Hours'] > 0, df['Unplanned_Absence_Hours'] / df['Scheduled_Hours'], 0).clip(0, 1)
    df['Total_Shrinkage'] = (df['Planned_Shrinkage'] + df['Unplanned_Shrinkage']).clip(0, 1)
    return df

@st.cache_data
def shrink_run_forecasting(_df, forecast_horizon_days, shrinkage_col):
    """Orchestrates the model competition to find the best forecast."""
    queues = _df["Queue"].unique()
    all_forecasts, errors, historical_ts_map = {}, [], {}
    for queue in queues:
        ts = _df[_df["Queue"] == queue][shrinkage_col].resample('D').mean().fillna(0)
        historical_ts_map[queue] = ts
        if len(ts) < 7: continue
        test_size = min(forecast_horizon_days, len(ts) - 3)
        train, test = ts[:-test_size], ts[-test_size:]
        fcts = {"Seasonal Naive (7-day)": shrink_forecast_seasonal_naive, "Moving Average (7-day)": shrink_forecast_moving_average, "Prophet": shrink_forecast_prophet}
        for name, func in fcts.items():
            try:
                preds = func(train, len(test))
                if not preds.empty:
                    errors.append({"MAE": np.mean(np.abs(test.values - preds.values)), "Queue": queue, "Method": name})
                future_preds = func(ts, forecast_horizon_days)
                if not future_preds.empty: all_forecasts[(queue, name)] = future_preds
            except Exception: continue
    if not errors: return pd.DataFrame(), pd.DataFrame(), {}
    error_df = pd.DataFrame(errors)
    best_methods = error_df.loc[error_df.groupby("Queue")["MAE"].idxmin()]
    best_forecast_dict = {row["Queue"]: all_forecasts.get((row["Queue"], row["Method"])) for _, row in best_methods.iterrows() if all_forecasts.get((row["Queue"], row["Method"])) is not None}
    best_forecast_df = pd.DataFrame(best_forecast_dict).clip(0, 0.7)
    return best_forecast_df, best_methods, historical_ts_map

@st.cache_data
def shrink_generate_interval_forecast(_daily_forecast_df, _historical_df, shrinkage_col):
    """Disaggregates a daily forecast into interval-level forecasts using historical patterns."""
    if _daily_forecast_df.empty or _historical_df.empty: return pd.DataFrame()
    hist = _historical_df.copy()
    hist['Hour'] = hist.index.hour
    hist['DayOfWeek'] = hist.index.day_name()
    profiles = pd.merge(hist.groupby(['Queue', 'DayOfWeek', 'Hour'])[shrinkage_col].mean().reset_index(), hist.groupby(['Queue', 'DayOfWeek'])[shrinkage_col].mean().reset_index().rename(columns={shrinkage_col: 'Hist_Daily_Avg'}), on=['Queue', 'DayOfWeek'])
    
    all_interval_forecasts = []
    for queue in _daily_forecast_df.columns:
        for date, daily_forecast_val in _daily_forecast_df[queue].items():
            day_profile = profiles[(profiles['Queue'] == queue) & (profiles['DayOfWeek'] == date.strftime('%A'))].copy()
            if day_profile.empty or day_profile['Hist_Daily_Avg'].iloc[0] == 0: continue
            adjustment_factor = daily_forecast_val / day_profile['Hist_Daily_Avg'].iloc[0]
            
            day_profile.loc[:, f'Forecasted_{shrinkage_col}'] = (day_profile[shrinkage_col] * adjustment_factor).clip(0, 0.7)
            day_profile.loc[:, 'Timestamp'] = day_profile['Hour'].apply(lambda h: date.replace(hour=int(h)))
            
            all_interval_forecasts.append(day_profile)
    
    if not all_interval_forecasts: return pd.DataFrame()
    return pd.concat(all_interval_forecasts)

@st.cache_data
def shrink_generate_aggregated_forecasts(_daily_forecast_df):
    """Rolls up daily forecasts into weekly and monthly summaries."""
    if _daily_forecast_df.empty: return pd.DataFrame(), pd.DataFrame()
    weekly_df = _daily_forecast_df.resample('W-MON', label='left', closed='left').mean()
    if not weekly_df.empty: weekly_df.loc['Subtotal'] = weekly_df.mean()
    monthly_df = _daily_forecast_df.resample('M').mean()
    if not monthly_df.empty: monthly_df.loc['Subtotal'] = monthly_df.mean()
    return weekly_df, monthly_df

# --- SECTION 5: VOLUME FORECAST ENGINE ---

COUNTRY_CODES = {"United States": "US", "United Kingdom": "GB", "Canada": "CA", "Australia": "AU", "Germany": "DE", "France": "FR", "Spain": "ES", "Italy": "IT", "India": "IN", "Brazil": "BR", "Mexico": "MX", "Japan": "JP", "None": "NONE"}
COUNTRY_NAMES = sorted(list(COUNTRY_CODES.keys()))

def vol_prepare_full_data(df):
    """Processes raw uploaded data for volume forecasting."""
    if not {"Date", "Interval", "Volume", "Queue"}.issubset(df.columns):
        raise ValueError("Volume file missing required columns: Date, Interval, Volume, Queue")
    
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df["Interval"]):
        df["Interval_td"] = pd.to_timedelta(df["Interval"] * 24 * 3600, unit="s")
    else:
        df["Interval_td"] = pd.to_timedelta(df["Interval"].astype(str), errors="coerce")
    
    df.dropna(subset=['Interval_td'], inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Date"]) + df["Interval_td"]
    return df.set_index("Timestamp")

def vol_calculate_error_metrics(actuals, preds):
    """Calculates ME, MAE, RMSE, and wMAPE for model evaluation."""
    actuals, preds = np.array(actuals), np.array(preds)
    me = np.mean(preds - actuals)
    mae = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    wmape = np.sum(np.abs(preds - actuals)) / np.sum(np.abs(actuals)) * 100 if np.sum(np.abs(actuals)) > 0 else 0
    return {"ME": round(me, 2), "MAE": round(mae, 2), "RMSE": round(rmse, 2), "wMAPE": round(wmape, 2)}

def vol_create_forecast_plot(historical_ts, forecast_series, queue_name, period_name="Period"):
    """Creates a Plotly graph comparing historical volume and forecasts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_ts.index, y=historical_ts.values, mode='lines', name='Historical Volume', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines', name='Best Forecast', line=dict(color='crimson', dash='dash')))
    fig.update_layout(title=f'Best Forecast vs. Historical Data for: {queue_name}', xaxis_title=period_name, yaxis_title='Volume')
    return fig

def vol_forecast_naive(ts, steps, freq='MS'):
    """Generates a naive forecast."""
    try:
        last_val = ts.iloc[-1] if not ts.empty else 0
        future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return pd.Series([last_val] * steps, index=future_idx).round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_seasonal_naive(ts, steps, freq='MS', seasonal_periods=12):
    """Generates a seasonal naive forecast."""
    try:
        if len(ts) < seasonal_periods: return vol_forecast_naive(ts, steps, freq)
        seasonal_vals = ts.tail(seasonal_periods)
        future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        seasonal_pattern = [seasonal_vals.iloc[i % seasonal_periods] for i in range(steps)]
        return pd.Series(seasonal_pattern, index=future_idx).round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_moving_average(ts, steps, window=3, freq='MS'):
    """Generates a moving average forecast."""
    try:
        if len(ts) < window: window = max(1, len(ts))
        val = ts.rolling(window=window, min_periods=1).mean().iloc[-1]
        future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return pd.Series([val] * steps, index=future_idx).round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_holtwinters(ts, steps, freq='MS', seasonal_periods=12):
    """Generates a Holt-Winters (triple exponential smoothing) forecast."""
    try:
        if len(ts) < seasonal_periods * 2: return pd.Series(dtype=float)
        model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
        fc = model.forecast(steps)
        fc.index = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return fc.round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_sarima(ts, steps, order, seasonal_order, freq='MS'):
    """Generates a SARIMA forecast."""
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fc = fit.get_forecast(steps=steps).predicted_mean
        fc.index = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return fc.round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_prophet(ts, steps, freq='MS', holidays=None):
    """Generates a Prophet forecast."""
    try:
        if len(ts) < 5: return vol_forecast_moving_average(ts, steps, window=len(ts), freq=freq)
        df = ts.reset_index(); df.columns = ['ds', 'y']
        model = Prophet(holidays=holidays).fit(df)
        future = model.make_future_dataframe(periods=steps, freq=freq)
        forecast = model.predict(future)
        preds = forecast[['ds', 'yhat']].tail(steps)
        preds.set_index('ds', inplace=True)
        return preds['yhat'].round().clip(lower=0)
    except Exception: return pd.Series(dtype=float)

def vol_create_features(df):
    """Creates time series features from a datetime index."""
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    return df

def vol_forecast_ml(ts, steps, model, freq='D'):
    """Generic function for ML models."""
    try:
        df = pd.DataFrame({'y': ts})
        df = vol_create_features(df)
        
        features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
        X_train, y_train = df[features], df['y']
        
        model.fit(X_train, y_train)
        
        future_dates = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        future_df = pd.DataFrame(index=future_dates)
        future_df = vol_create_features(future_df)
        
        predictions = model.predict(future_df[features])
        
        return pd.Series(predictions, index=future_dates).round()
    except Exception:
        return pd.Series(dtype=float)

def vol_run_monthly_forecasting(_df, horizon):
    """Orchestrates the monthly volume forecasting process by running a model competition."""
    df_prep = vol_prepare_full_data(_df)
    queues = df_prep["Queue"].unique()
    all_forecasts, errors = {}, []
    
    progress = st.progress(0, "Starting monthly volume forecast competition...")

    for i, q in enumerate(queues):
        progress.progress((i + 1) / len(queues), f"Processing Monthly Queue: {q}")
        ts = df_prep[df_prep["Queue"] == q]["Volume"].resample('MS').sum()
        
        if len(ts) < 3: 
            st.warning(f"Skipping queue {q}: insufficient data (requires at least 3 months).")
            continue
            
        test_size = 1
        if len(ts) <= 5: test_size = 0
        
        train, test = (ts[:-test_size], ts[-test_size:]) if test_size > 0 else (ts, ts[-1:])

        fcts = {
            "Naive": lambda d, s: vol_forecast_naive(d, s, freq='MS'),
            "Seasonal Naive": lambda d, s: vol_forecast_seasonal_naive(d, s, freq='MS'),
            "Moving Average (3m)": lambda d, s: vol_forecast_moving_average(d, s, window=3, freq='MS'),
            "Holt-Winters": lambda d, s: vol_forecast_holtwinters(d, s, freq='MS'),
            "SARIMA": lambda d, s: vol_forecast_sarima(d, s, (1,1,1), (1,1,1,12), freq='MS'),
            "Prophet": lambda d, s: vol_forecast_prophet(d, s, freq='MS'),
            "Random Forest": lambda d, s: vol_forecast_ml(d, s, RandomForestRegressor(n_estimators=100, random_state=42), freq='MS'),
        }
        
        for name, func in fcts.items():
            try:
                if test_size > 0:
                    preds = func(train, len(test))
                    if not preds.empty and len(preds) == len(test):
                        err = vol_calculate_error_metrics(test.values, preds.values)
                        err.update({"Queue": q, "Method": name})
                        errors.append(err)
                
                future_preds = func(ts, horizon)
                if not future_preds.empty:
                    all_forecasts[(q, name)] = future_preds
            except Exception: 
                continue

    progress.empty()
    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    all_forecasts_df = pd.DataFrame(all_forecasts).fillna(0)
    all_forecasts_df.columns = pd.MultiIndex.from_tuples(all_forecasts_df.columns, names=["Queue", "Method"])
    all_forecasts_df.index = pd.to_datetime(all_forecasts_df.index).strftime('%b-%y')

    if not errors:
        st.warning("Low data volume: Could not perform model competition. Defaulting to the first successful model.")
        best_forecast_dict = {}
        for q in queues:
            if q in all_forecasts_df.columns.get_level_values('Queue'):
                best_forecast_dict[q] = all_forecasts_df[q].iloc[:, 0]
        best_forecast_df = pd.DataFrame(best_forecast_dict)
        return all_forecasts_df, pd.DataFrame(), best_forecast_df, pd.DataFrame(), df_prep

    error_df = pd.DataFrame(errors).dropna(subset=['wMAPE'])
    best_methods_df = error_df.loc[error_df.groupby("Queue")["wMAPE"].idxmin()].set_index("Queue")
    
    best_forecast_dict = {
        q: all_forecasts_df[(q, best_methods_df.loc[q]["Method"])] 
        for q in best_methods_df.index if (q, best_methods_df.loc[q]["Method"]) in all_forecasts_df.columns
    }
    best_forecast_df = pd.DataFrame(best_forecast_dict)

    return all_forecasts_df, error_df, best_forecast_df, best_methods_df, df_prep

@st.cache_data
def vol_run_daily_forecasting(_df, horizon, country_code):
    """Orchestrates a daily volume model competition."""
    df_prep = vol_prepare_full_data(_df)
    queues = df_prep["Queue"].unique()
    all_forecasts, errors = {}, []
    holidays = pd.DataFrame(CountryHoliday(country_code, years=range(datetime.now().year-2, datetime.now().year+2)).items()) if country_code != "NONE" else None
    if holidays is not None:
        holidays.columns = ['ds', 'holiday']
        
    progress = st.progress(0, "Starting daily volume forecast competition...")
    for i, q in enumerate(queues):
        progress.progress((i + 1) / len(queues), f"Processing Daily Queue: {q}")
        ts = df_prep[df_prep["Queue"] == q]["Volume"].resample('D').sum()
        
        if len(ts) < 14:
            st.warning(f"Skipping queue {q} for daily forecast: insufficient data (requires at least 14 days).")
            continue
            
        test_size = min(horizon, len(ts) - 7)
        if test_size <= 0: test_size = 0
        
        train, test = (ts.iloc[:-test_size], ts.iloc[-test_size:]) if test_size > 0 else (ts, ts[-1:])
        
        daily_fcts = {
            "Seasonal Naive (7d)": lambda d, s: vol_forecast_seasonal_naive(d, s, freq='D', seasonal_periods=7),
            "Moving Average (7d)": lambda d, s: vol_forecast_moving_average(d, s, window=7, freq='D'),
            "Holt-Winters (Seasonal=7)": lambda d, s: vol_forecast_holtwinters(d, s, freq='D', seasonal_periods=7),
            "Prophet": lambda d, s: vol_forecast_prophet(d, s, freq='D', holidays=holidays),
            "Linear Regression": lambda d, s: vol_forecast_ml(d, s, LinearRegression(), freq='D'),
            "Random Forest": lambda d, s: vol_forecast_ml(d, s, RandomForestRegressor(n_estimators=100, random_state=42), freq='D'),
        }

        for name, func in daily_fcts.items():
            try:
                if test_size > 0:
                    preds = func(train, len(test))
                    if not preds.empty and len(preds) == len(test):
                        err = vol_calculate_error_metrics(test.values, preds.values)
                        err.update({"Queue": q, "Method": name})
                        errors.append(err)
                
                future_preds = func(ts, horizon)
                if not future_preds.empty:
                    all_forecasts[(q, name)] = future_preds
            except Exception:
                continue

    progress.empty()
    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_forecasts_df = pd.DataFrame(all_forecasts).fillna(0)
    all_forecasts_df.columns = pd.MultiIndex.from_tuples(all_forecasts_df.columns, names=["Queue", "Method"])

    if not errors:
        st.warning("Low data volume: Could not perform model competition. Defaulting to the first successful model.")
        best_forecast_dict = {}
        for q in queues:
            if q in all_forecasts_df.columns.get_level_values('Queue'):
                best_forecast_dict[q] = all_forecasts_df[q].iloc[:, 0]
        best_forecast_df = pd.DataFrame(best_forecast_dict)
        return all_forecasts_df, pd.DataFrame(), best_forecast_df, pd.DataFrame(), df_prep

    error_df = pd.DataFrame(errors).dropna(subset=['wMAPE'])
    best_methods_df = error_df.loc[error_df.groupby("Queue")["wMAPE"].idxmin()].set_index("Queue")
    
    best_forecast_dict = {
        q: all_forecasts_df[(q, best_methods_df.loc[q]["Method"])] 
        for q in best_methods_df.index if (q, best_methods_df.loc[q]["Method"]) in all_forecasts_df.columns
    }
    best_forecast_df = pd.DataFrame(best_forecast_dict)
    
    return all_forecasts_df, error_df, best_forecast_df, best_methods_df, df_prep

def vol_backtest_forecast(ts, _model_func, horizon):
    """Performs backtesting for a given model and time series."""
    forecasts = pd.Series(dtype=float)
    
    min_train_size = horizon * 2
    if len(ts) < min_train_size:
        return pd.DataFrame()

    for i in range(len(ts) - horizon, min_train_size - 1, -horizon):
        train = ts.iloc[:i]
        preds = _model_func(train, horizon)
        if not preds.empty:
            forecasts = pd.concat([forecasts, preds])
            
    if forecasts.empty:
        return pd.DataFrame()

    results = pd.DataFrame({'Actual': ts, 'Forecast': forecasts}).dropna()
    return results

def vol_generate_interval_forecast(daily_forecast_df, historical_df):
    """Disaggregates a daily volume forecast into interval-level forecasts."""
    if daily_forecast_df.empty or historical_df.empty: return pd.DataFrame()
    
    hist = historical_df.copy()
    hist['DayOfWeek'] = hist.index.day_name()
    hist['Time'] = hist.index.time
    
    profile = hist.groupby(['DayOfWeek', 'Time'])['Volume'].mean().reset_index()
    
    profile['Daily_Total'] = profile.groupby('DayOfWeek')['Volume'].transform('sum')
    profile['Interval_Ratio'] = profile['Volume'] / profile['Daily_Total']
    profile.loc[profile['Daily_Total'] == 0, 'Interval_Ratio'] = 0 
    
    all_interval_forecasts = []
    
    if isinstance(daily_forecast_df.columns, pd.MultiIndex):
        df_to_process = daily_forecast_df.copy()
        df_to_process.columns = ['_'.join(map(str, col)) for col in df_to_process.columns]
        queue_names = {col: col.split('_')[0] for col in df_to_process.columns}
    else:
        df_to_process = daily_forecast_df
        queue_names = {col: col for col in df_to_process.columns}

    for col_name in df_to_process.columns:
        queue = queue_names[col_name]
        for date, daily_total in df_to_process[col_name].items():
            day_of_week = date.strftime('%A')
            day_profile = profile[profile['DayOfWeek'] == day_of_week].copy()
            if day_profile.empty: continue
            
            day_profile['Forecast_Volume'] = day_profile['Interval_Ratio'] * daily_total
            day_profile['Timestamp'] = day_profile['Time'].apply(lambda t: datetime.combine(date.date(), t))
            day_profile['Queue'] = queue
            all_interval_forecasts.append(day_profile)
            
    if not all_interval_forecasts: return pd.DataFrame()
    return pd.concat(all_interval_forecasts)[['Timestamp', 'Queue', 'Forecast_Volume']].round()

def vol_generate_monthly_interval_forecast(monthly_forecast_df, historical_df):
    """Disaggregates a monthly volume forecast into interval-level forecasts."""
    if monthly_forecast_df.empty or historical_df.empty:
        return pd.DataFrame()
    
    hist = historical_df.copy()
    hist['Month'] = hist.index.month
    hist['DayOfWeek'] = hist.index.day_name()
    hist['Time'] = hist.index.time
    
    day_time_profile = hist.groupby(['DayOfWeek', 'Time'])['Volume'].mean().reset_index()
    day_time_profile['Daily_Total'] = day_time_profile.groupby('DayOfWeek')['Volume'].transform('sum')
    day_time_profile['Interval_Ratio'] = day_time_profile['Volume'] / day_time_profile['Daily_Total']
    day_time_profile.loc[day_time_profile['Daily_Total'] == 0, 'Interval_Ratio'] = 0
    
    month_day_profile = hist.groupby(['Month', 'DayOfWeek'])['Volume'].sum().reset_index()
    month_day_profile['Monthly_Total'] = month_day_profile.groupby('Month')['Volume'].transform('sum')
    month_day_profile['Day_Ratio'] = month_day_profile['Volume'] / month_day_profile['Monthly_Total']
    month_day_profile.loc[month_day_profile['Monthly_Total'] == 0, 'Day_Ratio'] = 0
    
    all_interval_forecasts = []
    
    for queue in monthly_forecast_df.columns:
        queue_hist = hist[hist['Queue'] == queue]
        if queue_hist.empty: continue
        
        for period_str, monthly_total in monthly_forecast_df[queue].items():
            try:
                forecast_month_start = pd.to_datetime(period_str, format='%b-%y')
            except ValueError:
                continue

            days_in_month = pd.date_range(start=forecast_month_start, end=forecast_month_start + pd.offsets.MonthEnd(0))
            
            month_profile = month_day_profile[month_day_profile['Month'] == forecast_month_start.month]
            if month_profile.empty:
                month_profile = hist.groupby('DayOfWeek')['Volume'].sum().reset_index()
                month_profile['Monthly_Total'] = month_profile['Volume'].sum()
                month_profile['Day_Ratio'] = month_profile['Volume'] / month_profile['Monthly_Total']
                
            daily_distribution = {day.day_name(): 0 for day in days_in_month}
            for day in days_in_month:
                daily_distribution[day.day_name()] += 1

            daily_totals = {}
            total_ratio_sum = 0
            for day_name, count in daily_distribution.items():
                ratio = month_profile[month_profile['DayOfWeek'] == day_name]['Day_Ratio'].values
                if len(ratio) > 0:
                    total_ratio_sum += ratio[0] * count
            
            if total_ratio_sum == 0: continue

            for day in days_in_month:
                day_name = day.day_name()
                day_ratio = month_profile[month_profile['DayOfWeek'] == day_name]['Day_Ratio'].values
                if len(day_ratio) > 0:
                    daily_totals[day] = (monthly_total * day_ratio[0]) / total_ratio_sum
            
            for day, daily_total_val in daily_totals.items():
                day_profile_intervals = day_time_profile[day_time_profile['DayOfWeek'] == day.day_name()].copy()
                if day_profile_intervals.empty: continue
                
                day_profile_intervals['Forecast_Volume'] = day_profile_intervals['Interval_Ratio'] * daily_total_val
                day_profile_intervals['Timestamp'] = day_profile_intervals['Time'].apply(lambda t: datetime.combine(day.date(), t))
                day_profile_intervals['Queue'] = queue
                all_interval_forecasts.append(day_profile_intervals)

    if not all_interval_forecasts: return pd.DataFrame()
    return pd.concat(all_interval_forecasts)[['Timestamp', 'Queue', 'Forecast_Volume']].round()

# --- SECTION 6: CAPACITY PLANNING HELPER FUNCTIONS ---

def run_workload_model(login_hours, aht_seconds, volume, occupancy, concurrency):
    """Calculates required FTE based on workload, with concurrency."""
    if concurrency <= 0: concurrency = 1
    adjusted_volume = volume / concurrency
    total_handle_time_hours = (adjusted_volume * aht_seconds) / 3600
    base_fte = total_handle_time_hours / login_hours
    adjusted_fte = base_fte / (occupancy / 100)
    return {"Required FTE": math.ceil(adjusted_fte)}

def run_erlang_c_model(login_hours, aht_seconds, volume, sla_target_percent, sla_target_seconds, concurrency):
    """Calculates required agents and wait time using Erlang C."""
    if concurrency <= 0: concurrency = 1
    adjusted_volume = volume / concurrency
    if adjusted_volume == 0: return {"Required HC": 0, "Predicted SL (%)": 100, "Avg Wait (s)": 0}
    
    interval_minutes = login_hours * 60
    # Handle potential division by zero if interval is 0
    if interval_minutes == 0: return {"Required HC": 0, "Predicted SL (%)": 0, "Avg Wait (s)": 0}

    traffic_intensity = (adjusted_volume * (aht_seconds / 60)) / interval_minutes
    num_agents = math.ceil(traffic_intensity)
    if num_agents == 0: num_agents = 1
    
    while True:
        try:
            erlang_b_num = traffic_intensity**num_agents / math.factorial(num_agents)
            erlang_b_den = sum(traffic_intensity**i / math.factorial(i) for i in range(num_agents + 1))
            prob_wait = erlang_b_num / erlang_b_den
            sl = (1 - prob_wait * math.exp(-(num_agents - traffic_intensity) * (sla_target_seconds / aht_seconds))) * 100
        except (OverflowError, ValueError): sl = 100
        if sl >= sla_target_percent or num_agents > traffic_intensity * 2 + 50: break
        num_agents += 1
    
    avg_wait_overall = -1
    if (num_agents - traffic_intensity) > 0:
        avg_wait_for_queued = aht_seconds / (num_agents - traffic_intensity)
        avg_wait_overall = prob_wait * avg_wait_for_queued

    return {"Required HC": num_agents, "Predicted SL (%)": round(sl, 2), "Avg Wait (s)": round(avg_wait_overall, 2)}

def run_monte_carlo_hc_model(login_hours, aht_seconds, volume, sla_target_percent, sla_target_seconds, concurrency, num_simulations=1000):
    """Calculates required HC to meet an SLA target using Monte Carlo simulation."""
    if concurrency <= 0: concurrency = 1
    adjusted_volume = volume / concurrency
    if adjusted_volume == 0: return {"Required HC": 0, "Predicted SLA (%)": 100, "Avg Wait (s)": 0}
    
    interval_seconds = login_hours * 3600
    if interval_seconds <= 0: return {"Required HC": 0, "Predicted SLA (%)": 0, "Avg Wait (s)": 0}

    arrival_rate = adjusted_volume / interval_seconds
    traffic_intensity = (adjusted_volume * aht_seconds) / interval_seconds
    num_agents = math.ceil(traffic_intensity)
    if num_agents == 0: num_agents = 1

    while True:
        wait_times = []
        sim_volume = int(min(adjusted_volume, 1000)) # Limit simulation size for performance
        if arrival_rate <= 0:
            predicted_sla = 100
            break

        # This part is computationally intensive, running a simplified simulation
        inter_arrival_times = np.random.exponential(1/arrival_rate, size=sim_volume)
        arrival_times = np.cumsum(inter_arrival_times)
        service_times = np.random.exponential(aht_seconds, size=sim_volume)
        finish_times = np.zeros(num_agents)
        for i in range(sim_volume):
            agent_available_time = np.min(finish_times)
            start_time = max(arrival_times[i], agent_available_time)
            finish_time = start_time + service_times[i]
            wait_times.append(start_time - arrival_times[i])
            finish_times[np.argmin(finish_times)] = finish_time
        
        sla_met_count = np.sum(np.array(wait_times) <= sla_target_seconds)
        predicted_sla = (sla_met_count / len(wait_times)) * 100 if wait_times else 100
        
        if predicted_sla >= sla_target_percent or num_agents > traffic_intensity * 2 + 50: break
        num_agents += 1
        
    avg_wait = np.mean(wait_times) if wait_times else 0
    return {"Required HC": num_agents, "Predicted SLA (%)": round(predicted_sla, 2), "Avg Wait (s)": round(avg_wait, 2)}


def run_fte_optimization_model(inputs):
    """Finds the most cost-effective mix of FTEs using OR-Tools."""
    required_call_hours = math.ceil((inputs['call_volume'] * inputs['call_aht_sec']) / 3600)
    concurrency = inputs['concurrency'] if inputs['concurrency'] > 0 else 1
    required_email_hours = math.ceil(((inputs['email_volume'] * inputs['email_aht_sec']) / 3600) / concurrency)
    
    model = cp_model.CpModel()
    max_fte = int((required_call_hours + required_email_hours) / inputs['hours_per_fte']) + 10
    num_voice_fte = model.NewIntVar(0, max_fte, 'num_voice_fte')
    num_email_fte = model.NewIntVar(0, max_fte, 'num_email_fte')
    num_blended_fte = model.NewIntVar(0, max_fte, 'num_blended_fte')
    hours_per_fte = int(inputs['hours_per_fte'])
    
    model.Add(num_voice_fte * hours_per_fte + num_blended_fte * hours_per_fte >= required_call_hours)
    model.Add(num_email_fte * hours_per_fte + num_blended_fte * hours_per_fte >= required_email_hours)
    
    total_cost = (num_voice_fte * inputs['cost_voice']) + (num_email_fte * inputs['cost_email']) + (num_blended_fte * inputs['cost_blended'])
    model.Minimize(total_cost)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        voice = solver.Value(num_voice_fte)
        email = solver.Value(num_email_fte)
        blended = solver.Value(num_blended_fte)
        return {
            "Total FTE": voice + email + blended,
            "Minimum Cost ($)": f"${int(solver.ObjectiveValue()):,}"
        }
    return {"Error": "Could not find an optimal solution."}

# --- SECTION 7: RENDER FUNCTIONS FOR EACH TAB ---

def check_password():
    """Renders a login form and loads permissions if credentials are correct."""
    if st.session_state.get("password_correct", False):
        return True

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');
            
            html, body, [class*="st-"] {
                font-family: 'Lato', sans-serif;
            }
            .stApp {
                background-color: #f0f2f5;
            }
            header, footer {
                visibility: hidden !important;
            }
            .main .block-container {
                padding-top: 5rem;
                padding-bottom: 5rem;
                max-width: 1000px;
            }
            .wise-logo-container {
                padding-top: 100px;
            }
            .wise-logo {
                width: 400px;
                height: 220px;
                quality: 90%;
            }
            .wise-tagline {
                font-family: Lato, sans-serif;
                font-size: 40px;
                font-type: bold;
                color: #1c1e21;
                line-height: 1.2;
                padding-bottom: 20px;
                padding-left: 100px;
            }
            div[data-testid="stForm"] {
                background-color: white;
                border: none;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, .1), 0 8px 16px rgba(0, 0, 0, .1);
                padding: 20px;
            }
            div[data-testid="stForm"] .stButton > button {
                width: 100%;
                background-color: #00B9FF;
                color: white;
                font-size: 20px;
                font-weight: bold;
                height: 48px;
                border-radius: 6px;
                border: none;
            }
            div[data-testid="stForm"] hr {
                margin: 20px 0;
            }
        </style>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1.5, 1])

    with left_col:
        st.markdown('<div class="wise-logo-container"><img src="https://i.postimg.cc/vmwmF50z/Remove-background-project.png" class="wise-logo"></div>', unsafe_allow_html=True)
        

    with right_col:
        st.markdown("<h5 style='text-align: center; color: #0d1b3f;'>🔑 Login to WFM Insights</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #3b3b3b;'>Wise Predictions, Smarter Decisions.</p>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            submitted = st.form_submit_button("Log In")

            if submitted:
                user_record = get_user_by_username(username)
                
                if user_record and check_password_hash(password, user_record['password_hash']):
                    st.session_state["password_correct"] = True
                    st.session_state["username"] = user_record['username']
                    
                    permission_cols = ['role', 'can_view_shrinkage', 'can_view_volume', 'can_view_capacity', 'can_manage_schedules']
                    for col in permission_cols:
                        value = user_record[col]
                        if isinstance(value, int):
                            st.session_state[col] = 'yes' if value == 1 else 'no'
                        else:
                            st.session_state[col] = str(value).lower()
                    
                    st.rerun()
                else:
                    st.session_state["password_correct"] = False
                    st.error("😕 The password you’ve entered is incorrect.")
            
            st.markdown('<hr>', unsafe_allow_html=True)

    return False

def render_shrinkage_forecast_tab():
    st.header("Shrinkage Forecast")
    
    title_col, clear_col = st.columns([0.8, 0.2])
    with title_col:
        st.subheader("1. Upload Shrinkage Data")
    with clear_col:
        st.write("")
        if st.button("Clear 🗑️", key="clear_shrinkage_data", use_container_width=True):
            keys_to_clear = ['shrink_uploader', 'shrink_horizon', 'shrink_show_graphs', 'shrinkage_results', 'manual_shrinkage_results']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            log_job_run("Shrinkage", "Cleared", "N/A", 0, "User cleared data for module.")
            st.rerun()

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload Shrinkage Raw Excel Data", type=["xlsx", "xls"], key="shrink_uploader", label_visibility="collapsed")
        with col2:
            st.write("") 
            st.write("")
            shrink_template_df = pd.DataFrame({'Activity End Time (UTC) Date': [pd.Timestamp('2025-01-01')], 'Activity Start Time (UTC) Hour of Day': [8], 'Site Name': ['Queue_A'], 'Scheduled Paid Time (h)': [100.5], 'Absence Time [Planned] (h)': [8.0], 'Absence Time [Unplanned] (h)': [4.5]})
            st.download_button(label="⬇️ Download Data Template", data=to_excel_bytes(shrink_template_df, index=False), file_name="shrinkage_template.xlsx", use_container_width=True)

    if uploaded_file:
        raw_data = pd.read_excel(uploaded_file)
        with st.expander("📄 View Raw Data Preview", expanded=True):
            st.dataframe(raw_data.head(), use_container_width=True, hide_index=True)
        
        with st.container(border=True):
            st.subheader("2. Configure & Run Forecast")
            form_cols = st.columns([1, 3])
            with form_cols[0]:
                horizon = st.number_input("Forecast Horizon (days)", 1, 90, 14, 1, key="shrink_horizon")
            with form_cols[1]:
                st.write("")
                st.write("")
                if st.button("🚀 Run Shrinkage Forecast", key="run_shrinkage", use_container_width=True):
                    st.session_state.start_time = time.time()
                    error_code = "N/A"; status = "Success"
                    details = "N/A"
                    
                    progress_bar = st.progress(0, text="Starting shrinkage forecast...")

                    try:
                        processed_data = shrink_process_raw_data(raw_data)
                        if processed_data is not None:
                            forecasts = {}
                            shrinkage_definitions = {'Total': 'Total_Shrinkage', 'Planned': 'Planned_Shrinkage', 'Unplanned': 'Unplanned_Shrinkage'}
                            
                            for i, (typ, col) in enumerate(shrinkage_definitions.items()):
                                progress_bar.progress((i + 1) / len(shrinkage_definitions), text=f"Forecasting {typ} shrinkage...")
                                daily_df, best_df, hist_map = shrink_run_forecasting(processed_data, int(horizon), col)
                                interval_df = shrink_generate_interval_forecast(daily_df, processed_data, col)
                                weekly_df, monthly_df = shrink_generate_aggregated_forecasts(daily_df)
                                backtest_dict = {q: shrink_backtest_forecast(hist_map[q], horizon, best_df.loc[best_df['Queue']==q, 'Method'].iloc[0] if q in best_df['Queue'].values else 'Prophet') for q in hist_map if len(hist_map[q]) > horizon}
                                forecasts[typ] = {"daily": daily_df, "best": best_df, "hist": hist_map, "interval": interval_df, "weekly": weekly_df, "monthly": monthly_df, "backtest": backtest_dict}
                            
                            st.session_state['shrinkage_results'] = {"forecasts": forecasts, "queues": processed_data["Queue"].unique(), "processed_data": processed_data, "types": ['Total', 'Planned', 'Unplanned'], "cols": shrinkage_definitions, "historical_min_date": processed_data.index.min().date(), "historical_max_date": processed_data.index.max().date()}
                            
                            details = "Shrinkage forecast completed successfully."

                        else: status = "Error"; error_code = "ERR#2"
                    except ValueError: status = "Error"; error_code = "ERR#2"; st.error("Data processing failed. Please check your Excel file."); details="Data processing failed. Please check your Excel file."
                    except Exception as e: status = "Error"; error_code = "ERR#4"; st.error(f"An unexpected error occurred: {e}"); details=str(e)
                    
                    progress_bar.empty()
                    log_job_run("Shrinkage Forecast", status, error_code, time.time() - st.session_state.start_time, details)
                    if status == "Success": 
                        st.success("Shrinkage forecast completed!")
                        time.sleep(1)
                    st.rerun()

    if 'shrinkage_results' in st.session_state and st.session_state.shrinkage_results:
        res = st.session_state.shrinkage_results
        st.subheader("3. View Results")
        with st.container(border=True):
            st.markdown("**Global Display Filters**")
            sel_type = st.radio("Shrinkage Type", res['types'], horizontal=True, key="global_shrinkage_type")
            sel_queues = st.multiselect("Select Queues", ["All"] + list(res['queues']), default=["All"], key="global_queues")
        
        all_possible_queues = list(res['queues'])
        
        if "All" in sel_queues or (set(sel_queues) == set(all_possible_queues)):
            queues_to_show = all_possible_queues
        else:
            queues_to_show = sel_queues

        data = res['forecasts'][sel_type]; col = res['cols'][sel_type]

        tab_hist, tab_monthly, tab_weekly, tab_daily, tab_comp, tab_manual = st.tabs(["Historical Patterns", "Monthly Summary", "Weekly Summary", "Daily Forecast", "Comparison", "Manual"])

        with tab_hist:
            with st.container(border=True):
                st.header("Historical Shrinkage Patterns")
                data_to_pivot = res['processed_data'][res['processed_data']['Queue'].isin(queues_to_show)]
                if not data_to_pivot.empty:
                    st.info(f"Displaying historical patterns for: **{', '.join(queues_to_show)}**")
                    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    df_pivot = data_to_pivot.pivot_table(index=data_to_pivot.index.hour, columns=data_to_pivot.index.day_name(), values=col, aggfunc='mean')
                    df_pivot = df_pivot.reindex(columns=days_order).fillna(0)
                    st.dataframe(df_pivot.style.background_gradient(cmap='RdYlGn_r', axis=None).format("{:.2%}"), use_container_width=True)
                    st.download_button("Download Pattern Table", to_excel_bytes(df_pivot), f"historical_pattern_{'_'.join(queues_to_show)}.xlsx")
                else: st.warning("No data available for the selected queues.")
                with st.expander("View Historical vs. Forecast Graph", expanded=st.session_state.get('shrink_show_graphs', False)):
                    for q_graph in queues_to_show:
                        if q_graph in data['hist'] and q_graph in data['daily']:
                            fig = shrink_create_forecast_plot(data['hist'][q_graph], data['daily'][q_graph], q_graph, sel_type)
                            st.plotly_chart(fig, use_container_width=True, key=f"hist_chart_{q_graph}")

        with tab_monthly:
            with st.container(border=True):
                st.header("Monthly Forecast Summary"); df = data['monthly'][[q for q in queues_to_show if q in data['monthly'].columns]]
                if not df.empty:
                    st.dataframe(df.style.format("{:.2%}"), use_container_width=True)
                    st.download_button("Download Monthly Forecast", to_excel_bytes(df), f"monthly_{sel_type}_forecast.xlsx")
                    with st.expander("View Monthly Aggregated Graph", expanded=st.session_state.get('shrink_show_graphs', False)):
                        hist_monthly = res['processed_data'][res['processed_data']['Queue'].isin(queues_to_show)].resample('M').mean(numeric_only=True)
                        fig = shrink_create_aggregated_plot(hist_monthly[[col]], df.drop('Subtotal', errors='ignore'), 'Monthly', sel_type)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No monthly forecast data to display for the selected queues.")
                
        with tab_weekly:
            with st.container(border=True):
                st.header("Weekly Forecast Summary"); df = data['weekly'][[q for q in queues_to_show if q in data['weekly'].columns]]
                if not df.empty:
                    st.dataframe(df.style.format("{:.2%}"), use_container_width=True)
                    st.download_button("Download Weekly Forecast", to_excel_bytes(df), f"weekly_{sel_type}_forecast.xlsx")
                    with st.expander("View Weekly Aggregated Graph", expanded=st.session_state.get('shrink_show_graphs', False)):
                        hist_weekly = res['processed_data'][res['processed_data']['Queue'].isin(queues_to_show)].resample('W-MON').mean(numeric_only=True)
                        fig = shrink_create_aggregated_plot(hist_weekly[[col]], df.drop('Subtotal', errors='ignore'), 'Weekly', sel_type)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No weekly forecast data to display for the selected queues.")

        with tab_daily:
            with st.container(border=True):
                st.header("Daily Forecast")
                best_methods_df = data['best'][data['best']['Queue'].isin(queues_to_show)].copy()
                if not best_methods_df.empty:
                    st.subheader("Best Method Analysis")
                    best_methods_df['Comments'] = best_methods_df['MAE'].apply(lambda x: get_significance_rating(x * 100, metric_type='mae'))
                    best_methods_df_display = best_methods_df.copy()
                    best_methods_df_display['MAE'] = best_methods_df_display['MAE'].map('{:.2%}'.format)
                    st.dataframe(best_methods_df_display[['Queue', 'Method', 'MAE', 'Comments']], use_container_width=True, hide_index=True)
                    st.download_button("Download Best Methods", to_excel_bytes(best_methods_df), "shrinkage_best_methods.xlsx")
                else:
                    st.info("No best method analysis to display for the selected queues.")

                df_interval = data['interval'] if 'interval' in data and not data['interval'].empty else pd.DataFrame()
                if not df_interval.empty and 'Queue' in df_interval.columns:
                    df_interval = df_interval[df_interval['Queue'].isin(queues_to_show)]

                if not df_interval.empty:
                    st.subheader("Interval-Level Forecast")
                    display_df_interval = df_interval.sort_values(by="Timestamp").tail(20)
                    st.caption("Showing the latest 20 records. Use the download button for the full forecast.")
                    
                    format_dict = {
                        'Planned_Shrinkage': '{:.2%}', 'Unplanned_Shrinkage': '{:.2%}',
                        'Total_Shrinkage': '{:.2%}', 'Hist_Daily_Avg': '{:.2%}',
                        f'Forecasted_{col}': '{:.2%}',
                    }
                    st.dataframe(display_df_interval.style.format(format_dict, na_rep='-'), use_container_width=True, hide_index=True)
                    st.download_button("Download Full Interval Forecast Data", to_excel_bytes(df_interval), f"interval_{sel_type}_forecast.xlsx", key=f"download_interval_{sel_type}")
                else: st.info("No interval-level data to display.")
                with st.expander("View Daily Forecast Graphs", expanded=st.session_state.get('shrink_show_graphs', False)):
                    for q_graph in queues_to_show:
                        if q_graph in data['hist'] and q_graph in data['daily']:
                            fig = shrink_create_forecast_plot(data['hist'][q_graph], data['daily'][q_graph], q_graph, sel_type)
                            st.plotly_chart(fig, use_container_width=True, key=f"daily_chart_{q_graph}")

        with tab_comp:
            with st.container(border=True):
                st.header("Shrinkage Comparison (Actual vs. Backtest Forecast)")
                date_range = st.date_input("Select Date Range for Comparison", [res['historical_min_date'], res['historical_max_date']], min_value=res['historical_min_date'], max_value=res['historical_max_date'], key="comparison_date_range")
                default_selection = [queues_to_show[0]] if len(queues_to_show) > 0 else []
                q_comp = st.multiselect("Select Queue(s) for Backtest Comparison:", queues_to_show, default=default_selection)
                
                if q_comp:
                    fig = go.Figure()
                    for queue in q_comp:
                        if queue in data['backtest']:
                            forecasted, actual = data['backtest'][queue]
                            if date_range and len(date_range) == 2:
                                actual = actual[(actual.index.date >= date_range[0]) & (actual.index.date <= date_range[1])]
                                forecasted = forecasted[(forecasted.index.date >= date_range[0]) & (forecasted.index.date <= date_range[1])]
                            fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name=f'Actual - {queue}'))
                            fig.add_trace(go.Scatter(x=forecasted.index, y=forecasted, mode='lines', name=f'Forecast - {queue}', line=dict(dash='dash')))
                    fig.update_layout(title=f"Backtest Comparison", yaxis=dict(tickformat=".1%"))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Download Historical Interval Data")
                if date_range and len(date_range) == 2:
                    filtered_processed_data = res['processed_data'][(res['processed_data'].index.date >= date_range[0]) & (res['processed_data'].index.date <= date_range[1]) & (res['processed_data']['Queue'].isin(queues_to_show))]
                    st.download_button("Download Filtered Interval Data", to_excel_bytes(filtered_processed_data), f"historical_interval_{date_range[0]}_to_{date_range[1]}.xlsx", key="download_comp_interval")

        with tab_manual:
            with st.container(border=True):
                st.header("Manual Shrinkage Forecasting")
                with st.form("manual_shrinkage_form"):
                    st.write("#### Configure Manual Forecast")
                    horizon_manual = st.number_input("Forecast Horizon (days)", 1, 365, 30, key="manual_shrink_horizon")
                    models_to_run = st.multiselect("Select models to run:", ["Seasonal Naive", "Moving Average", "Prophet"], default=["Seasonal Naive"])
                    submitted_manual = st.form_submit_button("🚀 Run Manual Shrinkage Forecast")

                if submitted_manual:
                    if not models_to_run:
                        st.error("Please select at least one model to run.")
                    else:
                        manual_forecasts = {}
                        queues_manual = res['processed_data']['Queue'].unique()
                        
                        with st.spinner("Running manual forecast..."):
                            for q in queues_manual:
                                ts = res['processed_data'][res['processed_data']["Queue"] == q][col].resample('D').mean().fillna(0)
                                if ts.empty: continue

                                for model_name in models_to_run:
                                    try:
                                        if model_name == "Seasonal Naive":
                                            forecast = shrink_forecast_seasonal_naive(ts, horizon_manual)
                                        elif model_name == "Moving Average":
                                            forecast = shrink_forecast_moving_average(ts, horizon_manual)
                                        elif model_name == "Prophet":
                                            forecast = shrink_forecast_prophet(ts, horizon_manual)
                                        
                                        if not forecast.empty:
                                            manual_forecasts[(q, model_name)] = forecast
                                    except Exception as e:
                                        st.warning(f"Model '{model_name}' failed for queue '{q}': {e}")
                        st.session_state.manual_shrinkage_results = pd.DataFrame(manual_forecasts)

                manual_results = st.session_state.get('manual_shrinkage_results')
                if manual_results is not None and not manual_results.empty:
                    st.subheader("Manual Forecast Results")
                    df_manual = st.session_state.manual_shrinkage_results
                    st.dataframe(df_manual.style.format("{:.2%}"), use_container_width=True)
                    st.download_button("Download Manual Forecast", to_excel_bytes(df_manual), "manual_shrinkage_forecast.xlsx")

def render_volume_forecast_tab():
    st.header("📦 Volume Forecast Engine")
    
    title_col, clear_col = st.columns([0.8, 0.2])
    with title_col:
        st.subheader("1. Upload Volume Data")
    with clear_col:
        st.write("")
        if st.button("Clear 🗑️", key="clear_volume_data", use_container_width=True):
            keys_to_clear = ['vol_uploader', 'm_horizon', 'd_horizon', 'df_volume_ready', 'df_volume_original', 'volume_monthly_results', 'volume_daily_results', 'manual_volume_results', 'backtest_volume_results']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            log_job_run("Volume", "Cleared", "N/A", 0, "User cleared data for module.")
            st.rerun()

    with st.container(border=True):
        col_uploader, col_template = st.columns([3,1])
        with col_uploader:
            uploaded_file = st.file_uploader("Upload Volume Raw Excel Data", type=["xlsx", "xls"], key="vol_uploader", label_visibility="collapsed")
        with col_template:
            st.write("") 
            st.write("")
            template_df = pd.DataFrame({"Date": ["2025-01-01"], "Interval": ["08:30:00"], "Volume": [15], "Queue": ["Support_L1"]})
            st.download_button(label="⬇️ Download Data Template", data=to_excel_bytes(template_df, index=False), file_name="volume_template.xlsx", use_container_width=True)

    if uploaded_file:
        try:
            df_volume = pd.read_excel(uploaded_file)
            df_prep = vol_prepare_full_data(df_volume)
            st.session_state.df_volume_ready = df_prep
            st.session_state.df_volume_original = df_volume
            with st.expander("📄 View Raw Data Preview"):
                st.dataframe(df_volume.head())
        except ValueError as e:
            st.error(f"❌ Data Error: {e}.")
            return
        except Exception as e:
            st.error(f"An unexpected error occurred while reading the file: {e}")
            return
    
    if 'df_volume_ready' in st.session_state:
        df_prep = st.session_state.df_volume_ready
        
        monthly_tab, daily_tab, manual_tab, backtest_tab = st.tabs(["📅 Monthly Forecast", "☀️ Daily Forecast", "🛠️ Manual Forecast", "🧪 Backtesting"])

        with monthly_tab:
            with st.container(border=True):
                st.subheader("2. Monthly Forecast Configuration")
                form_cols = st.columns([1,3])
                with form_cols[0]:
                    horizon_m = st.number_input("Forecast horizon (months)", 1, 24, 3, key="m_horizon")
                with form_cols[1]:
                    st.write("")
                    st.write("")
                    if st.button("🚀 Run Monthly Volume Forecast", use_container_width=True):
                        st.session_state.start_time = time.time()
                        all_fc, err, best_fc, best_methods, _ = vol_run_monthly_forecasting(st.session_state.df_volume_original, horizon_m)
                        
                        if best_fc.empty:
                            st.info("ℹ️ No forecast could be generated. This often happens if the uploaded data has insufficient history for every queue (e.g., less than 3 months).")
                            log_job_run("Monthly Volume", "Failed", "ERR#1", time.time() - st.session_state.start_time, "Insufficient data for forecasting.")
                        else:
                            st.session_state.volume_monthly_results = { "all_forecasts_df": all_fc, "error_df": err, "best_forecast_df": best_fc, "best_methods_df": best_methods, "original_df": df_prep }
                            log_job_run("Monthly Volume", "Success", "N/A", time.time() - st.session_state.start_time, "Monthly forecast completed.")
                            st.success("Monthly forecast competition complete!")
                            time.sleep(1)
                            st.rerun()

            if 'volume_monthly_results' in st.session_state and st.session_state.volume_monthly_results:
                res = st.session_state.volume_monthly_results
                
                st.subheader("3. View Results")
                for queue in res['best_forecast_df'].columns:
                    with st.container(border=True):
                        st.markdown(f"#### Results for Queue: **{queue}**")
                        kpi_cols = st.columns(4)
                        
                        winning_method, wmape, significance, mae = "N/A", "N/A", "N/A", "N/A"
                        if not res['best_methods_df'].empty and queue in res['best_methods_df'].index:
                            method_row = res['best_methods_df'].loc[queue]
                            winning_method = method_row['Method']
                            wmape_val = method_row['wMAPE']
                            wmape = f"{wmape_val:.2f}%"
                            significance = get_significance_rating(wmape_val, 'wmape')
                            mae = f"{method_row['MAE']:.2f}"

                        kpi_cols[0].metric("Winning Model", winning_method)
                        kpi_cols[1].metric("wMAPE", wmape)
                        kpi_cols[2].metric("MAE", mae)
                        kpi_cols[3].metric("Accuracy", significance)

                        hist = res['original_df'][res['original_df']['Queue']==queue]['Volume'].resample('MS').sum()
                        fc_ts = pd.Series(res['best_forecast_df'][queue].values, index=pd.to_datetime(res['best_forecast_df'].index, format='%b-%y'))
                        fig = vol_create_forecast_plot(hist, fc_ts, queue, "Month")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("View Detailed Tables & Downloads"):
                            st.markdown("**Best Forecast Data**")
                            st.dataframe(res['best_forecast_df'][[queue]])
                            
                            if not res['error_df'].empty:
                                st.markdown("**Model Competition Errors**")
                                st.dataframe(res['error_df'][res['error_df']['Queue']==queue])
                            
                            st.markdown("**Downloads**")
                            dl_cols = st.columns(4)
                            interval_fc_monthly = vol_generate_monthly_interval_forecast(res['best_forecast_df'][[queue]], res['original_df'])
                            dl_cols[0].download_button("Forecast (Monthly)", to_excel_bytes(res['best_forecast_df'][[queue]]), f"monthly_fc_{queue}.xlsx", key=f"dl_m_fc_{queue}")
                            dl_cols[1].download_button("Forecast (Interval)", to_excel_bytes(interval_fc_monthly), f"monthly_interval_{queue}.xlsx", key=f"dl_m_int_{queue}")
                            if not res['best_methods_df'].empty:
                                dl_cols[2].download_button("Winning Method", to_excel_bytes(res['best_methods_df'].loc[[queue]]), f"monthly_winner_{queue}.xlsx", key=f"dl_m_win_{queue}")
                            if not res['error_df'].empty:
                                dl_cols[3].download_button("All Errors", to_excel_bytes(res['error_df'][res['error_df']['Queue']==queue]), f"monthly_errors_{queue}.xlsx", key=f"dl_m_err_{queue}")

        with daily_tab:
            with st.container(border=True):
                st.subheader("2. Daily Forecast Configuration")
                form_cols_d = st.columns([1, 2, 2])
                with form_cols_d[0]:
                    horizon_d = st.number_input("Forecast horizon (days)", 1, 90, 14, key="d_horizon")
                with form_cols_d[1]:
                    country = st.selectbox("Country for Holidays", options=COUNTRY_NAMES, index=COUNTRY_NAMES.index("United States"))
                with form_cols_d[2]:
                    st.write(""); st.write("")
                    if st.button("🚀 Run Daily Volume Forecast", use_container_width=True):
                        st.session_state.start_time = time.time()
                        all_fc, err, best_fc, best_methods, _ = vol_run_daily_forecasting(st.session_state.df_volume_original, horizon_d, COUNTRY_CODES[country])
                        
                        if best_fc.empty:
                            st.info("ℹ️ No forecast could be generated. This often happens if the uploaded data has insufficient history for every queue (e.g., less than 14 days).")
                            log_job_run("Daily Volume", "Failed", "ERR#1", time.time() - st.session_state.start_time, "Insufficient data for forecasting.")
                        else:
                            st.session_state.volume_daily_results = { "all_forecasts_df": all_fc, "error_df": err, "best_forecast_df": best_fc, "best_methods_df": best_methods, "original_df": df_prep }
                            log_job_run("Daily Volume", "Success", "N/A", time.time() - st.session_state.start_time, "Daily forecast completed.")
                            st.success("Daily forecast complete!")
                            time.sleep(1)
                            st.rerun()

            if 'volume_daily_results' in st.session_state and st.session_state.volume_daily_results:
                res = st.session_state.volume_daily_results
                st.subheader("3. View Results")

                for queue in res['best_forecast_df'].columns:
                    with st.container(border=True):
                        st.markdown(f"#### Results for Queue: **{queue}**")
                        kpi_cols = st.columns(4)
                        
                        winning_method, wmape, significance, mae = "N/A", "N/A", "N/A", "N/A"
                        if not res['best_methods_df'].empty and queue in res['best_methods_df'].index:
                            method_row = res['best_methods_df'].loc[queue]
                            winning_method = method_row['Method']
                            wmape_val = method_row['wMAPE']
                            wmape = f"{wmape_val:.2f}%"
                            significance = get_significance_rating(wmape_val, 'wmape')
                            mae = f"{method_row['MAE']:.2f}"

                        kpi_cols[0].metric("Winning Model", winning_method)
                        kpi_cols[1].metric("wMAPE", wmape)
                        kpi_cols[2].metric("MAE", mae)
                        kpi_cols[3].metric("Accuracy", significance)

                        hist = res['original_df'][res['original_df']['Queue']==queue]['Volume'].resample('D').sum()
                        fig = vol_create_forecast_plot(hist, res['best_forecast_df'][queue], queue, "Day")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("View Detailed Tables & Downloads"):
                            st.markdown("**Best Forecast Data (Daily)**")
                            st.dataframe(res['best_forecast_df'][[queue]])
                            
                            if not res['error_df'].empty:
                                st.markdown("**Model Competition Errors**")
                                st.dataframe(res['error_df'][res['error_df']['Queue']==queue])
                            
                            st.markdown("**Downloads**")
                            dl_cols = st.columns(4)
                            daily_fc = res['best_forecast_df'][[queue]]; 
                            weekly_fc = daily_fc.resample('W-MON').sum(); 
                            interval_fc = vol_generate_interval_forecast(daily_fc, res['original_df'])
                            dl_cols[0].download_button("Forecast (Daily)", to_excel_bytes(daily_fc), f"daily_fc_{queue}.xlsx", key=f"dl_d_fc_{queue}")
                            dl_cols[1].download_button("Forecast (Weekly)", to_excel_bytes(weekly_fc), f"weekly_fc_{queue}.xlsx", key=f"dl_d_wk_{queue}")
                            dl_cols[2].download_button("Forecast (Interval)", to_excel_bytes(interval_fc[interval_fc['Queue']==queue]), f"interval_fc_{queue}.xlsx", key=f"dl_d_int_{queue}")
                            if not res['best_methods_df'].empty:
                                dl_cols[3].download_button("Winning Method", to_excel_bytes(res['best_methods_df'].loc[[queue]]), f"daily_winner_{queue}.xlsx", key=f"dl_d_win_{queue}")
        
        with manual_tab:
            st.header("🛠️ Manual Volume Forecast")
            if 'df_volume_ready' in st.session_state:
                chosen_df = st.session_state.df_volume_ready
                all_models = [
                    "Naive", "Seasonal Naive (7d)", "Moving Average (7d)", "Holt-Winters (Seasonal=7)",
                    "Prophet", "Linear Regression", "Random Forest",
                    "Seasonal Naive (12m)", "Moving Average (3m)", "Holt-Winters (Seasonal=12)", "SARIMA",
                ]
                with st.form("manual_vol_form"):
                    st.write("#### Configure Manual Forecast")
                    horizon_manual = st.number_input("Forecast Horizon (days)", 1, 365, 30)
                    models_to_run = st.multiselect("Select models to run:", all_models, default=all_models[:3])
                    country_manual = st.selectbox("Country for Holidays (for Prophet)", options=COUNTRY_NAMES, index=COUNTRY_NAMES.index("United States"))
                    submitted_manual = st.form_submit_button("🚀 Run Manual Forecast")
                if submitted_manual:
                    if not models_to_run: st.error("Please select at least one model to run.")
                    else:
                        manual_forecasts = {}; holidays_manual = pd.DataFrame(CountryHoliday(COUNTRY_CODES[country_manual], years=range(datetime.now().year-2, datetime.now().year+2)).items()) if country_manual != "NONE" else None
                        if holidays_manual is not None: holidays_manual.columns = ['ds', 'holiday']
                        queues_manual = chosen_df['Queue'].unique()
                        
                        with st.spinner("🏃‍♂️ Running manual forecast..."):
                            for i, q in enumerate(queues_manual):
                                ts_daily = chosen_df[chosen_df["Queue"] == q]["Volume"].resample('D').sum(); ts_monthly = chosen_df[chosen_df["Queue"] == q]["Volume"].resample('MS').sum()
                                
                                model_functions = {
                                    "Naive": lambda ts, h: vol_forecast_naive(ts, h, freq='D'),
                                    "Seasonal Naive (7d)": lambda ts, h: vol_forecast_seasonal_naive(ts, h, freq='D', seasonal_periods=7),
                                    "Moving Average (7d)": lambda ts, h: vol_forecast_moving_average(ts, h, window=7, freq='D'),
                                    "Holt-Winters (Seasonal=7)": lambda ts, h: vol_forecast_holtwinters(ts, h, freq='D', seasonal_periods=7),
                                    "Prophet": lambda ts, h: vol_forecast_prophet(ts, h, freq='D', holidays=holidays_manual),
                                    "Linear Regression": lambda ts, h: vol_forecast_ml(ts, h, LinearRegression(), freq='D'),
                                    "Random Forest": lambda ts, h: vol_forecast_ml(ts, h, RandomForestRegressor(n_estimators=100, random_state=42), freq='D'),
                                    "Seasonal Naive (12m)": lambda ts, h: vol_forecast_seasonal_naive(ts, h, freq='MS', seasonal_periods=12),
                                    "Moving Average (3m)": lambda ts, h: vol_forecast_moving_average(ts, h, window=3, freq='MS'),
                                    "Holt-Winters (Seasonal=12)": lambda ts, h: vol_forecast_holtwinters(ts, h, freq='MS', seasonal_periods=12),
                                    "SARIMA": lambda ts, h: vol_forecast_sarima(ts, h, (1,1,1), (1,1,1,12), freq='MS'),
                                }
                                
                                for model_name in models_to_run:
                                    try:
                                        is_daily_model = any(sub in model_name for sub in ['(7d)', 'Prophet', 'Regression', 'Forest', 'Naive'])
                                        if is_daily_model:
                                            forecast = model_functions[model_name](ts_daily, horizon_manual)
                                        else:
                                            horizon_months = int(np.ceil(horizon_manual / 30.44))
                                            forecast = model_functions[model_name](ts_monthly, horizon_months)
                                        if not forecast.empty: manual_forecasts[(q, model_name)] = forecast
                                    except Exception as e: st.warning(f"Model '{model_name}' failed for queue '{q}': {e}")
                        st.session_state.manual_volume_results = { "forecasts": pd.DataFrame(manual_forecasts), "historical": chosen_df }
                if 'manual_volume_results' in st.session_state and st.session_state.manual_volume_results:
                    st.subheader("Manual Forecast Results")
                    manual_res = st.session_state.manual_volume_results; manual_fc_df = manual_res['forecasts']
                    res_daily, res_weekly, res_monthly, res_interval = st.tabs(["Daily", "Weekly", "Monthly", "Interval"])
                    with res_daily: st.dataframe(manual_fc_df); st.download_button("Download Daily Data", to_excel_bytes(manual_fc_df), "manual_forecast_daily.xlsx")
                    with res_weekly: weekly_manual = manual_fc_df.resample('W-MON').sum(); st.dataframe(weekly_manual); st.download_button("Download Weekly Data", to_excel_bytes(weekly_manual), "manual_forecast_weekly.xlsx")
                    with res_monthly: monthly_manual = manual_fc_df.resample('M').sum(); st.dataframe(monthly_manual); st.download_button("Download Monthly Data", to_excel_bytes(monthly_manual), "manual_forecast_monthly.xlsx")
                    with res_interval:
                        st.info("Interval-level forecast is generated by disaggregating daily-frequency model forecasts.")
                        daily_model_cols = [col for col in manual_fc_df.columns if any(sub in col[1] for sub in ['(7d)', 'Prophet', 'Regression', 'Forest', 'Naive'])]
                        if daily_model_cols: daily_fc_subset = manual_fc_df[daily_model_cols]; interval_manual = vol_generate_interval_forecast(daily_fc_subset, manual_res['historical']); st.dataframe(interval_manual); st.download_button("Download Interval Data", to_excel_bytes(interval_manual), "manual_forecast_interval.xlsx")
                        else: st.warning("To generate an interval forecast, please include a daily model in your selection.")
            else:
                st.warning("Please upload a file to enable manual forecasting.")
        
        with backtest_tab:
            st.header("🧪 Volume Backtesting")
            if 'df_volume_ready' in st.session_state:
                df_prep_bt = st.session_state.df_volume_ready
                queues_bt = df_prep_bt['Queue'].unique()
                queue_bt_choice = st.selectbox("Select Queue to Backtest:", queues_bt)
                time_unit_bt = st.radio("Select Backtesting Frequency:", ["Daily", "Monthly"], horizontal=True)
                
                if time_unit_bt == "Daily":
                    models_bt = {
                        "Seasonal Naive (7d)": lambda ts, h: vol_forecast_seasonal_naive(ts, h, freq='D', seasonal_periods=7),
                        "Moving Average (7d)": lambda ts, h: vol_forecast_moving_average(ts, h, window=7, freq='D'),
                        "Holt-Winters (Seasonal=7)": lambda ts, h: vol_forecast_holtwinters(ts, h, freq='D', seasonal_periods=7),
                        "Prophet": lambda ts, h: vol_forecast_prophet(ts, h, freq='D'),
                        "Random Forest": lambda ts, h: vol_forecast_ml(ts, h, RandomForestRegressor(), freq='D')
                    }
                    horizon_label = "Backtesting Horizon (days)"; default_horizon = 7
                else: 
                    models_bt = {
                        "Seasonal Naive (12m)": lambda ts, h: vol_forecast_seasonal_naive(ts, h, freq='MS', seasonal_periods=12),
                        "Moving Average (3m)": lambda ts, h: vol_forecast_moving_average(ts, h, window=3, freq='MS'),
                        "SARIMA": lambda ts, h: vol_forecast_sarima(ts, h, (1,1,1), (1,1,1,12), freq='MS'),
                        "Random Forest": lambda ts, h: vol_forecast_ml(ts, h, RandomForestRegressor(), freq='MS')
                    }
                    horizon_label = "Backtesting Horizon (months)"; default_horizon = 2
                models_bt_to_run = st.multiselect("Select models to backtest:", list(models_bt.keys()), default=list(models_bt.keys())[0])
                horizon_bt = st.number_input(horizon_label, 1, 90, default_horizon)
                if st.button("🚀 Run Backtest"):
                    with st.spinner("⏳ Running backtest..."):
                        ts_bt_raw = df_prep_bt[df_prep_bt["Queue"] == queue_bt_choice]["Volume"]
                        ts_bt = ts_bt_raw.resample('MS').sum() if time_unit_bt == "Monthly" else ts_bt_raw.resample('D').sum()
                        all_backtest_results = {'Actual': ts_bt}
                        for model_name in models_bt_to_run:
                            model_func = models_bt[model_name]
                            bt_res = vol_backtest_forecast(ts_bt, model_func, horizon_bt)
                            if not bt_res.empty: all_backtest_results[model_name] = bt_res['Forecast']
                        results_df = pd.DataFrame(all_backtest_results).dropna()
                        if len(results_df.columns) <= 1: 
                            st.error("Backtesting failed.")
                        else: 
                            st.session_state.backtest_volume_results = { "results_df": results_df, "historical_df": df_prep_bt[df_prep_bt["Queue"] == queue_bt_choice] }
                            st.rerun()
                if 'backtest_volume_results' in st.session_state and st.session_state.backtest_volume_results:
                    res_bt = st.session_state.backtest_volume_results; results_df = res_bt['results_df']
                    st.subheader("Performance Metrics")
                    metrics_list = []
                    for col in results_df.columns:
                        if col != 'Actual': metrics = vol_calculate_error_metrics(results_df['Actual'], results_df[col]); metrics['Model'] = col; metrics_list.append(metrics)
                    st.dataframe(pd.DataFrame(metrics_list).set_index("Model"))
                    st.subheader("Backtest Comparison Chart")
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'], mode='lines', name='Actual', line=dict(color='black', width=3)))
                    for col in results_df.columns:
                        if col != 'Actual': fig_bt.add_trace(go.Scatter(x=results_df.index, y=results_df[col], mode='lines', name=col, line=dict(dash='dash')))
                    fig_bt.update_layout(title=f"Backtest Comparison for {queue_bt_choice}"); st.plotly_chart(fig_bt, use_container_width=True)
                    st.subheader("⬇️ Download Backtest Results")
                    main_bt_df = results_df; interval_bt_df = vol_generate_interval_forecast(results_df.drop(columns=['Actual']), res_bt['historical_df'])
                    dl_bt_cols = st.columns(2)
                    dl_bt_cols[0].download_button(f"Download {time_unit_bt} Results", to_excel_bytes(main_bt_df), f"backtest_{time_unit_bt.lower()}.xlsx")
                    dl_bt_cols[1].download_button("Download Interval Results", to_excel_bytes(interval_bt_df), "backtest_interval.xlsx")
            else:
                st.warning("Please upload a file to enable backtesting.")
            
def render_capacity_planning_tab():
    st.header("⚙️ Capacity Planning Modeler")

    title_col, clear_col = st.columns([0.8, 0.2])
    with title_col:
        st.subheader("1. Input Parameters")
    with clear_col:
        st.write("")
        if st.button("Clear 🗑️", key="clear_capacity_data", use_container_width=True):
            if 'capacity_model_results' in st.session_state: del st.session_state['capacity_model_results']
            log_job_run("Capacity", "Cleared", "N/A", 0, "User cleared data for module.")
            st.rerun()

    FTE_OPT_NAME = "FTE (OR Tool)"
    MONTE_CARLO_NAME = "Monte Carlo (HC)"

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            login_hours = st.number_input("Login Hours per FTE", 0.1, 24.0, 8.0, 0.1)
            volume = st.number_input("Primary Task Volume", 0, 100000, 5000)
            aht_seconds = st.number_input("Primary Task AHT (s)", 1, 3600, 300)
        with c2:
            occupancy = st.number_input("Target Occupancy (%)", 1.0, 100.0, 85.0, 0.5, "%.1f", help="Affects the Workload model only.")
            sla_target_percent = st.slider("SLA Target (%)", 1, 100, 80)
            sla_value_seconds = st.number_input("Target Service Time (s)", 1, 600, 30)
            st.caption("These targets affect the Erlang C and Monte Carlo models.")
        with c3:
            concurrency = st.number_input("Concurrency", 1.0, 10.0, 1.0, 0.1, "%.1f", help="Number of simultaneous tasks an agent can handle.")
            
        with st.expander(f"🧠 Show Advanced Inputs (for {FTE_OPT_NAME})"):
            c4, c5 = st.columns(2)
            with c4:
                email_volume = st.number_input("Secondary Task Volume", 0, 100000, 2000)
                email_aht_sec = st.number_input("Secondary Task AHT (s)", 1, 3600, 900)
            with c5:
                cost_primary = st.number_input("Primary Skill Cost ($)", 0, 10000, 800)
                cost_secondary = st.number_input("Secondary Skill Cost ($)", 0, 10000, 750)
                cost_blended = st.number_input("Blended Skill Cost ($)", 0, 10000, 950)
        st.markdown("---")
        
        models_to_run = st.multiselect(
            "Select Models to Run",
            ["Workload", "Erlang C", MONTE_CARLO_NAME, FTE_OPT_NAME],
            default=["Workload", "Erlang C", MONTE_CARLO_NAME, FTE_OPT_NAME]
        )
        
        if st.button("🚀 Run Capacity Models", use_container_width=True):
            results = {}
            progress_bar = st.progress(0, text="Starting capacity modeling...")
            total_models = len(models_to_run)
            
            for i, model_name in enumerate(models_to_run):
                progress_text = f"Running {model_name} model..."
                progress_bar.progress((i + 1) / total_models, text=progress_text)
                time.sleep(0.1)
                
                if model_name == "Workload":
                    results['Workload'] = run_workload_model(login_hours, aht_seconds, volume, occupancy, concurrency)
                elif model_name == "Erlang C":
                    results['Erlang C'] = run_erlang_c_model(login_hours, aht_seconds, volume, sla_target_percent, sla_value_seconds, concurrency)
                elif model_name == MONTE_CARLO_NAME:
                    results[MONTE_CARLO_NAME] = run_monte_carlo_hc_model(login_hours, aht_seconds, volume, sla_target_percent, sla_value_seconds, concurrency)
                elif model_name == FTE_OPT_NAME:
                    model_inputs = {
                        'call_volume': volume, 'call_aht_sec': aht_seconds, 'cost_voice': cost_primary,
                        'email_volume': email_volume, 'email_aht_sec': email_aht_sec, 'cost_email': cost_secondary,
                        'cost_blended': cost_blended, 'hours_per_fte': login_hours, 'concurrency': concurrency
                    }
                    results[FTE_OPT_NAME] = run_fte_optimization_model(model_inputs)
            
            st.session_state.capacity_model_results = results
            progress_bar.empty()

    if st.session_state.get('capacity_model_results'):
        st.subheader("2. Model Results")
        results = st.session_state.capacity_model_results
        
        for model_name, model_output in results.items():
            model_output["Target ASA (s)"] = sla_value_seconds
            if "Avg Wait (s)" not in model_output: model_output["Avg Wait (s)"] = "N/A"
            if "Predicted SL (%)" not in model_output and "Predicted SLA (%)" not in model_output: model_output["Predicted SLA (%)"] = "N/A"
        
        res_cols = st.columns(len(results))
        for i, (model_name, model_output) in enumerate(results.items()):
            with res_cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{model_name}**")
                    metric_order = ["Required FTE", "Required HC", "Total FTE", "Predicted SLA (%)", "Predicted SL (%)", "Target ASA (s)", "Avg Wait (s)", "Minimum Cost ($)"]
                    for key in metric_order:
                        if key in model_output:
                            st.metric(label=key, value=model_output[key])
        
        st.markdown("---")
        
        st.subheader("💡 Model Recommendation Analysis")
        summary_data = []
        model_comments = {
            "Workload": "⚠️ Lowest cost, highest service risk. Use as a baseline only.",
            "Erlang C": "⭐⭐ Industry standard. A great balance of cost and service.",
            MONTE_CARLO_NAME: "⭐⭐⭐ Most robust/conservative. Recommended for high-certainty planning.",
            FTE_OPT_NAME: "ℹ️ Best for cost optimization across multiple skills."
        }
        
        if "Workload" in results:
            summary_data.append({
                "Model": "Workload", "Required HC/FTE": results["Workload"].get("Required FTE", "N/A"),
                "Key Metric": f"{occupancy}% Occupancy", "Comments": model_comments["Workload"]
            })
        if "Erlang C" in results:
            summary_data.append({
                "Model": "Erlang C", "Required HC/FTE": results["Erlang C"].get("Required HC", "N/A"),
                "Key Metric": f"{results['Erlang C'].get('Predicted SL (%)', 0)}% SL", "Comments": model_comments["Erlang C"]
            })
        if MONTE_CARLO_NAME in results:
            summary_data.append({
                "Model": MONTE_CARLO_NAME, "Required HC/FTE": results[MONTE_CARLO_NAME].get("Required HC", "N/A"),
                "Key Metric": f"{results[MONTE_CARLO_NAME].get('Predicted SLA (%)', 0)}% SLA", "Comments": model_comments[MONTE_CARLO_NAME]
            })
        if FTE_OPT_NAME in results and "Error" not in results[FTE_OPT_NAME]:
            summary_data.append({
                "Model": FTE_OPT_NAME, "Required HC/FTE": results[FTE_OPT_NAME].get("Total FTE", "N/A"),
                "Key Metric": results[FTE_OPT_NAME].get("Minimum Cost ($)", "N/A"), "Comments": model_comments[FTE_OPT_NAME]
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            sl_models = []
            if "Erlang C" in results: sl_models.append({"name": "Erlang C", "hc": results["Erlang C"].get("Required HC", float('inf'))})
            if MONTE_CARLO_NAME in results: sl_models.append({"name": MONTE_CARLO_NAME, "hc": results[MONTE_CARLO_NAME].get("Required HC", float('inf'))})

            recommended_model = ""
            if sl_models:
                best_model = min(sl_models, key=lambda x: x['hc'])
                recommended_model = best_model['name']
                
            def highlight_recommendation(row):
                if row.Model == recommended_model:
                    return ['background-color: #BDE5F8'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                summary_df.style.apply(highlight_recommendation, axis=1), hide_index=True, use_container_width=True
            )
            if recommended_model:
                st.caption(f"✅ The highlighted row is the most efficient model that meets the SLA target.")

        results_df = pd.DataFrame.from_dict(results).transpose()
        st.download_button(
            label="⬇️ Download All Model Results", data=to_excel_bytes(results_df, index=True),
            file_name="capacity_model_results.xlsx", mime="application/vnd.ms-excel"
        )
def render_monthly_capacity_planner_tab():
    st.header("🗓️ Monthly Capacity Planner")
    st.markdown("---")

    # --- 1. Check for required forecast data ---
    if 'volume_monthly_results' not in st.session_state or st.session_state.volume_monthly_results is None:
        st.warning("Please run a Monthly Volume Forecast from the Volume tab first to enable this feature.")
        return

    # --- 2. Load and Prepare Data ---
    monthly_volume_forecast = st.session_state.volume_monthly_results['best_forecast_df']
    all_queues = sorted(list(monthly_volume_forecast.columns))
    
    if 'shrinkage_results' in st.session_state and st.session_state.shrinkage_results is not None:
        shrinkage_forecast = st.session_state.shrinkage_results['forecasts']['Total']['monthly']
    else:
        shrinkage_forecast = pd.DataFrame(index=monthly_volume_forecast.index, columns=all_queues).fillna(0)

    # --- 3. Load Saved Plan ---
    saved_plans = get_saved_plan_names()
    selected_plan_to_load = st.selectbox("Load Saved Plan (Optional)", [""] + saved_plans, key="load_plan_selector")
    
    # Logic to handle loading a plan
    if selected_plan_to_load and st.session_state.get('loaded_plan_name') != selected_plan_to_load:
        plan_details = load_capacity_plan(selected_plan_to_load)
        if plan_details:
            st.session_state.loaded_plan_name = selected_plan_to_load
            st.session_state.capacity_plan_inputs = plan_details['plan_data']
            st.session_state.loaded_queues = plan_details['queues']
            st.session_state.loaded_start_month = plan_details['start_month']
            st.success(f"Successfully loaded plan: '{selected_plan_to_load}'")
            st.rerun()

    # --- 4. User Filters for Month and Queue ---
    filter_cols = st.columns([1, 2])
    available_months = monthly_volume_forecast.index.tolist()
    
    start_month_default_index = available_months.index(st.session_state.get('loaded_start_month', available_months[0]))
    queues_default = st.session_state.get('loaded_queues', [all_queues[0]] if all_queues else [])

    start_month = filter_cols[0].selectbox("Select Start Month", options=available_months, index=start_month_default_index)
    selected_queues = filter_cols[1].multiselect("Select Queues to Plan", options=all_queues, default=queues_default)

    if not selected_queues:
        st.info("Please select at least one queue to view the capacity plan.")
        return

    # --- 5. Initialize and Build the Main Planner DataFrame ---
    plan_dates = pd.date_range(start=pd.to_datetime(start_month, format='%b-%y'), periods=12, freq='MS')
    plan_months_str = [d.strftime('%b-%y') for d in plan_dates]

    plan_structure = [
        "Opening Headcount", "New Hires", "Attrition %", "Attrition HC", "Ending Headcount", "HC in Training", "HC in Nesting", "Productive Headcount",
        "HC - Tenure <1 Month", "AHT - Tenure <1 Month (s)", "HC - Tenure 1-3 Months", "AHT - Tenure 1-3 Months (s)", "HC - Tenure >3 Months", "AHT - Tenure >3 Months (s)", "Blended AHT (s)",
        "Forecasted Volume", "Working Days", "Monthly Login Hours", "Target Occupancy (%)", "Target SLA (%)", "Target ASA (s)",
        "Forecasted Shrinkage (%)", "Shift Inflex (%)", "Total Shrinkage (%)",
        "Workload Required HC", "Erlang C Required HC", "Monte Carlo Required HC",
        "Over/Under (Workload)", "Over/Under (Erlang C)", "Over/Under (Monte Carlo)"
    ]
    plan_df = pd.DataFrame(0, index=plan_structure, columns=plan_months_str)

    # --- 6. Populate Data and Default Inputs ---
    vol_plan = monthly_volume_forecast.reindex(plan_months_str).ffill()
    shrink_plan = shrinkage_forecast.reindex(plan_months_str).ffill()
    
    plan_df.loc["Forecasted Volume"] = vol_plan[selected_queues].sum(axis=1)
    plan_df.loc["Forecasted Shrinkage (%)"] = shrink_plan[selected_queues].mean(axis=1)

    # Load data from session state if it exists, otherwise set defaults
    input_key = "-".join(sorted(selected_queues))
    saved_inputs = st.session_state.get('capacity_plan_inputs', {}).get(input_key, {})

    editable_rows = {
        "Opening Headcount": 100, "New Hires": 10, "Attrition %": 0.05, "HC in Training": 5, "HC in Nesting": 5,
        "HC - Tenure <1 Month": 10, "AHT - Tenure <1 Month (s)": 600, "HC - Tenure 1-3 Months": 20, "AHT - Tenure 1-3 Months (s)": 450,
        "HC - Tenure >3 Months": 70, "AHT - Tenure >3 Months (s)": 300, "Working Days": 21, "Monthly Login Hours": 160,
        "Target Occupancy (%)": 85, "Target SLA (%)": 80, "Target ASA (s)": 20, "Shift Inflex (%)": 0.05
    }

    for row, default_val in editable_rows.items():
        for month in plan_months_str:
            plan_df.loc[row, month] = saved_inputs.get(month, {}).get(row, default_val)

    # --- 7. Create Interactive Data Editor ---
    st.info("💡 You can edit the white cells directly in the table below. Calculations will update automatically.")
    
    edited_df = st.data_editor(plan_df, use_container_width=True)

    # --- 8. Perform All Calculations Based on Edited Data ---
    # (This section re-calculates everything based on the user's edits in the table)
    for i, month in enumerate(plan_months_str):
        # Headcount Flow
        if i > 0:
            edited_df.loc["Opening Headcount", month] = edited_df.loc["Ending Headcount", plan_months_str[i-1]]
        edited_df.loc["Attrition HC", month] = -np.floor(edited_df.loc["Opening Headcount", month] * edited_df.loc["Attrition %", month])
        edited_df.loc["Ending Headcount", month] = edited_df.loc["Opening Headcount", month] + edited_df.loc["New Hires", month] + edited_df.loc["Attrition HC", month]
        edited_df.loc["Productive Headcount", month] = edited_df.loc["Ending Headcount", month] - edited_df.loc["HC in Training", month] - edited_df.loc["HC in Nesting", month]

        # AHT Bell Curve
        total_hc = edited_df.loc["HC - Tenure <1 Month", month] + edited_df.loc["HC - Tenure 1-3 Months", month] + edited_df.loc["HC - Tenure >3 Months", month]
        if total_hc > 0:
            weighted_aht = ((edited_df.loc["HC - Tenure <1 Month", month] * edited_df.loc["AHT - Tenure <1 Month (s)", month]) + (edited_df.loc["HC - Tenure 1-3 Months", month] * edited_df.loc["AHT - Tenure 1-3 Months (s)", month]) + (edited_df.loc["HC - Tenure >3 Months", month] * edited_df.loc["AHT - Tenure >3 Months (s)", month])) / total_hc
            edited_df.loc["Blended AHT (s)", month] = weighted_aht
        else:
            edited_df.loc["Blended AHT (s)", month] = 0

        # Total Shrinkage
        edited_df.loc["Total Shrinkage (%)", month] = edited_df.loc["Forecasted Shrinkage (%)", month] + edited_df.loc["Shift Inflex (%)", month]

        # Required HC Calculations
        inputs = edited_df[month]
        work_days = inputs["Working Days"] if inputs["Working Days"] > 0 else 1
        daily_hours = inputs["Monthly Login Hours"] / work_days
        daily_vol = inputs["Forecasted Volume"] / work_days
        net_shrink_factor = 1 - inputs["Total Shrinkage (%)"]

        workload_req = ((daily_vol * inputs["Blended AHT (s)"]) / 3600) / (daily_hours * (inputs["Target Occupancy (%)"] / 100))
        edited_df.loc["Workload Required HC", month] = math.ceil(workload_req / net_shrink_factor) if net_shrink_factor < 1 else math.ceil(workload_req)
        
        erlang_res = run_erlang_c_model(daily_hours, inputs["Blended AHT (s)"], daily_vol, inputs["Target SLA (%)"], inputs["Target ASA (s)"], 1)
        edited_df.loc["Erlang C Required HC", month] = math.ceil(erlang_res.get("Required HC", 0) / net_shrink_factor) if net_shrink_factor < 1 else math.ceil(erlang_res.get("Required HC", 0))

        mc_res = run_monte_carlo_hc_model(daily_hours, inputs["Blended AHT (s)"], daily_vol, inputs["Target SLA (%)"], inputs["Target ASA (s)"], 1)
        edited_df.loc["Monte Carlo Required HC", month] = math.ceil(mc_res.get("Required HC", 0) / net_shrink_factor) if net_shrink_factor < 1 else math.ceil(mc_res.get("Required HC", 0))

        # Over/Under
        edited_df.loc["Over/Under (Workload)", month] = edited_df.loc["Productive Headcount", month] - edited_df.loc["Workload Required HC", month]
        edited_df.loc["Over/Under (Erlang C)", month] = edited_df.loc["Productive Headcount", month] - edited_df.loc["Erlang C Required HC", month]
        edited_df.loc["Over/Under (Monte Carlo)", month] = edited_df.loc["Productive Headcount", month] - edited_df.loc["Monte Carlo Required HC", month]

    # --- 9. Display Final Table and Save Option ---
    st.dataframe(edited_df.style.format("{:,.2f}", na_rep='-'), use_container_width=True)

    st.download_button(
        label=f"⬇️ Download Plan for {', '.join(selected_queues)}",
        data=to_excel_bytes(edited_df),
        file_name=f"monthly_capacity_plan_{'_'.join(selected_queues)}.xlsx"
    )

    with st.sidebar:
        st.markdown("---")
        st.subheader("Save Current Plan")
        plan_name_to_save = st.text_input("Enter Plan Name to Save")
        if st.button("Save Plan"):
            if plan_name_to_save:
                # Extract only the editable data for saving
                data_to_save = edited_df.loc[editable_rows.keys()].to_dict()
                result = save_capacity_plan(plan_name_to_save, selected_queues, start_month, data_to_save)
                if result is True:
                    st.sidebar.success(f"Plan '{plan_name_to_save}' saved!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error(result)
            else:
                st.sidebar.warning("Please enter a name for the plan.")
    
def render_admin_tab():
    st.header("👤 User & Permissions Management")
    
    if st.session_state.get('role', '').lower() != 'admin':
        st.error("You do not have permission to access this page.")
        return

    users_df = get_all_users()
    users_df['can_view_shrinkage'] = users_df['can_view_shrinkage'].apply(lambda x: 'yes' if x == 1 else 'no')
    users_df['can_view_volume'] = users_df['can_view_volume'].apply(lambda x: 'yes' if x == 1 else 'no')
    users_df['can_view_capacity'] = users_df['can_view_capacity'].apply(lambda x: 'yes' if x == 1 else 'no')
    users_df['can_manage_schedules'] = users_df['can_manage_schedules'].apply(lambda x: 'yes' if x == 1 else 'no')
    
    st.subheader("Existing Users")
    st.dataframe(users_df, use_container_width=True, hide_index=True)

    st.subheader("Add/Edit User")
    with st.form("user_management_form"):
        selected_user = st.selectbox("Select User to Edit (or select None for new user)", options=['None'] + list(users_df['username'].unique()))
        
        is_new_user = selected_user == 'None'
        current_user_data = users_df[users_df['username'] == selected_user].iloc[0] if not is_new_user else {}

        username = st.text_input("Username", value=current_user_data.get('username', ''), disabled=not is_new_user)
        password = st.text_input("Password (leave blank to keep current)", type="password")
        role = st.selectbox("Role", options=['Admin', 'Manager', 'Agent'], index=['Admin', 'Manager', 'Agent'].index(current_user_data.get('role', 'Agent')))
        
        st.markdown("---")
        st.write("**Module Permissions**")
        p_cols = st.columns(4)
        shrink_perm = p_cols[0].checkbox("Shrinkage", value=current_user_data.get('can_view_shrinkage', 'no') == 'yes')
        volume_perm = p_cols[1].checkbox("Volume", value=current_user_data.get('can_view_volume', 'no') == 'yes')
        capacity_perm = p_cols[2].checkbox("Capacity", value=current_user_data.get('can_view_capacity', 'no') == 'yes')
        schedule_perm = p_cols[3].checkbox("Schedules", value=current_user_data.get('can_manage_schedules', 'no') == 'yes')

        submitted = st.form_submit_button("Save User")
        
        if submitted:
            if is_new_user:
                if username and password:
                    if add_user(username, password, role, {'can_view_shrinkage': shrink_perm, 'can_view_volume': volume_perm, 'can_view_capacity': capacity_perm, 'can_manage_schedules': schedule_perm}):
                        st.success(f"User '{username}' added successfully!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("Username already exists.")
                else:
                    st.error("Username and password are required for a new user.")
            else:
                update_user(selected_user, password if password else None, role, {'can_view_shrinkage': shrink_perm, 'can_view_volume': volume_perm, 'can_view_capacity': capacity_perm, 'can_manage_schedules': schedule_perm})
                st.success(f"User '{selected_user}' updated successfully!")
                st.cache_data.clear()
                st.rerun()
    
    if not users_df.empty:
        st.subheader("Delete User")
        with st.form("delete_user_form"):
            user_to_delete = st.selectbox("Select a user to delete:", options=users_df['username'].unique())
            if st.form_submit_button("Delete User", type="primary"):
                if user_to_delete == st.session_state['username']:
                    st.error("You cannot delete your own account.")
                else:
                    delete_user(user_to_delete)
                    st.warning(f"User '{user_to_delete}' deleted.")
                    st.cache_data.clear()
                    st.rerun()

def render_logs_on_admin_tab():
    st.header("📜 System & Job History")
    
    history_df = get_run_history()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No jobs have been run yet.")




# --- SECTION 8: MAIN APP EXECUTION ---


if __name__ == "__main__":
    create_db_tables()
    initialize_session_state()
    
    # Check if any users exist. If not, prompt to create an admin user first.
    if not get_all_users().empty:
        if not check_password():
            st.stop()
    else:
        st.warning("No users found. Please create an initial Admin user.")
        with st.form("initial_admin_setup"):
            st.subheader("Create Initial Admin User")
            username = st.text_input("Admin Username")
            password = st.text_input("Admin Password", type="password")
            if st.form_submit_button("Create User"):
                if username and password:
                    add_user(username, password, 'Admin', {'can_view_shrinkage': 1, 'can_view_volume': 1, 'can_view_capacity': 1, 'can_manage_schedules': 1})
                    st.success("Admin user created! Please refresh and log in.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Username and password cannot be empty.")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("🛠️ Controls")
        st.info(f"Logged in as: **{st.session_state.get('username', '')}**")
        if st.button("Logout"):
            # Clear all session state keys to ensure a clean logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")

    # --- Main Page Header ---
    logo_col, title_col, refresh_col = st.columns([0.6, 0.2, 0.2])
    with logo_col:
        st.image("https://i.postimg.cc/vmwmF50z/Remove-background-project.png", width=350)
    with refresh_col:
        st.write("#")
        if st.button("Refresh 🔄", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    current_day = datetime.now().strftime("%A")
    st.markdown(f"**Happy {current_day}!** Hey **{st.session_state.get('username', 'User')}**, ready to forecast the future?")
    st.markdown("Welcome to the all-in-one Workforce Management tool. Select a module below to get started.")
    st.markdown("---")   
    
    # --- Dynamic Tab Creation based on Permissions ---
    allowed_tabs = []
    tab_render_map = {
        "👥 Shrinkage": render_shrinkage_forecast_tab,
        "📦 Volume": render_volume_forecast_tab,
        "🗓️ Monthly Planner": render_monthly_capacity_planner_tab,
        "⚙️ Capacity Modeler": render_capacity_planning_tab,
        "👤 Admin": render_admin_tab
    }

    if str(st.session_state.get("can_view_shrinkage")).lower() == 'yes':
        allowed_tabs.append("👥 Shrinkage")
    if str(st.session_state.get("can_view_volume")).lower() == 'yes':
        allowed_tabs.append("📦 Volume")
    if str(st.session_state.get("can_view_capacity")).lower() == 'yes':
        allowed_tabs.append("🗓️ Monthly Planner")
        allowed_tabs.append("⚙️ Capacity Modeler")
    if st.session_state.get('role', '').lower() == 'admin':
        allowed_tabs.append("👤 Admin")
        
    if not allowed_tabs:
        st.warning("Your user role has no modules enabled. Please contact an administrator.")
    else:
        created_tabs = st.tabs(allowed_tabs)
        for i, tab_title in enumerate(allowed_tabs):
            with created_tabs[i]:
                if tab_title == "👤 Admin":
                    render_admin_tab()
                    st.markdown("---")
                    render_logs_on_admin_tab()
                else:
                    # This will call the correct render function for the tab
                    tab_render_map[tab_title]()

    st.markdown("---")
    st.markdown("<footer style='text-align:center;color:#94a3b8;'>Powered by PayOps WFM | One App to Rule Them All</footer>", unsafe_allow_html=True)
