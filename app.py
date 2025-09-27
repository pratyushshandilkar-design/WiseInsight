# ==============================================================================
# UNIFIED FORECAST ENGINE - WISEINSIGHTS (FINAL POLISHED VERSION)
# ==============================================================================
# This application combines two separate forecasting tools into a single,
# professional web application with a secure login page.
#
# MODIFIED: Fixed AttributeError, changed background to white,
# added spinners for long processes, and restyled upload sections.
# ==============================================================================

# --- SECTION 1: UNIVERSAL LIBRARY IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import itertools
import warnings
import logging
import os
import zipfile
import shutil
import time

# Machine Learning and Time Series Libraries
from prophet import Prophet
import plotly.graph_objects as go
from holidays import CountryHoliday
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# --- SECTION 2: GLOBAL CONFIGURATION & UTILITIES ---

st.set_page_config(
    page_title="WiseInsights",
    page_icon="https://d21buns5ku92am.cloudfront.net/69645/images/470451-Frame%2039321-0745ed-medium-1677657684.png",
    layout="wide"
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

def to_excel_bytes(data, index=True):
    """Converts a DataFrame or a dictionary of DataFrames into Excel format as bytes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if isinstance(data, dict):
            for sheet_name, df in data.items():
                df.to_excel(writer, index=index, sheet_name=sheet_name)
        else:
            df_to_save = data
            if isinstance(df_to_save.columns, pd.MultiIndex):
                df_to_save = df_to_save.copy()
                df_to_save.columns = ['_'.join(map(str, col)).strip() for col in df_to_save.columns.values]
            df_to_save.to_excel(writer, index=index, sheet_name="Sheet1")
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

# --- SECTION 3: LOGIN, SESSION STATE, AND MAIN UI STRUCTURE ---

@st.cache_data(ttl=600)
def load_user_credentials(file_path="users.xlsx"):
    """Loads user credentials from the specified Excel file."""
    try:
        df = pd.read_excel(file_path)
        if 'username' not in df.columns or 'password' not in df.columns:
            st.error("Invalid users.xlsx file. It must contain 'username' and 'password' columns.")
            return None
        df['password'] = df['password'].astype(str)
        return df
    except FileNotFoundError:
        st.error("`users.xlsx` not found. Please create it in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"Error loading user credentials: {e}")
        return None

def check_password():
    """Renders a login form and returns `True` if credentials are correct."""
    credentials_df = load_user_credentials()

    if st.session_state.get("password_correct", False):
        return True

    if credentials_df is None:
        return False

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,400;0,600;1,400;1,600&display=swap');
            body, .stApp {
                font-family: 'Poppins', sans-serif;
            }
            [data-testid="stAppViewContainer"] > .main { 
                background-color: #F0F2F5; 
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.image("https://cdn.worldvectorlogo.com/logos/wise-2.svg", width=250)
        st.subheader("Wise Predictions, Smarter Decisions")
    with col2:
        with st.form("login_form"):
            st.markdown("##### Log in to WiseInsights")
            username = st.text_input("Username", placeholder="Username")
            password = st.text_input("Password", type="password", placeholder="Password")
            submitted = st.form_submit_button("Log In")

            if submitted:
                user_record = credentials_df[credentials_df['username'] == username]
                if not user_record.empty and password == user_record.iloc[0]['password']:
                    st.session_state["password_correct"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.session_state["password_correct"] = False
                    st.error("😕 Incorrect username or password.")
    
    return False


def initialize_session_state():
    """Initializes all required session state variables."""
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


################################################################################
#                                                                              #
#                   SECTION 4: SHRINKAGE FORECAST ENGINE                       #
#                                                                              #
################################################################################

def log_job_run(job_type, status, error_code, time_took, archive_file="N/A"):
    """Appends a record to the unified run history."""
    new_run = {
        "Job Type": job_type, "Status": status,
        "Error Code": error_code, "Time Took (s)": f"{time_took:.2f}",
        "Job run by": st.session_state.get("username", "N/A"), "Archive File": archive_file
    }
    st.session_state.run_history.insert(0, new_run)

def shrink_forecast_moving_average(ts, steps, window=7, freq='D'):
    if len(ts) < window: window = max(1, len(ts))
    val = ts.rolling(window=window, min_periods=1).mean().iloc[-1]
    future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.Series([val] * steps, index=future_idx).clip(0, 1)

def shrink_forecast_naive(ts, steps, freq='D'):
    last_val = ts.iloc[-1] if not ts.empty else 0
    future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.Series([last_val] * steps, index=future_idx).clip(0, 1)

def shrink_forecast_seasonal_naive(ts, steps, freq='D', seasonal_periods=7):
    if len(ts) < seasonal_periods: return shrink_forecast_naive(ts, steps, freq)
    future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    seasonal_pattern = [ts.iloc[-seasonal_periods:].iloc[i % seasonal_periods] for i in range(steps)]
    return pd.Series(seasonal_pattern, index=future_idx).clip(0, 1)

def shrink_forecast_prophet(ts, steps, holidays_df=None, prophet_params=None):
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_ts.index, y=historical_ts.values, mode='lines', name=f'Historical {shrinkage_type}', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines', name=f'Forecasted {shrinkage_type}', line=dict(color='crimson', dash='dash')))
    fig.update_layout(title=f'📈 {shrinkage_type} Shrinkage Forecast: {queue_name}', xaxis_title='Date', yaxis_title='Shrinkage %', yaxis=dict(tickformat=".1%"))
    return fig

def shrink_create_aggregated_plot(historical_df, forecast_df, aggregation, shrinkage_type="Total"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df.mean(axis=1).index, y=historical_df.mean(axis=1).values, mode='lines', name='Historical Avg', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=forecast_df.mean(axis=1).index, y=forecast_df.mean(axis=1).values, mode='lines', name='Forecasted Avg', line=dict(color='crimson', dash='dash')))
    fig.update_layout(title=f'📊 {aggregation} Shrinkage Forecast vs. Historical (Aggregated)', xaxis_title='Date', yaxis_title='Shrinkage %', yaxis=dict(tickformat=".1%"))
    return fig

def shrink_backtest_forecast(ts, horizon, method):
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
    if _daily_forecast_df.empty or _historical_df.empty: return pd.DataFrame()
    hist = _historical_df.copy(); hist['Hour'] = hist.index.hour; hist['DayOfWeek'] = hist.index.day_name()
    profiles = pd.merge(hist.groupby(['Queue', 'DayOfWeek', 'Hour'])[shrinkage_col].mean().reset_index(), hist.groupby(['Queue', 'DayOfWeek'])[shrinkage_col].mean().reset_index().rename(columns={shrinkage_col: 'Hist_Daily_Avg'}), on=['Queue', 'DayOfWeek'])
    shrinkage_type = shrinkage_col.split('_')[0]
    all_interval_forecasts = []
    for queue in _daily_forecast_df.columns:
        for date, daily_forecast_val in _daily_forecast_df[queue].items():
            day_profile = profiles[(profiles['Queue'] == queue) & (profiles['DayOfWeek'] == date.strftime('%A'))].copy()
            if day_profile.empty or day_profile['Hist_Daily_Avg'].iloc[0] == 0: continue
            adjustment_factor = daily_forecast_val / day_profile['Hist_Daily_Avg'].iloc[0]
            day_profile[f'Forecasted_{shrinkage_type}'] = (day_profile[shrinkage_col] * adjustment_factor).clip(0, 0.7)
            day_profile['Timestamp'] = day_profile['Hour'].apply(lambda h: date.replace(hour=int(h)))
            all_interval_forecasts.append(day_profile)
    if not all_interval_forecasts: return pd.DataFrame()
    return pd.concat(all_interval_forecasts)

@st.cache_data
def shrink_generate_aggregated_forecasts(_daily_forecast_df):
    if _daily_forecast_df.empty: return pd.DataFrame(), pd.DataFrame()
    weekly_df = _daily_forecast_df.resample('W-MON', label='left', closed='left').mean()
    if not weekly_df.empty: weekly_df.loc['Subtotal'] = weekly_df.mean()
    monthly_df = _daily_forecast_df.resample('M').mean()
    if not monthly_df.empty: monthly_df.loc['Subtotal'] = monthly_df.mean()
    return weekly_df, monthly_df

def render_shrinkage_forecast_tab():
    st.header("Shrinkage Forecast")
    
    # FIX: Restyled upload section
    with st.container(border=True):
        st.subheader("1. Upload Shrinkage Data")
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
        with st.expander("📄 View Raw Data Preview"):
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
                    # FIX: Added spinner for better user feedback
                    with st.spinner("⏳ Running shrinkage forecast..."):
                        st.session_state.start_time = time.time()
                        error_code = "N/A"; status = "Success"
                        try:
                            processed_data = shrink_process_raw_data(raw_data)
                            if processed_data is not None:
                                forecasts = {}
                                shrinkage_definitions = {'Total': 'Total_Shrinkage', 'Planned': 'Planned_Shrinkage', 'Unplanned': 'Unplanned_Shrinkage'}
                                for i, (typ, col) in enumerate(shrinkage_definitions.items()):
                                    daily_df, best_df, hist_map = shrink_run_forecasting(processed_data, int(horizon), col)
                                    interval_df = shrink_generate_interval_forecast(daily_df, processed_data, col)
                                    weekly_df, monthly_df = shrink_generate_aggregated_forecasts(daily_df)
                                    backtest_dict = {q: shrink_backtest_forecast(hist_map[q], horizon, best_df.loc[best_df['Queue']==q, 'Method'].iloc[0] if q in best_df['Queue'].values else 'Prophet') for q in hist_map if len(hist_map[q]) > horizon}
                                    forecasts[typ] = {"daily": daily_df, "best": best_df, "hist": hist_map, "interval": interval_df, "weekly": weekly_df, "monthly": monthly_df, "backtest": backtest_dict}
                                st.session_state['shrinkage_results'] = {"forecasts": forecasts, "queues": processed_data["Queue"].unique(), "processed_data": processed_data, "types": ['Total', 'Planned', 'Unplanned'], "cols": shrinkage_definitions, "historical_min_date": processed_data.index.min().date(), "historical_max_date": processed_data.index.max().date()}
                            else: status = "Error"; error_code = "ERR#2"
                        except ValueError: status = "Error"; error_code = "ERR#2"; st.error("Data processing failed. Please check your Excel file.")
                        except Exception: status = "Error"; error_code = "ERR#4"; st.error("An unexpected error occurred.")
                        
                        log_job_run("Shrinkage Forecast", status, error_code, time.time() - st.session_state.start_time)
                    if status == "Success": 
                        st.success("Shrinkage forecast completed!")
                        time.sleep(1) # Brief pause to let user see success message
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
                with st.expander("View Historical vs. Forecast Graph", expanded=st.session_state.shrink_show_graphs):
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
                    with st.expander("View Monthly Aggregated Graph", expanded=st.session_state.shrink_show_graphs):
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
                    with st.expander("View Weekly Aggregated Graph", expanded=st.session_state.shrink_show_graphs):
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

                df_interval = data['interval'][data['interval']['Queue'].isin(queues_to_show)]
                if not df_interval.empty:
                    st.subheader("Interval-Level Forecast")
                    display_df_interval = df_interval.sort_values(by="Timestamp").tail(20)
                    st.caption("Showing the latest 20 records. Use the download button for the full forecast.")
                    
                    # FIX: Format relevant columns as percentages and hide index
                    format_dict = {
                        'Planned_Shrinkage': '{:.2%}', 'Unplanned_Shrinkage': '{:.2%}',
                        'Total_Shrinkage': '{:.2%}', 'Hist_Daily_Avg': '{:.2%}',
                        'Forecasted_Total': '{:.2%}', 'Forecasted_Planned': '{:.2%}',
                        'Forecasted_Unplanned': '{:.2%}'
                    }
                    st.dataframe(display_df_interval.style.format(format_dict, na_rep='-'), use_container_width=True, hide_index=True)
                    st.download_button("Download Full Interval Forecast Data", to_excel_bytes(df_interval), f"interval_{sel_type}_forecast.xlsx", key=f"download_interval_{sel_type}")
                else: st.info("No interval-level data to display.")
                with st.expander("View Daily Forecast Graphs", expanded=st.session_state.shrink_show_graphs):
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
                # FIX: Overhauled Manual Shrinkage Tab
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

                # FIX: Added robust check for None before checking .empty
                manual_results = st.session_state.get('manual_shrinkage_results')
                if manual_results is not None and not manual_results.empty:
                    st.subheader("Manual Forecast Results")
                    df_manual = st.session_state.manual_shrinkage_results
                    st.dataframe(df_manual.style.format("{:.2%}"), use_container_width=True)
                    st.download_button("Download Manual Forecast", to_excel_bytes(df_manual), "manual_shrinkage_forecast.xlsx")


################################################################################
#                                                                              #
#                   SECTION 5: VOLUME FORECAST ENGINE                          #
#                                                                              #
################################################################################

# ------------------------------------------------------------------------------
# 5.1. Volume: Archiving and Global Variables
# ------------------------------------------------------------------------------

VOL_ARCHIVE_DIR = "volume_forecast_archive"
os.makedirs(VOL_ARCHIVE_DIR, exist_ok=True)
COUNTRY_CODES = {"United States": "US", "United Kingdom": "GB", "Canada": "CA", "Australia": "AU", "Germany": "DE", "France": "FR", "Spain": "ES", "Italy": "IT", "India": "IN", "Brazil": "BR", "Mexico": "MX", "Japan": "JP", "None": "NONE"}
COUNTRY_NAMES = sorted(list(COUNTRY_CODES.keys()))

def vol_archive_run_results(run_ts, results, fcast_type):
    """Saves all result DataFrames from a forecast run into a timestamped folder."""
    run_dir = os.path.join(VOL_ARCHIVE_DIR, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    prefix = "daily" if fcast_type == "Daily" else "monthly"
    for key, df in results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            with open(os.path.join(run_dir, f"{prefix}_{key}.xlsx"), "wb") as f:
                f.write(to_excel_bytes(df, index=True))

def vol_create_zip_for_run(run_ts):
    """Creates a ZIP archive of a given forecast run folder."""
    run_dir = os.path.join(VOL_ARCHIVE_DIR, run_ts)
    if not os.path.isdir(run_dir): return None
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for file in files: zf.write(os.path.join(root, file), arcname=file)
    return zip_buffer.getvalue()

def vol_get_archived_runs():
    """Returns a list of all archived run timestamps."""
    try:
        return sorted([d for d in os.listdir(VOL_ARCHIVE_DIR) if os.path.isdir(os.path.join(VOL_ARCHIVE_DIR, d))], reverse=True)
    except FileNotFoundError: return []

# ------------------------------------------------------------------------------
# 5.2. Volume: Data Preparation and Plotting
# ------------------------------------------------------------------------------

def robust_interval_to_timedelta(s):
    """Robustly converts a Series to timedelta."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_timedelta(s * 24 * 3600, unit="s")
    else:
        return pd.to_timedelta(s.astype(str), errors="coerce")

def vol_prepare_full_data(df):
    """Processes raw uploaded data for volume forecasting."""
    if not {"Date", "Interval", "Volume", "Queue"}.issubset(df.columns):
        raise ValueError("Volume file missing required columns: Date, Interval, Volume, Queue")
    
    df = df.copy()
    df["Interval_td"] = robust_interval_to_timedelta(df["Interval"])
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

# ------------------------------------------------------------------------------
# 5.3. Volume: Core Forecasting Models & Functions
# ------------------------------------------------------------------------------

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

def vol_forecast_weighted_moving_average(ts, steps, window=4, freq='MS'):
    """Generates a weighted moving average forecast."""
    try:
        if len(ts) < window: window = max(1, len(ts))
        weights = np.arange(1, window + 1)
        wma_series = ts.rolling(window, min_periods=1).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)
        val = wma_series.iloc[-1]
        future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return pd.Series([val] * steps, index=future_idx).round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_double_exp_smoothing(ts, steps, freq='MS'):
    """Generates a double exponential smoothing forecast."""
    try:
        if len(ts) < 3: return vol_forecast_moving_average(ts, steps, window=len(ts), freq=freq)
        model = ExponentialSmoothing(ts, trend="add", seasonal=None).fit()
        fc = model.forecast(steps)
        fc.index = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return fc.round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_holtwinters(ts, steps, freq='MS', seasonal_periods=12):
    """Generates a Holt-Winters (triple exponential smoothing) forecast."""
    try:
        if len(ts) < seasonal_periods * 2: return vol_forecast_double_exp_smoothing(ts, steps, freq)
        model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
        fc = model.forecast(steps)
        fc.index = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return fc.round()
    except Exception: return pd.Series(dtype=float)

def vol_forecast_decomposition(ts, steps, freq='MS', seasonal_periods=12):
    """Generates a forecast based on seasonal decomposition."""
    try:
        if len(ts) < seasonal_periods * 2: return vol_forecast_moving_average(ts, steps, window=min(3, len(ts)), freq=freq)
        result = seasonal_decompose(ts, model='additive', period=seasonal_periods)
        last_trend_val = result.trend.dropna().iloc[-1]
        trend_diff = result.trend.diff().mean()
        future_trend = np.arange(1, steps + 1) * trend_diff + last_trend_val
        last_seasonal_cycle = result.seasonal.tail(seasonal_periods)
        future_seasonal = np.tile(last_seasonal_cycle, steps // seasonal_periods + 1)[:steps]
        fc = future_trend + future_seasonal
        future_idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
        return pd.Series(fc, index=future_idx).round()
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

# ------------------------------------------------------------------------------
# 5.4. Volume: Main Forecasting Pipelines
# ------------------------------------------------------------------------------

@st.cache_data
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

@st.cache_data
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


@st.cache_data
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

@st.cache_data
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

# ------------------------------------------------------------------------------
# 5.5. Volume: Main UI Rendering Function
# ------------------------------------------------------------------------------

def render_volume_forecast_tab():
    """Renders the Volume Forecast UI, with separate sub-tabs for monthly and daily forecasts."""
    st.header("📦 Volume Forecast Engine")
    
    with st.container(border=True):
        st.subheader("1. Upload Volume Data")
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
                    if st.button("🚀 Run Monthly Volume Forecast", key="run_vol_month", use_container_width=True):
                        with st.spinner("🚀 Running monthly forecast competition... This may take a moment."):
                            st.session_state.start_time = time.time()
                            all_fc, err, best_fc, best_methods, _ = vol_run_monthly_forecasting(st.session_state.df_volume_original, horizon_m)
                            
                            if best_fc.empty:
                                st.info("ℹ️ No forecast could be generated. This often happens if the uploaded data has insufficient history for every queue (e.g., less than 3 months).")
                                log_job_run("Monthly Volume", "Failed", "ERR#1", time.time() - st.session_state.start_time)
                            else:
                                st.session_state.volume_monthly_results = { "all_forecasts_df": all_fc, "error_df": err, "best_forecast_df": best_fc, "best_methods_df": best_methods, "original_df": df_prep }
                                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_Monthly")
                                log_job_run("Monthly Volume", "Success", "N/A", time.time() - st.session_state.start_time, ts_str)
                                vol_archive_run_results(ts_str, st.session_state.volume_monthly_results, "Monthly")
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
                        with st.spinner("🚀 Running daily forecast competition... This may take a moment."):
                            st.session_state.start_time = time.time()
                            all_fc, err, best_fc, best_methods, _ = vol_run_daily_forecasting(st.session_state.df_volume_original, horizon_d, COUNTRY_CODES[country])
                            
                            if best_fc.empty:
                                st.info("ℹ️ No forecast could be generated. This often happens if the uploaded data has insufficient history for every queue (e.g., less than 14 days).")
                                log_job_run("Daily Volume", "Failed", "ERR#1", time.time() - st.session_state.start_time)
                            else:
                                st.session_state.volume_daily_results = { "all_forecasts_df": all_fc, "error_df": err, "best_forecast_df": best_fc, "best_methods_df": best_methods, "original_df": df_prep }
                                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_Daily")
                                log_job_run("Daily Volume", "Success", "N/A", time.time() - st.session_state.start_time, ts_str)
                                vol_archive_run_results(ts_str, st.session_state.volume_daily_results, "Daily")
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
                
                # FIX: Expanded backtesting model options
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



# --- SECTION 6: MAIN APPLICATION EXECUTION ---
if __name__ == "__main__":
    initialize_session_state()

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="st-"], [class*="css-"] {
            font-family: 'Poppins', sans-serif; 
            font-weight: 300;
        }
        /* FIX: Changed main background to white */
        .stApp { background-color: #FFFFFF; } 

        /* Main Tabs */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 10px; 
            border-bottom: 2px solid #E0E0E0;
        }
        .stTabs [data-baseweb="tab"] { 
            height: auto;
            padding: 10px 18px; 
            font-size: 15px;
            font-weight: 400;
            background-color: #F0F2F5; 
            border-radius: 8px 8px 0 0;
            border: 1px solid #E0E0E0;
            margin-bottom: -2px;
        }
        .stTabs [aria-selected="true"] { 
            background-color: #FFFFFF;
            color: #1a1a1a;
            font-weight: 600;
            border-bottom: 2px solid #FFFFFF;
        }
        
        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: #F8F9FA;
            border: 1px solid #EAEAEA;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        }

    </style>
    """, unsafe_allow_html=True)


    if not check_password():
        st.stop()

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("https://cdn.worldvectorlogo.com/logos/wise-2.svg", width=150)
        with col2:
            st.markdown("<h1 style='text-align: center; color: black; font-weight:600;'>WiseInsights</h1>", unsafe_allow_html=True)
            welcome_name = st.session_state.get("username", "User").capitalize()
            st.markdown(f"<p style='text-align: center;'>Welcome, {welcome_name}! Today is {datetime.now().strftime('%A, %B %d, %Y')}.</p>", unsafe_allow_html=True)

    st.markdown("---")

    with st.sidebar:
        st.header("🛠️ Controls & Archives")
        st.info(f"Logged in as: **{welcome_name}**")
        if st.button("Logout"):
            for key in st.session_state.keys(): del st.session_state[key]
            st.rerun()
        st.markdown("---")
        st.subheader("📜 Run History")
        
        if st.session_state.run_history:
            history_df = pd.DataFrame(st.session_state.run_history)
            st.dataframe(history_df, use_container_width=True, hide_index=True,
                column_config={ "Time Took (s)": st.column_config.NumberColumn("Time (s)", width="small") }
            )
            for i, run in enumerate(st.session_state.run_history):
                if run["Archive File"] != "N/A":
                    zip_data = vol_create_zip_for_run(run["Archive File"])
                    if zip_data:
                        st.download_button(f"Download Archive for Job {i+1}", zip_data, f"vol_archive_{run['Archive File']}.zip", key=f"dl_archive_{i}")
        else:
            st.info("No jobs have been run in this session.")

        if st.button("🗑️ Clear All History & Archives"):
            st.session_state.run_history = []
            if os.path.exists(VOL_ARCHIVE_DIR): shutil.rmtree(VOL_ARCHIVE_DIR)
            os.makedirs(VOL_ARCHIVE_DIR, exist_ok=True)
            for key in ['volume_monthly_results', 'volume_daily_results', 'shrinkage_results', 'manual_shrinkage_results']:
                if key in st.session_state: st.session_state[key] = None
            st.rerun()
            
        st.markdown("---")
        st.subheader("📖 Error Code Legend")
        error_data = { "Code": ["ERR#1", "ERR#2", "ERR#3", "ERR#4"], "Meaning": ["Low Data Volume", "Data Processing Error", "Forecast Model Failure", "Unknown Code Error"] }
        st.table(pd.DataFrame(error_data))
    
    shrinkage_tab, volume_tab = st.tabs(["👥 Shrinkage Forecast", "📦 Volume Forecast"])
    with shrinkage_tab:
        render_shrinkage_forecast_tab()
    with volume_tab:
        render_volume_forecast_tab()

    st.markdown("<hr><footer style='text-align:center;color:#94a3b8;'>Powered by PayOps WFM | One App to Rule Them All</footer>", unsafe_allow_html=True)