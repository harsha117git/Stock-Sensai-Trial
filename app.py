import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
import yfinance as yf

# Set API keys in environment variables
# Using only Anthropic (Claude) API integration
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")

# Import utility modules
from utils.data_fetcher import (
    fetch_stock_data, get_stock_name_from_symbol, get_available_indices,
    get_stocks_for_index, fetch_news, get_sector_for_stock, get_sector_weights,
    fetch_economic_indicators
)
from utils.ml_models import train_and_predict
from utils.sentiment_analyzer import analyze_news_sentiment, generate_sentiment_time_series
from utils.economic_factors import calculate_weighted_factors
from utils.visualizations import (
    plot_stock_price_history, plot_prediction_comparison, plot_sentiment_analysis,
    plot_sentiment_time_series, plot_economic_factors_influence, generate_simulation_plot,
    plot_vertical_price_history, plot_model_predictions_vertical, generate_prediction_report
)
from utils.visualization_utils import generate_docx_report
from utils.database import (
    save_stock_data, save_prediction, save_news_articles, save_economic_indicators,
    save_search, get_stock_price_data, get_saved_searches, get_latest_predictions,
    get_latest_news
)
from utils.theme_manager import (
    get_current_theme, apply_theme_to_plotly_figure, get_color_scale, 
    get_theme_list, initialize_theme
)

# Configure Streamlit page (only if not already configured)
if 'page_config_set' not in st.session_state:
    st.set_page_config(
        page_title="Stock Price Predictor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_config_set = True

# Import AI provider module
from utils.ai_provider import (
    initialize_ai_settings, get_current_provider, 
    set_ai_provider, validate_api_keys, research_with_ai,
    get_last_research_time
)

# Initialize theme and AI settings
initialize_theme()
initialize_ai_settings()

# Initialize session state with default values
def initialize_session_state():
    """Initialize all required session state variables with default values"""
    # Core application state
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = 'RELIANCE.NS'
    if 'time_period' not in st.session_state:
        st.session_state.time_period = '1y'
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'news_list' not in st.session_state:
        st.session_state.news_list = None
    if 'sentiment_summary' not in st.session_state:
        st.session_state.sentiment_summary = None
    if 'last_data_fetch_time' not in st.session_state:
        st.session_state.last_data_fetch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'sentiment_df' not in st.session_state:
        st.session_state.sentiment_df = None
    if 'economic_factors' not in st.session_state:
        st.session_state.economic_factors = None
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    # Initialize page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    # User preferences for data sources
    if 'use_ai_data' not in st.session_state:
        st.session_state.use_ai_data = True
    if 'data_exchange' not in st.session_state:
        st.session_state.data_exchange = 'yahoo'
    if 'data_source_changed' not in st.session_state:
        st.session_state.data_source_changed = False
    # Debug mode for troubleshooting data fetching issues
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'data_fetch_logs' not in st.session_state:
        st.session_state.data_fetch_logs = []
    # Initialize stock exchange
    if 'stock_exchange' not in st.session_state:
        st.session_state.stock_exchange = "NSE"

# Call the initialization function
initialize_session_state()

def toggle_theme():
    """Toggle between available themes."""
    # Use the theme_manager's toggle function which handles the Streamlit config update
    from utils.theme_manager import toggle_theme as tm_toggle_theme
    tm_toggle_theme()
    
    # Force Streamlit to rerun to apply the theme change
    st.rerun()

def render_dashboard():
    """Render the main dashboard page."""
    st.title("Stock Price Prediction Dashboard")
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("Controls")
        
        # Market header - fixed to India
        st.subheader("Market: India")
        
        # Set NSE as the default stock exchange (removed selection option)
        if 'stock_exchange' not in st.session_state:
            st.session_state.stock_exchange = "NSE"
            
        # Use NSE as the fixed stock exchange
        selected_stock_exchange = "NSE"
            
        # Index selection
        st.subheader("Select Index")
        available_indices = get_available_indices()
        selected_index = st.selectbox("Index", available_indices)
        
        # Stock selection
        st.subheader("Select Stock")
        # Use the selected stock exchange to get appropriate stock symbols
        stocks = get_stocks_for_index(selected_index, exchange=selected_stock_exchange)
        stock_options = {get_stock_name_from_symbol(symbol): symbol for symbol in stocks}
        selected_stock_name = st.selectbox("Stock", list(stock_options.keys()))
        selected_stock = stock_options[selected_stock_name]
        
        # Time period selection
        st.subheader("Select Time Period")
        time_periods = {
            "1 Week": "1w",
            "1 Month": "1m",
            "6 Months": "6m",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        selected_period_name = st.selectbox("Period", list(time_periods.keys()), index=3)
        selected_period = time_periods[selected_period_name]
        
        # Action buttons
        st.subheader("Actions")
        predict_button = st.button("Predict Next Day Price", type="primary")
        update_button = st.button("Update Data")
        ai_news_button = st.button("Get AI-Powered Research", help="Generate fresh stock news and economic research using advanced AI")
        
        # Data source selection
        st.subheader("Data Source")
        
        # Initialize data source preferences in session state if not present
        if 'use_ai_data' not in st.session_state:
            st.session_state.use_ai_data = True
        
        if 'data_exchange' not in st.session_state:
            st.session_state.data_exchange = 'yahoo'
        
        # Data source options
        data_source_options = {
            'yahoo': 'Yahoo Finance',
            'alpha_vantage': 'Alpha Vantage API'
        }
        
        selected_data_source = st.selectbox(
            "Select Data Source:",
            list(data_source_options.keys()),
            format_func=lambda x: data_source_options[x],
            index=0 if st.session_state.data_exchange not in data_source_options else 
                  list(data_source_options.keys()).index(st.session_state.data_exchange)
        )
        
        # Option for fallback data
        use_ai = st.checkbox(
            "Allow data filtering for better display",
            value=st.session_state.use_ai_data,
            help="When enabled, the app will apply data filtering to improve visualization quality."
        )
        
        # Update session state if data source changed
        if selected_data_source != st.session_state.data_exchange:
            st.session_state.data_exchange = selected_data_source
            st.session_state.data_source_changed = True
        else:
            if 'data_source_changed' not in st.session_state:
                st.session_state.data_source_changed = False
        
        # Update session state if Claude AI preference changed
        if use_ai != st.session_state.use_ai_data:
            st.session_state.use_ai_data = use_ai
            # If Claude AI usage changed and it could affect the current view, refresh
            if selected_data_source == 'yahoo' and selected_period in ['1w', '1m', '3m']:
                st.session_state.data_source_changed = True
        
        # Special debugging buttons for troubleshooting data fetch issues
        if st.button("🛠️ Debug Data Sources", key="debug_button", help="Run diagnostics on data sources"):
            st.warning("⚠️ Debug Mode - Running diagnostics on data sources")
            st.session_state.debug_mode = True
            
            # Initialize or clear existing logs
            if 'data_fetch_logs' not in st.session_state:
                st.session_state.data_fetch_logs = []
            else:
                st.session_state.data_fetch_logs = []
                
            # Add header to logs
            st.session_state.data_fetch_logs.append(f"DEBUG: Testing data sources for {selected_stock} ({datetime.now().strftime('%H:%M:%S')})")
            st.session_state.data_fetch_logs.append(f"DEBUG: Period: {selected_period}, Exchange: {selected_data_source}")
            
            # Force data refresh with logging
            with st.spinner("Running diagnostics..."):
                # First try NSE data
                st.session_state.data_fetch_logs.append(f"DEBUG: Trying NSE data source...")
                try:
                    from utils.indian_exchange_data import get_data_from_exchange
                    nse_data = get_data_from_exchange(selected_stock, exchange="nse", period=selected_period)
                    if nse_data is not None and not nse_data.empty:
                        st.session_state.data_fetch_logs.append(f"DEBUG: ✓ Successfully fetched NSE data ({len(nse_data)} rows)")
                    else:
                        st.session_state.data_fetch_logs.append(f"DEBUG: ✗ Failed to fetch NSE data - returned empty dataset")
                except Exception as e:
                    st.session_state.data_fetch_logs.append(f"DEBUG: ✗ Error with NSE data: {str(e)}")
                
                # Then try Yahoo Finance
                st.session_state.data_fetch_logs.append(f"DEBUG: Trying Yahoo Finance...")
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(selected_stock)
                    yf_data = ticker.history(period=selected_period)
                    if yf_data is not None and not yf_data.empty:
                        st.session_state.data_fetch_logs.append(f"DEBUG: ✓ Successfully fetched Yahoo Finance data ({len(yf_data)} rows)")
                    else:
                        st.session_state.data_fetch_logs.append(f"DEBUG: ✗ Failed to fetch Yahoo data - returned empty dataset")
                except Exception as e:
                    st.session_state.data_fetch_logs.append(f"DEBUG: ✗ Error with Yahoo Finance: {str(e)}")
                
                # Finally try Alpha Vantage if an API key is available
                from utils.api_keys import get_api_key
                alpha_key = get_api_key('ALPHA_VANTAGE_API_KEY')
                if alpha_key:
                    st.session_state.data_fetch_logs.append(f"DEBUG: Trying Alpha Vantage...")
                    try:
                        import requests
                        # Convert symbol format for Alpha Vantage
                        av_symbol = selected_stock.replace('.NS', '')
                        av_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_symbol}&apikey={alpha_key}"
                        response = requests.get(av_url)
                        if response.status_code == 200 and 'Time Series (Daily)' in response.json():
                            st.session_state.data_fetch_logs.append(f"DEBUG: ✓ Successfully connected to Alpha Vantage API")
                        else:
                            error_msg = response.json().get('Note', response.json().get('Error Message', 'Unknown error'))
                            st.session_state.data_fetch_logs.append(f"DEBUG: ✗ Alpha Vantage error: {error_msg}")
                    except Exception as e:
                        st.session_state.data_fetch_logs.append(f"DEBUG: ✗ Error with Alpha Vantage: {str(e)}")
                else:
                    st.session_state.data_fetch_logs.append(f"DEBUG: ➖ Alpha Vantage API key not configured")
            
            # Add completion marker to logs
            st.session_state.data_fetch_logs.append(f"DEBUG: Diagnostics completed at {datetime.now().strftime('%H:%M:%S')}")
            # Force refresh to show results
            st.rerun()
            
        # Display logs if available
        if st.session_state.get('debug_mode', False) and st.session_state.get('data_fetch_logs', []):
            with st.expander("Data Source Diagnostics", expanded=True):
                for log in st.session_state.data_fetch_logs:
                    st.text(log)
        
        # Theme selection
        st.subheader("Theme")
        theme_options = get_theme_list()
        theme_names = {
            "light": "Default Light",
            "dark": "Default Dark",
            "midnight": "Midnight OLED Dark",
            "solarizedDark": "Solarized Dark",
            "solarizedLight": "Solarized Light",
            "hacker": "Hacker Theme",
            "nord": "Nord Theme",
            "dracula": "Dracula",
            "zen": "Zen Mode"
        }
        theme_labels = [theme_names.get(t, t.capitalize()) for t in theme_options]
        selected_theme_index = theme_options.index(st.session_state.theme) if st.session_state.theme in theme_options else 0
        
        selected_theme_label = st.selectbox(
            "Select Theme:",
            theme_labels,
            index=selected_theme_index
        )
        
        # Map the label back to the theme key
        selected_theme = next(
            (k for k, v in theme_names.items() if v == selected_theme_label), 
            theme_options[theme_labels.index(selected_theme_label)]
        )
        
        # Update theme if changed
        if selected_theme != st.session_state.theme:
            # Use theme manager's set_theme function to update Streamlit's theme
            from utils.theme_manager import set_theme
            set_theme(selected_theme)
            st.rerun()
            
        # Remove AI Provider section as requested
        
        # Display last data fetch time
        if 'last_data_fetch_time' in st.session_state:
            st.caption(f"Last data fetched: {st.session_state.last_data_fetch_time}")
        
        # Display economic indicators in an enhanced format
        st.subheader("Economic Indicators")
        indicators = fetch_economic_indicators()
        
        # Create a DataFrame for better display
        indicator_df = pd.DataFrame({
            "Indicator": list(indicators.keys()),
            "Value": list(indicators.values()),
        })
        
        # Get the current date/time for freshness indicator
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Use full width for the economic indicators table
        # Create a stylish table with CSS
        st.markdown("""
        <style>
        .economic-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }
        .economic-table th {
            background-color: #262730;
            color: white;
            text-align: left;
            padding: 12px 15px;
            font-size: 16px;
            font-weight: bold;
        }
        .economic-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            font-size: 15px;
        }
        .economic-table tr:nth-child(even) {
            background-color: rgba(240, 240, 240, 0.5);
        }
        .economic-table tr:hover {
            background-color: rgba(144, 238, 144, 0.1);
        }
        .data-source {
            font-size: 14px;
            color: #555;
            font-style: italic;
            margin-top: 10px;
        }
        .freshness {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the indicators in HTML table
        html_table = "<table class='economic-table'><tr><th>Indicator</th><th>Value</th><th>Impact</th></tr>"
        
        # Define impacts for each indicator
        impacts = {
            'GDP Growth Rate': "Higher GDP growth indicates economic expansion, positive for corporate earnings",
            'Inflation Rate': "Moderate inflation (3-6%) is ideal; higher values can lead to interest rate hikes",
            'Interest Rates': "Higher rates increase borrowing costs and may slow economic growth",
            'Rupee Exchange Rate': "A stronger rupee benefits importers but may hurt exporters",
            'Crude Oil Prices': "Higher oil prices increase input costs for many sectors, especially transportation",
            'Fiscal Deficit': "Higher deficit may lead to increased government borrowing, affecting interest rates",
            'Market Liquidity': "Higher liquidity generally supports stock prices and economic activity",
            'FPI Flows': "Positive flows indicate foreign investor confidence in the Indian market"
        }
        
        # Add rows with impact information
        for i, row in indicator_df.iterrows():
            indicator = row['Indicator']
            impact = impacts.get(indicator, "")
            html_table += f"<tr><td><strong>{indicator}</strong></td><td>{row['Value']}</td><td>{impact}</td></tr>"
        
        html_table += "</table>"
        html_table += "<div class='data-source'>Data source: Real-time economic data via Claude AI</div>"
        html_table += f"<div class='freshness'>Last updated: {current_time}</div>"
        
        st.markdown(html_table, unsafe_allow_html=True)
    
    # Check if stock or time period changed
    stock_changed = st.session_state.selected_stock != selected_stock
    period_changed = st.session_state.time_period != selected_period
    
    # Update selected stock and time period in session state
    st.session_state.selected_stock = selected_stock
    st.session_state.time_period = selected_period
    
    # Check if data source changed
    data_source_changed = getattr(st.session_state, 'data_source_changed', False)
    # Reset the flag
    st.session_state.data_source_changed = False
    
    # Fetch data if needed, if update requested, or if stock/period/data source changed
    if update_button or ai_news_button or st.session_state.stock_data is None or stock_changed or period_changed or data_source_changed:
        with st.spinner("Fetching stock data..."):
            # Get stock name, sector, etc. once to avoid redundant calls
            stock_name = get_stock_name_from_symbol(selected_stock)
            sector = get_sector_for_stock(selected_stock)
            
            # Create placeholder for progress reporting
            progress_container = st.empty()
            
            # Progress reporting function
            def update_progress(msg, pct=None):
                with progress_container.container():
                    if pct is not None:
                        st.progress(pct)
                    st.write(msg)
            
            update_progress("Fetching price data...", 0.1)
            
            # Get the user's data source preference
            use_ai_for_data = st.session_state.use_ai_data
            
            # Only use database cache if not updating and data source hasn't changed
            use_db_cache = not update_button and not data_source_changed
            
            if use_db_cache:
                # Try to get data from database first
                db_data = get_stock_price_data(selected_stock)
                
                if not db_data.empty:
                    st.session_state.stock_data = db_data
                    update_progress("Loaded price data from database", 0.3)
                    use_db_cache = True
                else:
                    use_db_cache = False
            
            if not use_db_cache:
                # Get the selected data source from session state
                selected_data_source = st.session_state.data_exchange
                
                # Determine data source label based on selection
                if selected_data_source == 'alpha_vantage':
                    update_progress("Fetching from Alpha Vantage API...", 0.2)
                    data_source_label = "Alpha Vantage API"
                else:  # 'yahoo' or any other default
                    update_progress("Fetching from Yahoo Finance API...", 0.2)
                    data_source_label = "Yahoo Finance API"
                    selected_data_source = 'yahoo'  # Ensure it's explicitly set
                
                # Fetch the data
                # Use the selected stock exchange (NSE or BSE) when the data source is Yahoo
                exchange_param = selected_stock_exchange.lower() if selected_data_source == 'yahoo' else selected_data_source
                
                st.session_state.stock_data = fetch_stock_data(
                    selected_stock, 
                    period=selected_period,
                    use_ai=use_ai,  # Just use the checkbox value
                    exchange=exchange_param
                )
                
                # Add data source info to session state and update fetch timestamp
                st.session_state.data_source = data_source_label
                st.session_state.last_data_fetch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # If data is empty, show an error message
                if st.session_state.stock_data is not None and st.session_state.stock_data.empty and selected_data_source == 'yahoo':
                    update_progress(f"No data available from {selected_stock_exchange} via Yahoo Finance.", 0.2)
                    st.error(f"No stock data available for {selected_stock} from the selected source. Please try a different stock or data source.")
                
                # Save to database in background (avoid blocking the UI)
                if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
                    try:
                        save_stock_data(selected_stock, stock_name, sector, selected_index, st.session_state.stock_data)
                        update_progress(f"Stock data saved to database (source: {data_source_label})", 0.3)
                    except Exception as e:
                        print(f"Error saving stock data: {e}")
            
            # Fetch news in parallel to speed things up
            update_progress("Fetching news and sentiment data...", 0.4)
            
            # Get news and analyze
            # Standard news fetching from official sources
            update_progress("Fetching news from financial data sources...", 0.45)
            st.session_state.news_list = fetch_news(stock_name)
            update_progress("News data retrieved successfully", 0.5)
            
            update_progress("Analyzing sentiment...", 0.6)
            sentiment_data = analyze_news_sentiment(st.session_state.news_list)
            st.session_state.news_list = sentiment_data['news']
            st.session_state.sentiment_summary = sentiment_data['overview']['counts']
            st.session_state.sentiment_df = sentiment_data['time_series']
            
            # Save news to database in background without blocking UI
            try:
                save_news_articles(st.session_state.news_list, selected_stock)
            except Exception as e:
                print(f"Error saving news: {e}")
            
            # Get economic factors
            update_progress("Calculating economic factors...", 0.8)
            
            if ai_news_button:
                # User clicked AI-powered research button - use ChatGPT4o for enhanced economic research
                update_progress("Retrieving economic data from financial sources...", 0.85)
                try:
                    from utils.web_scraper import get_economic_research_from_apis
                    
                    # Get economic research from financial data APIs
                    economic_research = get_economic_research_from_apis(selected_stock, sector)
                    
                    # Format and store the research results
                    if economic_research and isinstance(economic_research, dict):
                        # Store in session state
                        st.session_state.economic_factors = {
                            'detailed': {},  # Will be populated below
                            'flat': {},  # Legacy format
                            'india_specific': True,
                            'sector_score': economic_research.get('sector_score', 0),
                            'economic_health_index': economic_research.get('economic_health_index', 50),
                            'economic_condition': economic_research.get('economic_condition', 'Neutral'),
                            'metadata': {
                                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'source': 'AI-Powered Research'
                            }
                        }
                        
                        # Extract key economic indicators
                        if 'key_indicators' in economic_research:
                            for key, value in economic_research['key_indicators'].items():
                                st.session_state.economic_factors['detailed'][key] = {
                                    'value': value,
                                    'weight': 1.0,
                                    'raw_impact': 0.5,
                                    'adjusted_impact': 0.5,
                                    'value_for_prediction': 0.5
                                }
                        
                        update_progress("Economic data retrieval completed", 0.9)
                    else:
                        # Fall back to standard calculation
                        st.session_state.economic_factors = calculate_weighted_factors(sector)
                except Exception as e:
                    st.warning(f"Error retrieving economic data: {str(e)}")
                    # Fall back to standard calculation
                    st.session_state.economic_factors = calculate_weighted_factors(sector)
            else:
                # Standard economic factors calculation
                st.session_state.economic_factors = calculate_weighted_factors(sector)
            
            # Save economic indicators to database in background
            try:
                # Use AI-powered indicators if the AI research button was clicked
                indicators = fetch_economic_indicators()
                from utils.database import save_economic_indicators
                save_economic_indicators(indicators)
                print(f"Successfully saved economic indicators to database")
            except Exception as e:
                print(f"Error with economic indicators: {e}")
                
            update_progress("Data loading complete!", 1.0)
            progress_container.empty()
    
    # Generate predictions if requested
    if predict_button:
        with st.spinner("Generating predictions..."):
            try:
                # Make a copy of the stock data and ensure all dates are timezone naive
                stock_data_copy = st.session_state.stock_data.copy()
                
                # If the index has timezone info, convert to timezone naive
                if hasattr(stock_data_copy.index, 'tz') and stock_data_copy.index.tz is not None:
                    stock_data_copy.index = stock_data_copy.index.tz_localize(None)
                
                # Now train and predict with timezone-consistent data
                # Check if economic_factors is in the new format, and extract the flat data if needed
                economic_data = st.session_state.economic_factors
                if isinstance(economic_data, dict) and 'flat' in economic_data:
                    # Using new enhanced economic factors format
                    economic_factors_for_model = economic_data['flat']
                else:
                    # Using old format
                    economic_factors_for_model = economic_data
                
                st.session_state.predictions = train_and_predict(
                    stock_data_copy,
                    economic_factors_for_model,
                    st.session_state.sentiment_df
                )
                
                # Save prediction to database - create a naive datetime for consistency
                target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                save_prediction(selected_stock, st.session_state.predictions, target_date)
                
                st.success("Predictions generated successfully and saved to database!")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Error generating predictions: {str(e)}")
                st.code(error_details)
                print(f"Prediction error: {str(e)}\n{error_details}")
                st.session_state.predictions = None
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Chart", "News & Sentiment", "Economic Factors", "Simulation", "History"])
    
    # Tab 1: Price Chart
    with tab1:
        st.subheader(f"Price Chart for {get_stock_name_from_symbol(selected_stock)}")
        
        if st.session_state.stock_data is not None:
            # Get latest date
            try:
                if not st.session_state.stock_data.empty:
                    latest_date_str = str(st.session_state.stock_data.index.max())
                    latest_date_obj = pd.to_datetime(latest_date_str)
                    latest_date = latest_date_obj.strftime('%Y-%m-%d')
                    st.write(f"Latest data as of: **{latest_date}**")
                else:
                    st.write("No data available")
            except:
                st.write("Date information not available")
            
            # Add metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate metrics
            current_price = st.session_state.stock_data['Close'].iloc[-1]
            prev_price = st.session_state.stock_data['Close'].iloc[-2] if len(st.session_state.stock_data) > 1 else current_price
            change = current_price - prev_price
            pct_change = (change / prev_price) * 100 if prev_price > 0 else 0
            
            # Calculate moving averages
            ma50 = st.session_state.stock_data['Close'].rolling(window=50).mean().iloc[-1] if len(st.session_state.stock_data) >= 50 else current_price
            ma200 = st.session_state.stock_data['Close'].rolling(window=200).mean().iloc[-1] if len(st.session_state.stock_data) >= 200 else current_price
            
            # Display metrics
            col1.metric("Current Price", f"₹{current_price:.2f}", f"{pct_change:.2f}%")
            col2.metric("Day Range", f"₹{st.session_state.stock_data['Low'].iloc[-1]:.2f} - ₹{st.session_state.stock_data['High'].iloc[-1]:.2f}")
            col3.metric("50-Day MA", f"₹{ma50:.2f}", f"{((current_price/ma50)-1)*100:.2f}%")
            col4.metric("200-Day MA", f"₹{ma200:.2f}", f"{((current_price/ma200)-1)*100:.2f}%")
            
            # Display chart options
            chart_col1, chart_col2 = st.columns([2, 3])
            
            with chart_col1:
                chart_type = st.radio(
                    "Chart Type:",
                    ["Line Chart", "Bar Chart", "Candlestick", "OHLC"],
                    horizontal=True
                )
                
                # Time period for visualization (may be different from data fetching period)
                viz_time_periods = {
                    "1 Week": 7,
                    "1 Month": 30,
                    "3 Months": 90,
                    "6 Months": 180,
                    "1 Year": 365,
                    "2 Years": 730,
                    "3 Years": 1095,
                    "5 Years": 1825,
                    "All Data": None
                }
                
                viz_period = st.selectbox(
                    "Visualization Period:",
                    list(viz_time_periods.keys()),
                    index=min(3, len(viz_time_periods) - 1)  # Default to 6 Months
                )
            
            with chart_col2:
                # Technical indicators selection
                tech_indicators = st.multiselect(
                    "Technical Indicators:",
                    ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Volume"],
                    default=["Moving Averages"]
                )
            
            # Calculate technical indicators
            stock_data_with_indicators = st.session_state.stock_data.copy()
            
            # Calculate and add selected technical indicators
            if "Moving Averages" in tech_indicators:
                # Add more comprehensive moving averages
                stock_data_with_indicators['MA20'] = stock_data_with_indicators['Close'].rolling(window=20).mean()
                stock_data_with_indicators['MA50'] = stock_data_with_indicators['Close'].rolling(window=50).mean()
                stock_data_with_indicators['MA100'] = stock_data_with_indicators['Close'].rolling(window=100).mean()
                stock_data_with_indicators['MA200'] = stock_data_with_indicators['Close'].rolling(window=200).mean()
                
            if "Bollinger Bands" in tech_indicators:
                # Calculate Bollinger Bands (20-day, 2 standard deviations)
                window = 20
                stock_data_with_indicators['MA20'] = stock_data_with_indicators['Close'].rolling(window=window).mean()
                stock_data_with_indicators['BB_upper'] = stock_data_with_indicators['MA20'] + 2 * stock_data_with_indicators['Close'].rolling(window=window).std()
                stock_data_with_indicators['BB_lower'] = stock_data_with_indicators['MA20'] - 2 * stock_data_with_indicators['Close'].rolling(window=window).std()
                
            if "RSI" in tech_indicators:
                # Calculate RSI (14-day)
                delta = stock_data_with_indicators['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # First RSI calculation
                rs = avg_gain / avg_loss
                stock_data_with_indicators['RSI'] = 100 - (100 / (1 + rs))
                
            if "MACD" in tech_indicators:
                # Calculate MACD
                exp1 = stock_data_with_indicators['Close'].ewm(span=12, adjust=False).mean()
                exp2 = stock_data_with_indicators['Close'].ewm(span=26, adjust=False).mean()
                stock_data_with_indicators['MACD'] = exp1 - exp2
                stock_data_with_indicators['MACD_signal'] = stock_data_with_indicators['MACD'].ewm(span=9, adjust=False).mean()
                stock_data_with_indicators['MACD_hist'] = stock_data_with_indicators['MACD'] - stock_data_with_indicators['MACD_signal']
            
            # Filter data based on selected visualization period
            days_to_display = viz_time_periods[viz_period]
            if days_to_display is not None:
                # Get the end date (latest date in data)
                end_date = stock_data_with_indicators.index.max()
                # Calculate start date based on selected period
                start_date = end_date - pd.Timedelta(days=days_to_display)
                # Filter data
                display_data = stock_data_with_indicators.loc[stock_data_with_indicators.index >= start_date]
            else:
                # Use all data
                display_data = stock_data_with_indicators
            
            # Get current theme for visualization
            current_theme = get_current_theme()
            
            # Plot based on selected chart type
            if chart_type == "Line Chart":
                # Create a line chart
                fig = go.Figure()
                
                # Add closing price line
                fig.add_trace(go.Scatter(
                    x=display_data.index, 
                    y=display_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=current_theme["primary_color"], width=2)
                ))
                
                # Add technical indicators if selected
                if "Moving Averages" in tech_indicators:
                    if 'MA20' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA20'], mode='lines', name='20-Day MA', line=dict(color=current_theme["visualization_colors"][0], width=1.5)))
                    if 'MA50' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA50'], mode='lines', name='50-Day MA', line=dict(color=current_theme["visualization_colors"][1], width=1.5)))
                    if 'MA100' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA100'], mode='lines', name='100-Day MA', line=dict(color=current_theme["visualization_colors"][2], width=1.5)))
                    if 'MA200' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA200'], mode='lines', name='200-Day MA', line=dict(color=current_theme["visualization_colors"][3], width=1.5)))
                
                if "Bollinger Bands" in tech_indicators:
                    if 'BB_upper' in display_data.columns and 'BB_lower' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['BB_upper'], mode='lines', name='Bollinger Upper', line=dict(color=current_theme["visualization_colors"][4], width=1, dash='dash')))
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['BB_lower'], mode='lines', name='Bollinger Lower', line=dict(color=current_theme["visualization_colors"][4], width=1, dash='dash')))
                
                # Add volume in a separate subplot if selected
                if "Volume" in tech_indicators:
                    fig.add_trace(go.Bar(
                        x=display_data.index, 
                        y=display_data['Volume'] if 'Volume' in display_data.columns else display_data['Shares Traded'],
                        name='Volume',
                        marker=dict(color=current_theme["neutral_color"], opacity=0.5),
                        yaxis='y2'
                    ))
                    
                    # Update layout to include a secondary y-axis for volume
                    fig.update_layout(
                        yaxis2=dict(
                            title='Volume',
                            titlefont=dict(color=current_theme["neutral_color"]),
                            tickfont=dict(color=current_theme["neutral_color"]),
                            overlaying='y',
                            side='right',
                            showgrid=False
                        )
                    )
                
                # Add predictions if available
                if st.session_state.predictions is not None and 'ensemble_prediction' in st.session_state.predictions:
                    next_day = display_data.index[-1] + pd.Timedelta(days=1)
                    fig.add_trace(go.Scatter(
                        x=[next_day],
                        y=[st.session_state.predictions['ensemble_prediction']],
                        mode='markers',
                        name='Next Day Prediction',
                        marker=dict(color='yellow', size=12, symbol='star')
                    ))
                
                # Get the latest date for the chart
                try:
                    if not display_data.empty:
                        latest_date_str = str(display_data.index.max())
                        latest_date_obj = pd.to_datetime(latest_date_str)
                        latest_date = latest_date_obj.strftime('%Y-%m-%d')
                    else:
                        latest_date = "N/A"
                except:
                    latest_date = "N/A"
                
                # Update layout
                fig.update_layout(
                    title=f"{get_stock_name_from_symbol(selected_stock)} - Price History ({viz_period}) - Latest: {latest_date}",
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    legend_title='Legend',
                    hovermode='x unified'
                )
                
                # Apply theme
                fig = apply_theme_to_plotly_figure(fig, current_theme)
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Bar Chart":
                # Create a bar chart that resembles the reference image
                fig = go.Figure()
                
                # Get closing prices in ascending order
                sorted_prices = display_data['Close'].copy().reset_index(drop=True)
                sorted_prices = sorted_prices.sort_values().reset_index()
                
                # Create a blue color gradient
                blue_scale = [
                    "#8DA9C4",  # Lightest blue
                    "#5085A5",
                    "#31708E",
                    "#226291",
                    "#1A5276",
                    "#154360",
                    "#0F3A5F",  # Darkest blue
                ]
                
                # Create a color gradient based on position
                num_bars = len(sorted_prices)
                colors = []
                
                for i in range(num_bars):
                    idx = min(int(i / num_bars * len(blue_scale)), len(blue_scale) - 1)
                    colors.append(blue_scale[idx])
                
                # Add vertical bars in ascending order
                fig.add_trace(go.Bar(
                    x=sorted_prices.index,  # Use indices as x-values for evenly spaced bars
                    y=sorted_prices['Close'],
                    name='Closing Price',
                    marker_color=colors,
                    marker_line_width=0
                ))
                
                # Remove x-axis labels
                fig.update_xaxes(showticklabels=False)
                
                # Format y-axis
                fig.update_yaxes(
                    title='Price (₹)',
                    gridcolor=current_theme["chart_grid"],
                    showgrid=True
                )
                
                # Update layout to be clean and minimalist
                fig.update_layout(
                    title=f"{get_stock_name_from_symbol(selected_stock)} - Price Distribution ({viz_period})",
                    xaxis_title='Price Points',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                # Apply theme
                fig = apply_theme_to_plotly_figure(fig, current_theme)
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Candlestick":
                # Create a candlestick chart
                fig = go.Figure()
                
                # Add candlestick trace
                fig.add_trace(go.Candlestick(
                    x=display_data.index,
                    open=display_data['Open'],
                    high=display_data['High'],
                    low=display_data['Low'],
                    close=display_data['Close'],
                    name='Price',
                    increasing_line_color=current_theme["up_color"],
                    decreasing_line_color=current_theme["down_color"]
                ))
                
                # Add technical indicators if selected
                if "Moving Averages" in tech_indicators:
                    if 'MA20' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA20'], mode='lines', name='20-Day MA', line=dict(color=current_theme["visualization_colors"][0], width=1.5)))
                    if 'MA50' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA50'], mode='lines', name='50-Day MA', line=dict(color=current_theme["visualization_colors"][1], width=1.5)))
                
                if "Bollinger Bands" in tech_indicators:
                    if 'BB_upper' in display_data.columns and 'BB_lower' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['BB_upper'], mode='lines', name='Bollinger Upper', line=dict(color=current_theme["visualization_colors"][4], width=1, dash='dash')))
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['BB_lower'], mode='lines', name='Bollinger Lower', line=dict(color=current_theme["visualization_colors"][4], width=1, dash='dash')))
                
                # Add volume in a separate subplot if selected
                if "Volume" in tech_indicators:
                    fig.add_trace(go.Bar(
                        x=display_data.index, 
                        y=display_data['Volume'] if 'Volume' in display_data.columns else display_data['Shares Traded'],
                        name='Volume',
                        marker=dict(color=current_theme["neutral_color"], opacity=0.5),
                        yaxis='y2'
                    ))
                    
                    # Update layout to include a secondary y-axis for volume
                    fig.update_layout(
                        yaxis2=dict(
                            title='Volume',
                            titlefont=dict(color=current_theme["neutral_color"]),
                            tickfont=dict(color=current_theme["neutral_color"]),
                            overlaying='y',
                            side='right',
                            showgrid=False
                        )
                    )
                
                # Get the latest date for the chart
                try:
                    if not display_data.empty:
                        latest_date_str = str(display_data.index.max())
                        latest_date_obj = pd.to_datetime(latest_date_str)
                        latest_date = latest_date_obj.strftime('%Y-%m-%d')
                    else:
                        latest_date = "N/A"
                except:
                    latest_date = "N/A"
                
                # Update layout
                fig.update_layout(
                    title=f"{get_stock_name_from_symbol(selected_stock)} - Candlestick Chart ({viz_period}) - Latest: {latest_date}",
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    legend_title='Legend',
                    hovermode='x unified'
                )
                
                # Apply theme
                fig = apply_theme_to_plotly_figure(fig, current_theme)
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # OHLC chart
                # Create an OHLC chart
                fig = go.Figure()
                
                # Add OHLC trace
                fig.add_trace(go.Ohlc(
                    x=display_data.index,
                    open=display_data['Open'],
                    high=display_data['High'],
                    low=display_data['Low'],
                    close=display_data['Close'],
                    name='Price',
                    increasing_line_color=current_theme["up_color"],
                    decreasing_line_color=current_theme["down_color"]
                ))
                
                # Add technical indicators if selected
                if "Moving Averages" in tech_indicators:
                    if 'MA20' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA20'], mode='lines', name='20-Day MA', line=dict(color=current_theme["visualization_colors"][0], width=1.5)))
                    if 'MA50' in display_data.columns:
                        fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA50'], mode='lines', name='50-Day MA', line=dict(color=current_theme["visualization_colors"][1], width=1.5)))
                
                # Update layout
                fig.update_layout(
                    title=f"{get_stock_name_from_symbol(selected_stock)} - OHLC Chart ({viz_period})",
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    legend_title='Legend',
                    hovermode='x unified'
                )
                
                # Apply theme
                fig = apply_theme_to_plotly_figure(fig, current_theme)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Download historical data as CSV or Excel
            st.subheader("Download Historical Data")
            col1, col2 = st.columns(2)
            
            # Convert DataFrame to CSV and Excel for download
            # Make a copy to handle the timezone issue
            export_df = st.session_state.stock_data.copy()
            
            # Convert timezone-aware dates to timezone-naive
            if hasattr(export_df.index, 'tz') and export_df.index.tz is not None:
                export_df.index = export_df.index.tz_localize(None)
                
            csv = export_df.to_csv().encode('utf-8')
            
            # Create Excel file
            excel_buffer = pd.ExcelWriter("stock_data.xlsx", engine='xlsxwriter')
            export_df.to_excel(excel_buffer, sheet_name='Stock Data')
            excel_buffer.close()  # Use close() instead of save()
            
            with open("stock_data.xlsx", "rb") as f:
                excel_data = f.read()
            
            with col1:
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_stock}_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{selected_stock}_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Display data table
            st.subheader("Historical Data Table")
            st.dataframe(st.session_state.stock_data, use_container_width=True)
            
            # If predictions available, show comparison plots
            if st.session_state.predictions is not None:
                st.subheader("Model Predictions Comparison")
                
                # Display visualization options
                viz_type = st.radio(
                    "Select Visualization Type:",
                    ["Standard Comparison", "Detailed Model Breakdown"],
                    horizontal=True
                )
                
                if viz_type == "Standard Comparison":
                    fig = plot_prediction_comparison(st.session_state.predictions)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = plot_model_predictions_vertical(st.session_state.predictions)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Allow downloading the prediction report
                st.subheader("Prediction Report")
                
                # Generate HTML report
                stock_name = get_stock_name_from_symbol(selected_stock)
                sector = get_sector_for_stock(selected_stock)
                report_html = generate_prediction_report(
                    st.session_state.stock_data,
                    st.session_state.predictions,
                    selected_stock,
                    stock_name,
                    selected_index
                )
                
                # Save report to file for download
                with open("prediction_report.html", "w") as f:
                    f.write(report_html)
                
                with open("prediction_report.html", "rb") as f:
                    report_data = f.read()
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Download HTML Report",
                        data=report_data,
                        file_name=f"{selected_stock}_prediction_{datetime.now().strftime('%Y-%m-%d')}.html",
                        mime="text/html"
                    )
                
                with col2:
                    # Generate DOCX report
                    docx_file = generate_docx_report(
                        st.session_state.stock_data,
                        st.session_state.predictions,
                        tech_indicators,
                        st.session_state.sentiment_summary,
                        st.session_state.economic_factors
                    )
                    
                    with open(docx_file, "rb") as f:
                        docx_data = f.read()
                    
                    st.download_button(
                        label="Download DOCX Report",
                        data=docx_data,
                        file_name=f"{selected_stock}_prediction_{datetime.now().strftime('%Y-%m-%d')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                
                # Display preview
                with st.expander("Preview Report"):
                    st.components.v1.html(report_html, height=600)
        else:
            st.warning("No data available for selected stock.")
    
    # Tab 2: News & Sentiment
    with tab2:
        st.subheader(f"News & Sentiment Analysis for {get_stock_name_from_symbol(selected_stock)}")
        
        if st.session_state.news_list:
            # Split into two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display news
                st.write("### Latest News")
                for news in st.session_state.news_list:
                    sentiment_color = {
                        'positive': 'green',
                        'negative': 'red',
                        'neutral': 'gray'
                    }[news['sentiment']]
                    
                    source = news.get('source', 'Financial News')
                    url = news.get('url', '#')
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {sentiment_color}; padding-left: 10px; margin-bottom: 15px;">
                        <h4 style="margin:0">{news['title']}</h4>
                        <p style="margin:2px 0; color: gray;">{news['date']} • <a href="{url}" target="_blank">{source}</a></p>
                        <p>{news['description']}</p>
                        <p style="color: {sentiment_color}; font-weight: bold;">{news['sentiment'].capitalize()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Display sentiment analysis
                st.write("### Sentiment Distribution")
                fig = plot_sentiment_analysis(st.session_state.sentiment_summary)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("### Sentiment Trend")
                fig = plot_sentiment_time_series(st.session_state.sentiment_df)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No news available for selected stock.")
    
    # Tab 3: Economic Factors
    with tab3:
        st.subheader(f"Economic Factors for {get_stock_name_from_symbol(selected_stock)}")
        
        # Get sector for the selected stock
        sector = get_sector_for_stock(selected_stock)
        st.write(f"Sector: **{sector}**")
        
        # Create two columns for economic analysis options
        econ_col1, econ_col2 = st.columns(2)
        
        with econ_col1:
            # Button to perform deep research using AI
            research_button = st.button("Perform Deep Economic Research", 
                                       help="Uses AI to analyze how economic factors affect this specific stock")
        
        with econ_col2:
            # Button to analyze sector economic trends
            sector_button = st.button(f"Analyze {sector} Sector", 
                                     help="Uses AI to analyze how economic factors affect this entire sector")
        
        # Initialize session state for research if not exists
        if 'stock_economic_research' not in st.session_state:
            st.session_state.stock_economic_research = None
            
        if 'sector_economic_analysis' not in st.session_state:
            st.session_state.sector_economic_analysis = None
        
        # Perform deep research if button clicked
        if research_button:
            with st.spinner("Performing deep economic research on this stock using AI..."):
                from utils.economic_factors import research_stock_economic_impact
                
                try:
                    # Get stock information
                    stock_name = get_stock_name_from_symbol(selected_stock)
                    
                    # Perform the research
                    research_results = research_stock_economic_impact(
                        stock_name=stock_name,
                        stock_symbol=selected_stock,
                        sector=sector
                    )
                    
                    # Store in session state
                    st.session_state.stock_economic_research = research_results
                    
                    # Reset sector analysis to avoid confusion
                    st.session_state.sector_economic_analysis = None
                except Exception as e:
                    st.error(f"Error performing economic research: {str(e)}")
                    st.session_state.stock_economic_research = {"error": f"Research failed: {str(e)}. This could be due to connectivity issues or data availability."}
        
        # Perform sector analysis if button clicked
        if sector_button:
            with st.spinner(f"Analyzing {sector} sector economic trends using AI..."):
                from utils.economic_factors import get_sector_impact_summary
                
                try:
                    # Perform the sector analysis
                    sector_results = get_sector_impact_summary(sector)
                    
                    # Store in session state
                    st.session_state.sector_economic_analysis = sector_results
                    
                    # Reset stock research to avoid confusion
                    st.session_state.stock_economic_research = None
                except Exception as e:
                    st.error(f"Error performing sector analysis: {str(e)}")
                    st.session_state.sector_economic_analysis = {"error": f"Analysis failed: {str(e)}. This could be due to connectivity issues or data availability."}
        
        # Display research results if available
        if st.session_state.stock_economic_research is not None:
            research = st.session_state.stock_economic_research
            
            # Check if there was an error
            if 'error' in research:
                st.error(f"Research Error: {research['error']}")
            else:
                st.subheader("🔍 Economic Impact Analysis")
                
                # Display summary
                st.markdown(f"### Summary\n{research.get('summary', 'No summary available')}")
                
                # Display factor impacts in an expander
                with st.expander("Detailed Factor Impact Analysis", expanded=True):
                    # Get factor impacts
                    factor_impacts = research.get('factor_impacts', {})
                    
                    for factor, impact in factor_impacts.items():
                        st.markdown(f"**{factor}**")
                        st.write(impact)
                        st.markdown("---")
                
                # Display outlook
                st.markdown(f"### Economic Outlook\n{research.get('outlook', 'No outlook available')}")
                
                # Show metadata
                if 'metadata' in research:
                    st.caption(f"Analysis generated: {research['metadata'].get('generated_at', 'Unknown')} | "
                              f"Data source: {research['metadata'].get('data_source', 'Unknown')}")
        
        # Display sector analysis if available
        elif st.session_state.sector_economic_analysis is not None:
            analysis = st.session_state.sector_economic_analysis
            
            # Check if there was an error
            if 'error' in analysis:
                st.error(f"Sector Analysis Error: {analysis['error']}")
            else:
                st.subheader(f"🏭 {sector} Sector Economic Analysis")
                
                # Display summary
                st.markdown(f"### Overview\n{analysis.get('summary', 'No summary available')}")
                
                # Display factor impacts in an expander
                with st.expander("Sector-Specific Factor Impacts", expanded=True):
                    # Get factor impacts
                    factor_impacts = analysis.get('factor_impacts', {})
                    
                    for factor, impact in factor_impacts.items():
                        st.markdown(f"**{factor}**")
                        st.write(impact)
                        st.markdown("---")
                
                # Display opportunities and threats in columns
                opportunities = analysis.get('opportunities', [])
                threats = analysis.get('threats', [])
                
                if opportunities or threats:
                    op_col, th_col = st.columns(2)
                    
                    with op_col:
                        st.markdown("### 🚀 Economic Opportunities")
                        for opportunity in opportunities:
                            st.markdown(f"- {opportunity}")
                    
                    with th_col:
                        st.markdown("### ⚠️ Economic Threats")
                        for threat in threats:
                            st.markdown(f"- {threat}")
                
                # Display comparison with other sectors
                if 'comparison' in analysis:
                    st.markdown("### Sector Comparison")
                    st.write(analysis['comparison'])
                
                # Show metadata
                if 'metadata' in analysis:
                    st.caption(f"Analysis generated: {analysis['metadata'].get('generated_at', 'Unknown')} | "
                              f"Data source: {analysis['metadata'].get('data_source', 'Unknown')}")
        
        # Display original economic factors influence
        if not research_button and not sector_button:
            st.write("### Impact of Economic Factors")
            
            # Get sector weights
            sector_weights = get_sector_weights(sector)
            
            # Check if economic_factors is in the new format
            economic_data = st.session_state.economic_factors
            india_specific = False
            
            if isinstance(economic_data, dict) and 'detailed' in economic_data:
                # Using enhanced India-specific economic data
                india_specific = True
                
                # Display text summary of economic conditions instead of gauge charts
                if 'economic_health_index' in economic_data and 'sector_score' in economic_data:
                    health_index = economic_data['economic_health_index']
                    health_condition = economic_data.get('economic_condition', 'Neutral')
                    sector_score = economic_data['sector_score']
                    
                    # Create a simple card with the information
                    with st.container():
                        st.subheader("Economic Indicators")
                        st.write(f"**Economic Condition:** {health_condition}")
                        st.write(f"**Economic Health Index:** {health_index:.1f}/100")
                        st.write(f"**Sector Score Impact:** {sector_score:.1f}/10")
                        
                        # Color-coded description of market conditions
                        if health_index >= 65:
                            st.success("Market conditions are favorable for this sector")
                        elif health_index >= 45:
                            st.info("Market conditions are neutral for this sector")
                        else:
                            st.warning("Market conditions are challenging for this sector")
                
                # Information box about India-specific adjustments
                st.info("Using enhanced economic factor analysis with India-specific market dynamics")
                
                # Create a dataframe for displaying detailed economic factor information
                factor_data = []
                for factor, details in economic_data['detailed'].items():
                    factor_data.append({
                        'Factor': factor,
                        'Current Value': details['value'],
                        'Weight': details['weight'],
                        'Raw Impact': details['raw_impact'],
                        'Adjusted Impact': details['adjusted_impact'],
                        'India-Specific Adjustment': details['adjusted_impact'] / details['raw_impact'] if details['raw_impact'] != 0 else 1.0
                    })
                
                factor_df = pd.DataFrame(factor_data)
                
                # Plot using the adjusted impact values
                factors_for_plot = {}
                for factor, details in economic_data['detailed'].items():
                    factors_for_plot[factor] = details['adjusted_impact']
                
                # Display factor influence visualization
                st.subheader("Factor Influence")
                fig = plot_economic_factors_influence(factors_for_plot)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display tabular data
                st.write("### Economic Factors Detail")
                st.dataframe(factor_df.style.format({
                    'Current Value': '{:.2f}',
                    'Weight': '{:.2f}',
                    'Raw Impact': '{:.2f}',
                    'Adjusted Impact': '{:.2f}',
                    'India-Specific Adjustment': '{:.2f}x'
                }), use_container_width=True)
                
            else:
                # Using traditional economic data
                fig = plot_economic_factors_influence(sector_weights)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display explanation of factors
            st.write("### Factor Explanations")
            
            from utils.economic_factors import get_factor_impact_explanation
            explanations = get_factor_impact_explanation(sector)
            
            # Display overall explanation first
            if 'overall' in explanations:
                st.markdown(f"""
                <div style="margin-bottom: 20px; padding: 15px; background-color: rgba(0,0,0,0.05); border-radius: 5px;">
                    <h4 style="margin:0">Overall Impact Assessment</h4>
                    <p>{explanations['overall']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Remove overall from explanations to avoid displaying it twice
                overall_explanation = explanations.pop('overall', None)
            
            # Define sectors and their special characteristics for Indian market
            india_sector_insights = {
                'Information Technology': "Export-oriented sector highly sensitive to Rupee-Dollar exchange rates and global IT spending.",
                'Pharmaceuticals': "Benefits from a weaker Rupee due to export focus, but import costs for raw materials can offset gains.",
                'Financial Services': "Highly sensitive to interest rates and RBI monetary policy changes.",
                'Energy': "Crude oil price fluctuations have significant impact as India imports over 80% of its oil needs.",
                'Automotive': "Affected by both interest rates (vehicle financing) and fuel prices. Electric vehicle transition is reshaping sensitivities.",
                'Real Estate': "Interest rate changes significantly impact both developer financing costs and home loan affordability."
            }
            
            # Display India-specific sector insight if available
            if sector in india_sector_insights:
                st.markdown(f"""
                <div style="margin-bottom: 20px; padding: 10px; background-color: rgba(255,223,0,0.1); border-left: 5px solid rgba(255,223,0,0.5); border-radius: 2px;">
                    <h4 style="margin:0">India Market Insight: {sector}</h4>
                    <p>{india_sector_insights[sector]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display individual factor explanations
            if india_specific:
                # Sort factors by adjusted impact for India-specific data
                sorted_factors = sorted(
                    economic_data['detailed'].items(),
                    key=lambda x: abs(x[1]['adjusted_impact']),
                    reverse=True
                )
                
                for factor, details in sorted_factors[:5]:  # Show top 5 by impact
                    if factor in explanations:
                        adjustment = details['adjusted_impact'] / details['raw_impact'] if details['raw_impact'] != 0 else 1.0
                        adjustment_text = ""
                        if adjustment > 1.05:
                            adjustment_text = f"<span style='color:green'>Enhanced by {(adjustment-1)*100:.1f}% for Indian market</span>"
                        elif adjustment < 0.95:
                            adjustment_text = f"<span style='color:orange'>Reduced by {(1-adjustment)*100:.1f}% for Indian market</span>"
                            
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #f0f0f0; border-radius: 5px;">
                            <h4 style="margin:0">{factor}</h4>
                            <p><strong>Value:</strong> {details['value']:.2f} | <strong>Weight:</strong> {details['weight']:.2f} | <strong>Impact:</strong> {details['adjusted_impact']:.2f}</p>
                            <p>{adjustment_text}</p>
                            <p>{explanations[factor]}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # For traditional data, sort by weight
                for factor, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                    if factor in explanations:
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; border: 1px solid #f0f0f0; border-radius: 5px;">
                            <h4 style="margin:0">{factor}</h4>
                            <p><strong>Weight:</strong> {weight:.2f}</p>
                            <p>{explanations[factor]}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("No economic factors available for selected stock sector.")
    
    # Tab 4: Simulation
    with tab4:
        st.subheader(f"Price Simulation for {get_stock_name_from_symbol(selected_stock)}")
        
        # Generate simulation
        if st.session_state.stock_data is not None:
            # Simulation parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                days = st.slider("Simulation Days", 7, 90, 30)
            
            with col2:
                num_scenarios = st.slider("Number of Scenarios", 10, 200, 100)
            
            with col3:
                volatility_factor = st.slider("Volatility Factor", 0.5, 2.0, 1.0, step=0.1)
            
            # Generate and display simulation
            fig = generate_simulation_plot(st.session_state.stock_data, st.session_state.predictions, days)
            st.plotly_chart(fig, use_container_width=True)
            
            # Simulation explanation
            st.markdown("""
            ### About This Simulation
            
            This simulation shows potential price paths over the selected time period based on:
            
            1. **Historical Volatility**: Calculated from past price movements
            2. **Model Predictions**: Used as the starting point for projections
            3. **Monte Carlo Method**: Generates multiple random scenarios
            
            The gold line represents the average projected path, while the shaded area shows the 95% confidence interval.
            
            *Note: This simulation is for educational purposes only and should not be considered financial advice.*
            """)
        else:
            st.warning("No data available for simulation.")
    
    # Tab 5: History
    with tab5:
        st.subheader("Prediction History")
        
        # First, update actual closing values for predictions with passed target dates
        with st.spinner("Updating actual closing values for past predictions..."):
            from utils.database import update_actual_closing_prices
            updated_count = update_actual_closing_prices()
            if updated_count > 0:
                st.success(f"Updated actual closing prices for {updated_count} predictions")
        
        # Add filter options
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Show all stocks", value=False)
        with col2:
            limit = st.slider("Number of records", 5, 50, 20)
        
        # Show last update time
        last_update_time = st.session_state.get('last_data_fetch_time', 'Unknown')
        st.caption(f"Data last fetched: {last_update_time}")
        
        # Get prediction history from database
        if show_all:
            predictions = get_latest_predictions(limit=limit)
        else:
            predictions = get_latest_predictions(symbol=selected_stock, limit=limit)
        
        if predictions:
            # Convert to DataFrame for easier display
            data = []
            data_raw = []  # For exporting raw data without formatting
            
            for pred in predictions:
                # Get stock symbol
                stock_symbol = pred.stock.symbol if hasattr(pred.stock, 'symbol') else "Unknown"
                stock_name = pred.stock.name if hasattr(pred.stock, 'name') else "Unknown"
                
                # Format date
                prediction_date = pred.prediction_date.strftime('%Y-%m-%d %H:%M') if pred.prediction_date else "Unknown"
                target_date = pred.target_date.strftime('%Y-%m-%d') if pred.target_date else "Unknown"
                
                # Get predictions
                ensemble = pred.ensemble_prediction
                rf = pred.random_forest_prediction
                xgb = pred.xgboost_prediction
                lgbm = pred.lightgbm_prediction
                svm = pred.svm_prediction
                knn = pred.knn_prediction
                arima = pred.arima_prediction
                
                # Determine accuracy if actual price is available
                accuracy = "N/A"
                accuracy_val = None
                if pred.actual_price and ensemble:
                    error = abs(pred.actual_price - ensemble) / pred.actual_price * 100
                    accuracy = f"{100 - error:.2f}%"
                    accuracy_val = 100 - error
                
                # For display (formatted values)
                data.append({
                    "Stock": f"{stock_name} ({stock_symbol})",
                    "Symbol": stock_symbol,
                    "Prediction Date": prediction_date,
                    "Target Date": target_date,
                    "Ensemble": f"₹{ensemble:.2f}" if ensemble else "N/A",
                    "Random Forest": f"₹{rf:.2f}" if rf else "N/A",
                    "XGBoost": f"₹{xgb:.2f}" if xgb else "N/A",
                    "LightGBM": f"₹{lgbm:.2f}" if lgbm else "N/A",
                    "SVM": f"₹{svm:.2f}" if svm else "N/A",
                    "KNN": f"₹{knn:.2f}" if knn else "N/A",
                    "ARIMA": f"₹{arima:.2f}" if arima else "N/A",
                    "Actual": f"₹{pred.actual_price:.2f}" if pred.actual_price else "Not Yet Available",
                    "Accuracy": accuracy
                })
                
                # For export (raw values)
                data_raw.append({
                    "Stock Name": stock_name,
                    "Stock Symbol": stock_symbol,
                    "Prediction Date": prediction_date,
                    "Target Date": target_date,
                    "Ensemble Prediction": ensemble,
                    "Random Forest Prediction": rf,
                    "XGBoost Prediction": xgb,
                    "LightGBM Prediction": lgbm,
                    "SVM Prediction": svm,
                    "KNN Prediction": knn,
                    "ARIMA Prediction": arima,
                    "Actual Price": pred.actual_price,
                    "Accuracy (%)": accuracy_val
                })
            
            # Create DataFrames
            df = pd.DataFrame(data)
            df_raw = pd.DataFrame(data_raw)
            
            # Display the table
            st.dataframe(df)
            
            # Prepare for download
            st.subheader("Download Prediction History")
            col1, col2 = st.columns(2)
            
            # Convert DataFrame to CSV and Excel for download
            
            # Make a copy to handle time zones
            export_df = df_raw.copy()
            
            # Convert timezone-aware dates to timezone-naive
            for col in export_df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']):
                if hasattr(export_df[col], 'dt') and hasattr(export_df[col].dt, 'tz'):
                    if export_df[col].dt.tz is not None:
                        export_df[col] = export_df[col].dt.tz_localize(None)
            
            csv = export_df.to_csv(index=False).encode('utf-8')
            
            # Create Excel file
            excel_buffer = pd.ExcelWriter("prediction_history.xlsx", engine='xlsxwriter')
            export_df.to_excel(excel_buffer, sheet_name='Prediction History', index=False)
            
            # Format the Excel file
            workbook = excel_buffer.book
            worksheet = excel_buffer.sheets['Prediction History']
            
            # Add header formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            # Apply header format
            for col_num, value in enumerate(export_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Adjust column widths
            for col_num, column in enumerate(export_df.columns):
                column_width = max(len(str(column)), export_df[column].astype(str).map(len).max())
                worksheet.set_column(col_num, col_num, column_width + 2)
            
            excel_buffer.close()  # Use close() instead of save()
            
            with open("prediction_history.xlsx", "rb") as f:
                excel_data = f.read()
            
            # Download buttons
            with col1:
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Visualization of prediction history
            if len(df) > 1:
                st.subheader("Prediction Accuracy Visualization")
                
                # Filter to only rows with actual prices and accuracy
                df_with_accuracy = df_raw[df_raw['Accuracy (%)'].notna()]
                
                if not df_with_accuracy.empty:
                    # Create a bar chart of prediction accuracy by date
                    fig = px.bar(
                        df_with_accuracy,
                        x='Target Date',
                        y='Accuracy (%)',
                        color='Stock Symbol',
                        title='Prediction Accuracy by Date',
                        labels={'Accuracy (%)': 'Accuracy (%)', 'Target Date': 'Target Date'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a grouped bar chart comparing predicted vs actual
                    df_compare = df_with_accuracy[['Target Date', 'Stock Symbol', 'Ensemble Prediction', 'Actual Price']]
                    df_compare_melt = pd.melt(
                        df_compare, 
                        id_vars=['Target Date', 'Stock Symbol'],
                        value_vars=['Ensemble Prediction', 'Actual Price'],
                        var_name='Price Type',
                        value_name='Price'
                    )
                    
                    fig = px.bar(
                        df_compare_melt,
                        x='Target Date',
                        y='Price',
                        color='Price Type',
                        barmode='group',
                        facet_row='Stock Symbol' if show_all else None,
                        title='Predicted vs Actual Prices',
                        labels={'Price': 'Price (₹)', 'Target Date': 'Target Date'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No records with actual prices yet for visualization. Accuracy will be available once the target date has passed.")
            
            # Add explanation
            st.markdown("""
            ### About This History
            
            This table shows the history of predictions made by the system. For each prediction:
            
            - **Stock**: The stock symbol and name
            - **Prediction Date**: When the prediction was made
            - **Target Date**: The date for which the price was predicted
            - **Model Predictions**: Price predictions from different models
            - **Actual**: The actual closing price on the target date (if available)
            - **Accuracy**: How close the ensemble prediction was to the actual price
            
            Predictions for future dates will show "Not Yet Available" for the actual price.
            You can download this data in CSV or Excel format for further analysis.
            """)
        else:
            st.warning("No prediction history available. Make some predictions first!")

def render_upload_data():
    """Render the page for uploading custom data."""
    st.title("Upload Your Own Data")
    
    st.write("""
    ### Upload Custom Stock Data
    
    Use this page to upload your own historical stock data for analysis and prediction.
    
    **Acceptable formats:**
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    
    **Required columns:**
    - Date
    - Open
    - High
    - Low
    - Close
    - Shares Traded / Volume
    - Turnover (₹ Cr) / Value
    """)
    
    from utils.data_fetcher import process_uploaded_file
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            with st.spinner("Processing uploaded data..."):
                data = process_uploaded_file(uploaded_file)
                
                # Store in session state
                st.session_state.stock_data = data
                
                # Clear previous predictions and related data
                st.session_state.predictions = None
                st.session_state.news_list = None
                st.session_state.sentiment_summary = None
                st.session_state.sentiment_df = None
                st.session_state.economic_factors = None
                
                # Show success message
                st.success("Data uploaded successfully!")
                
                # Show sample of the data
                st.write("### Sample Data")
                st.dataframe(data.head())
                
                # Navigation button
                if st.button("Proceed to Analysis", type="primary"):
                    st.session_state.page = "dashboard"
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def render_about():
    """Render the about page."""
    st.title("About This Application")
    
    st.write("""
    ## Stock Price Prediction Dashboard
    
    This application helps you predict stock prices using multiple machine learning and deep learning models.
    
    ### Features
    
    - **Multiple Models**: Combines traditional ML (Random Forest, XGBoost, LightGBM, SVM, KNN), time series (ARIMA), and deep learning (LSTM, RNN) for more robust predictions
    - **Sentiment Analysis**: Analyzes news sentiment to understand market perception
    - **Economic Factors**: Incorporates relevant economic indicators based on sector
    - **Interactive Visualizations**: Dynamic charts for better insights
    - **Price Simulation**: Monte Carlo simulation for potential future scenarios
    
    ### How It Works
    
    1. **Data Collection**: Historical price data, news, and economic indicators
    2. **Preprocessing**: Technical indicators, normalization, and feature engineering
    3. **Model Training**: Multiple models trained on historical data
    4. **Ensemble Prediction**: Weighted combination of individual model predictions
    5. **Visualization**: Interactive charts and insights
    
    ### Disclaimer
    
    This tool is for educational and research purposes only. It does not constitute financial advice, and investment decisions should not be made solely based on its predictions.
    
    ### Credits
    
    Developed using:
    - Streamlit for the web interface
    - Pandas and NumPy for data manipulation
    - Scikit-learn, TensorFlow, XGBoost, and LightGBM for machine learning
    - NLTK for sentiment analysis
    - Plotly for interactive visualizations
    - yfinance for stock data retrieval
    """)

# Main application
def main():
    # Initialize session state with default values
    initialize_session_state()
    
    # Initialize theme and ensure it's applied
    initialize_theme()
    
    # Get current theme for UI components
    current_theme = get_current_theme()
    
    # Authentication is now handled in auth_app.py
    # This section now assumes the user is already authenticated
    
    # Get user information from session state (set in auth_app.py)
    name = st.session_state.get("user_name", "User")
    username = st.session_state.get("username", "")
    
    # Display welcome message in sidebar
    st.sidebar.success(f"Welcome, {name}")
    
    # Add logout button with unique key
    if st.sidebar.button("Logout", key="main_logout_button"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Redirect to login page
        st.switch_page("auth_app.py")
        
        # Navigation menu in sidebar
        with st.sidebar:
            st.title("Navigation")
            
            # Determine current page
            if 'page' not in st.session_state:
                st.session_state.page = "dashboard"
            
        # Navigation buttons with unique keys
        if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
            
        if st.button("Upload Data", key="nav_upload", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
            
        if st.button("About", key="nav_about", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()
    
    # Render the appropriate page
    if st.session_state.page == "dashboard":
        render_dashboard()
    elif st.session_state.page == "upload":
        render_upload_data()
    elif st.session_state.page == "about":
        render_about()

# Create .streamlit/config.toml if it doesn't exist
def ensure_streamlit_config():
    """Ensure the Streamlit configuration file exists with headless mode enabled."""
    import os
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    # Create config directory if it doesn't exist
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    # Check if config file exists and contains server settings
    config_exists = os.path.exists(config_file)
    has_server_config = False
    
    if config_exists:
        with open(config_file, "r") as f:
            has_server_config = "[server]" in f.read()
    
    # Add server configuration if needed
    if not config_exists or not has_server_config:
        with open(config_file, "a") as f:
            f.write("\n[server]\nheadless = true\naddress = \"0.0.0.0\"\nport = 5000\n")

if __name__ == "__main__":
    ensure_streamlit_config()
    main()