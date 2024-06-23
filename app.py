import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Function to fetch data with different intervals
def fetch_data(ticker, interval, start_date):
    df = yf.download(ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'), interval=interval)
    return df

# Function to calculate EMA
def calculate_ema(df, window):
    return df['Close'].ewm(span=window, adjust=False).mean()

# Function to calculate EMA crossover signals
def ema_crossover_signals(df, short_window=9, long_window=20):
    df['EMA9'] = calculate_ema(df, short_window)
    df['EMA20'] = calculate_ema(df, long_window)
    df['EMA50'] = calculate_ema(df, 50)  # Adding EMA 50 calculation

    df['Signal'] = 0
    if len(df) > short_window:
        df.loc[df.index[short_window:], 'Signal'] = np.where(df['EMA9'][short_window:] > df['EMA20'][short_window:], 1, 0)
    df['Position'] = df['Signal'].diff()

    crossover_dates = df[df['Position'] != 0].index

    return df, crossover_dates

# Function to backtest strategies
def backtest_strategies(df):
    if len(df) == 0:
        return 0.0, 0.0

    initial_capital = 10000
    capital = initial_capital
    shares = 0
    buy_hold_values = [initial_capital]
    crossover_values = [initial_capital]

    buy_hold_return = 0.0
    crossover_return = 0.0

    # Buy and hold strategy
    buy_hold_shares = capital / df['Close'][0]
    buy_hold_value = buy_hold_shares * df['Close'][-1]
    buy_hold_return = (buy_hold_value - initial_capital) / initial_capital * 100

    # EMA crossover strategy
    for i in range(1, len(df)):
        if df['Signal'][i] == 1 and capital > 0:  # Buy signal
            shares = capital / df['Close'][i - 1]
            capital = 0
        elif df['Signal'][i] == -1 and shares > 0:  # Sell signal
            capital = shares * df['Close'][i]
            shares = 0

        crossover_value = shares * df['Close'][i] if shares > 0 else capital
        crossover_values.append(crossover_value)

    crossover_return = (crossover_values[-1] - initial_capital) / initial_capital * 100

    return buy_hold_return, crossover_return

# Streamlit app
st.title('EMA Crossovers and Backtesting Results')

# Dropdown filter for intervals
interval = st.selectbox('Select Interval', ['1d', '1h'])

# Determine start date based on selected interval
if interval == '1d':
    start_date = datetime.now() - timedelta(days=90)
elif interval == '1h':
    start_date = datetime.now() - timedelta(days=30)

# List of tickers
tickers = ['SPY', 'QQQ', 'IWM', 'META', 'NVDA', 'AAPL', 'TSLA', 'AMD', 'PLTR', 'TSM', 'MU', 'AMZN', 'DELL',
           'MSFT', 'ARM']

# Initialize a list to store tickers with crossovers
tickers_with_crossovers = []
recent_crossovers = []

# Check for crossovers for each ticker
for ticker in tickers:
    df = fetch_data(ticker, interval, start_date)
    df, crossover_dates = ema_crossover_signals(df)
    if len(crossover_dates) > 0:
        tickers_with_crossovers.append({
            'Ticker': ticker,
            'Crossover Dates': crossover_dates,
            'Data': df
        })
        # Check for crossovers in the last 5 days
        recent_crossover_dates = crossover_dates[crossover_dates > (datetime.now() - timedelta(days=5))]
        if len(recent_crossover_dates) > 0:
            for date in recent_crossover_dates:
                crossover_close = df.loc[date, 'Close']
                current_close = df['Close'].iloc[-1]
                percent_change = ((current_close - crossover_close) / crossover_close) * 100
                crossover_type = "Bull" if df.loc[date, 'EMA9'] > df.loc[date, 'EMA20'] else "Bear"
                recent_crossovers.append({
                    'Ticker': ticker,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Crossover Close': crossover_close,
                    'Current Price': current_close,
                    'Percent Change': percent_change,
                    'Crossover Type': crossover_type
                })

# Display tickers with recent crossovers in a dataframe
if len(recent_crossovers) > 0:
    st.subheader('Recent EMA Crossovers (Last 5 Days)')
    recent_crossovers_df = pd.DataFrame(recent_crossovers)
    st.dataframe(recent_crossovers_df)

# Display tickers with crossovers and backtesting results
if len(tickers_with_crossovers) > 0:
    for ticker_data in tickers_with_crossovers:
        st.write(f"**{ticker_data['Ticker']}**")

        # Perform backtesting
        buy_hold_return, crossover_return = backtest_strategies(ticker_data['Data'])

        # Print backtesting results
        st.write(f"Buy & Hold Strategy Return: {buy_hold_return:.2f}%")
        st.write(f"EMA Crossover Strategy Return: {crossover_return:.2f}%")

        # Plot data with EMA crossovers
        fig = go.Figure()

        # Closing price trace
        fig.add_trace(go.Scatter(x=ticker_data['Data'].index, y=ticker_data['Data']['Close'],
                                 mode='lines', line=dict(color='blue', width=1),
                                 name='Close Price'))

        # EMA 9 trace
        fig.add_trace(go.Scatter(x=ticker_data['Data'].index, y=ticker_data['Data']['EMA9'],
                                 mode='lines', line=dict(color='green', width=1),
                                 name='EMA 9'))

        # EMA 20 trace
        fig.add_trace(go.Scatter(x=ticker_data['Data'].index, y=ticker_data['Data']['EMA20'],
                                 mode='lines', line=dict(color='orange', width=1),
                                 name='EMA 20'))

        # EMA 50 trace
        fig.add_trace(go.Scatter(x=ticker_data['Data'].index, y=ticker_data['Data']['EMA50'],
                                 mode='lines', line=dict(color='red', width=1),
                                 name='EMA 50'))

        # Add crossover annotations
        annotations = [dict(x=date, y=ticker_data['Data'].loc[date, 'Close'],
                            xref="x", yref="y",
                            text="EMA Crossover", showarrow=True,
                            arrowhead=1, ax=-50, ay=-30) for date in ticker_data['Crossover Dates']]

        fig.update_layout(title=f"{ticker_data['Ticker']} Price with EMA Crossovers ({interval} Interval)",
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False,
                          annotations=annotations)

        # Print the chart
        st.plotly_chart(fig, use_container_width=True)

        st.write('---')
else:
    st.write('No tickers found with EMA crossovers in the last selected interval.')
