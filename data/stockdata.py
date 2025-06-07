import pandas as pd
import numpy as np
from data.marketdata import MarketData
import yfinance as yf
from tools.search_online import search_ddg_news, extract_article_text
from utils.summarization import summarize_news
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO

market = MarketData()
# yf.set_config(
#     proxy = None
# )

def fetch_financial_data(ticker, start_date, end_date):
    """Fetch historical price data and fundamental metrics for a given ticker."""
    try:
        hist_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        yf_ticker = yf.Ticker(ticker)
        fundamentals = yf_ticker.info
        return hist_data, fundamentals
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), {}


def calculate_statistics(hist_data, fundamentals):
    """Calculate technical indicators and fundamental metrics."""
    stats = {}
    
    # Technical indicators
    if not hist_data.empty:
        # Price data
        stats['current_price'] = hist_data['Close'].iloc[-1].item()
        
        # Calculate 52-week high/low using 252 trading days approximation
        stats['52w_high'] = hist_data['High'].rolling(252).max().iloc[-1].item()
        stats['52w_low'] = hist_data['Low'].rolling(252).min().iloc[-1].item()

        # Moving averages
        stats['sma_50'] = hist_data['Close'].rolling(50).mean().iloc[-1].item()
        stats['sma_200'] = hist_data['Close'].rolling(200).mean().iloc[-1].item()

        # RSI calculation with proper scalar handling
        delta = hist_data['Close'].diff().dropna()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(14).mean().iloc[-1].item()  # Get last value as scalar
        avg_loss = loss.rolling(14).mean().iloc[-1].item()   # Get last value as scalar

        if avg_loss == 0:
            stats['rsi'] = 100.0
        else:
            rs = avg_gain / avg_loss
            stats['rsi'] = 100.0 - (100.0 / (1 + rs))

        # Volume and volatility
        stats['volume_avg_20d'] = hist_data['Volume'].rolling(20).mean().iloc[-1].item()
        stats['volatility'] = hist_data['Close'].pct_change().std().item() * 252**0.5

    fundamental_map = {
        'pe_ratio': 'trailingPE',
        'eps': 'trailingEps',
        'market_cap': 'marketCap',
        'pb_ratio': 'priceToBook',
        'dividend_yield': 'dividendYield',
        'beta': 'beta'
    }
    
    for stat, key in fundamental_map.items():
        value = fundamentals.get(key)
        if value is not None:
            # Special handling for dividend yield
            if stat == 'dividend_yield' and value is not None:
                stats[stat] = value * 100
            # Convert market cap to billions
            elif stat == 'market_cap' and value is not None:
                stats[stat] = value / 1e9
            else:
                stats[stat] = value

    # Remove None/NaN values
    return pd.DataFrame([(k, float(v)) for k, v in stats.items() if v is not None and not pd.isna(v)],columns=["Metric", "Value"])


def generate_financial_summary(stats_df):
    """Generate a formatted financial summary from calculated statistics."""
    if stats_df.empty:
        return "No data available"
    
    stats = stats_df.iloc[0]
    summary = []
    
    # Price Analysis
    price_info = []
    for metric in ['current_price', '52w_high', '52w_low']:
        if metric in stats:
            price_info.append(f"{metric.replace('_', ' ').title()}: ${stats[metric]:.2f}")
    if price_info:
        summary.append("**Price Analysis**\n- " + "\n- ".join(price_info))
    
    # Technical Indicators
    tech_info = []
    for metric, fmt in [('sma_50', '${:.2f}'), ('sma_200', '${:.2f}'), ('rsi', '{:.1f}')]:
        if metric in stats:
            tech_info.append(f"{metric.upper().replace('_', ' ')}: {fmt.format(stats[metric])}")
    if tech_info:
        summary.append("**Technical Indicators**\n- " + "\n- ".join(tech_info))
    
    # Fundamentals
    fundamental_info = []
    for metric, fmt in [('pe_ratio', '{:.1f}'), ('market_cap', '${:.2f}B'), 
                        ('pb_ratio', '{:.2f}'), ('dividend_yield', '{:.2f}%')]:
        if metric in stats:
            fundamental_info.append(f"{metric.upper().replace('_', ' ')}: {fmt.format(stats[metric])}")
    if fundamental_info:
        summary.append("**Fundamental Metrics**\n- " + "\n- ".join(fundamental_info))
    
    # Risk Analysis
    risk_info = []
    if 'volatility' in stats:
        risk_info.append(f"Annualized Volatility: {stats['volatility']:.2%}")
    if 'beta' in stats:
        risk_info.append(f"Beta: {stats['beta']:.2f}")
    if risk_info:
        summary.append("**Risk Analysis**\n- " + "\n- ".join(risk_info))
    
    # Insights
    insights = []
    if 'rsi' in stats:
        if stats['rsi'] > 70:
            insights.append("Overbought (RSI > 70)")
        elif stats['rsi'] < 30:
            insights.append("Oversold (RSI < 30)")
    
    if 'sma_50' in stats and 'sma_200' in stats:
        if stats['sma_50'] > stats['sma_200']:
            insights.append("Golden Cross (50D SMA > 200D SMA)")
    
    if insights:
        summary.append("**Market Insights**\n- " + "\n- ".join(insights))
    
    return "\n\n".join(summary)



def generate_plots(data, company, start_date, end_date):
    # Convert dates and filter data
    data.index = pd.to_datetime(data.index)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    mask = (data.index >= start_date) & (data.index <= end_date)
    filtered_data = data.loc[mask]
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_data.index, filtered_data['Close'], 
            label=f'{company} Stock Price', color='#1f77b4', linewidth=2)
    
    ax.set_title(f'{company} Stock Price: {start_date.date()} to {end_date.date()}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True)
    
    # Save to numpy array
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Convert to numpy array
    img = plt.imread(buf)
    img_np = (img * 255).astype(np.uint8)  # Convert to 0-255 scale
    
    return img_np

    

def fetch_data(company: str, start_date: str, end_date: str):
    global financial_summary_context, latest_ticker
    ticker = market.company_mapping[company]
    latest_ticker = ticker  # Store this globally so it can be used for news fetching.
    data,fundamentals = fetch_financial_data(ticker, start_date, end_date)
    if data is not  None or not data.empty:
        stats = calculate_statistics(data,fundamentals)
        financial_summary = generate_financial_summary(stats)
        financial_summary_context = financial_summary  # Update global context.
        plot = generate_plots(data, company,start_date, end_date)
    else:
        stats = pd.DataFrame(columns=["Metric", "Value"])
        financial_summary = "No financial data available for this company."
        plot = np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder image  
    raw_news = search_ddg_news(f"{company} financial news", max_results=5)
    news_text = "\n".join([
        f"Title:{article['title']}\n url:({result['url']}): \n text:{article['text'][:500]}"
        for result in raw_news if (article := extract_article_text(result))])
    
    news_summary = summarize_news(news_text)

    return stats, plot, news_summary if news_summary else "No recent news found",{"finance":financial_summary,"news":news_text}




