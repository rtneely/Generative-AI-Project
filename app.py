import streamlit as st
import json
import boto3

from dotenv import load_dotenv
import os

load_dotenv()

import yfinance as yf
import mplfinance as mpf

def get_stock_chart(ticker, period='6mo', interval='1d'):
    try:
        # Normalize period input (fixes '6M' → '6mo')
        period_map = {
            "1D": "1d", "5D": "5d", "1W": "5d", "1MO": "1mo", "3MO": "3mo",
            "6M": "6mo", "6MO": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y",
            "10Y": "10y", "YTD": "ytd", "MAX": "max"
        }
        period = period_map.get(period.upper(), period.lower())

        # Fetch historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)

        # Handle empty or too-small datasets
        if hist.empty or len(hist) < 2:
            print(f"No price data found for {ticker}")
            return {
                "chart_file": None,
                "trend_summary": f"No price data available for {ticker}."
            }

        # Moving averages
        hist['MA10'] = hist['Close'].rolling(10).mean()
        hist['MA50'] = hist['Close'].rolling(50).mean()

        # Plot overlays
        apds = [
            mpf.make_addplot(hist['MA10'], color='orange', width=1),
            mpf.make_addplot(hist['MA50'], color='blue', width=1)
        ]

        # Save chart
        filename = f'{ticker}_chart.png'
        mpf.plot(
            hist,
            type='candle',
            style='nightclouds',
            title=f'{ticker} Candlestick Chart ({period})',
            ylabel='Price',
            volume=True,
            addplot=apds,
            panel_ratios=(5,1),
            figratio=(16,9),
            figscale=1.5,
            savefig=filename
        )

        # Trend summary
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        pct_change = round((end_price - start_price) / start_price * 100, 2)

        trend = "increased" if pct_change > 0 else "decreased" if pct_change < 0 else "remained stable"

        return {
            "chart_file": filename,
            "trend_summary": f"The stock has {trend} by {abs(pct_change)}% over the past {period}."
        }

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return {
            "chart_file": None,
            "trend_summary": f"Could not generate chart for {ticker} due to an error."
        }

import json

# Load your name-to-ticker map
with open("russell3000_final_company_map.json", "r") as f:
    name_to_ticker = json.load(f)

# Reverse it: ticker → most descriptive name
from collections import defaultdict

ticker_to_names = defaultdict(list)
for name, ticker in name_to_ticker.items():
    ticker_to_names[ticker.upper()].append(name)

# Choose the longest name per ticker (e.g., "Apple Inc." over "Apple")
TICKER_TO_NAME = {
    ticker: max(names, key=len)
    for ticker, names in ticker_to_names.items()
}

def get_company_name(ticker):
    name = TICKER_TO_NAME.get(ticker.upper(), ticker)
    return name.title()

from datetime import datetime, timedelta, timezone
import os
import requests

def get_scored_headlines(ticker, max_articles=5):
    company_name = get_company_name(ticker)
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=3)

    url = (
        f"https://finnhub.io/api/v1/company-news?symbol={ticker}&"
        f"from={from_date.strftime('%Y-%m-%d')}&to={to_date.strftime('%Y-%m-%d')}&"
        f"token={FINNHUB_API_KEY}"
    )
    response = requests.get(url)
    articles = response.json()
    if not articles:
        return []

    headlines = [a["headline"] for a in sorted(articles, key=lambda x: x["datetime"], reverse=True)]

    financial_keywords = [
        "earnings", "revenue", "guidance", "investigation", "sec", "merger",
        "acquisition", "buyback", "dividend", "regulator", "lawsuit", "fine",
        "profit", "loss", "forecast", "upgrade", "recall"
    ]

    block_keywords = [
        "portfolio", "etf", "fund", "commentary", "buffett", "top stocks", "picks",
        "retirement", "passive income", "newsletter", "magnificent 7", "millionaire"
    ]

    def score_headline(h):
        score = 0
        h_lower = h.lower()
        words = h_lower.split()

        if any(k in h_lower for k in financial_keywords):
            score += 1
        if company_name.lower() in h_lower or ticker.lower() in h_lower:
            score += 1
        if any(company_name.lower() in w or ticker.lower() in w for w in words[:6]):
            score += 1
        if not any(k in h_lower for k in block_keywords):
            score += 1

        # Penalize vague or opinion-based headlines
        if "according to" in h_lower or "best stock" in h_lower or "top stock" in h_lower:
            score -= 1

        # Penalize multi-company fluff
        if "," in h and any(t in h for t in ["AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA", "NVDA"]):
            score -= 1

        return score

    scored = sorted([(score_headline(h), h) for h in headlines], key=lambda x: x[0], reverse=True)
    return [h for score, h in scored[:max_articles]]

def get_stock_chart_tool(inputs):
    ticker = inputs.get("ticker")
    period_raw = inputs.get("period", "6mo")
    interval_raw = inputs.get("interval", "1d")

    # Normalize period
    period_map = {
        "6M": "6mo", "6MO": "6mo", "1Y": "1y", "YTD": "ytd"
    }
    period = period_map.get(period_raw.upper(), period_raw.lower())

    # Force interval override for stability
    safe_intervals = {"1d", "5d", "1mo"}
    interval = interval_raw.lower() if interval_raw.lower() in safe_intervals else "1d"

    return get_stock_chart(ticker, period, interval)


def get_fundamentals(ticker):
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if 'metric' not in data:
        return {}

    m = data['metric']
    fundamentals = {}

    def format_market_cap(value):
        if not value:
            return None
        if value >= 1_000_000:
            return f"${value / 1_000_000:.2f} trillion"
        elif value >= 1_000:
            return f"${value / 1_000:.2f} billion"
        else:
            return f"${value:.2f} million"

    def format_number(value, decimals=2):
        if value in [None, 0, "N/A"]:
            return None
        return f"{round(value, decimals)}"

    def format_percent(value, decimals=2):
        if value in [None, 0, "N/A"]:
            return None
        return f"{round(value, decimals)}%"

    # Only add if value exists
    market_cap = format_market_cap(m.get("marketCapitalization"))
    if market_cap: fundamentals["Market Cap"] = market_cap

    pe_ratio = format_number(m.get("peInclExtraTTM"))
    if pe_ratio: fundamentals["P/E Ratio"] = pe_ratio

    eps = format_number(m.get("epsInclExtraItemsTTM"))
    if eps: fundamentals["EPS"] = eps

    revenue_growth = m.get("revenueGrowthTTM")
    if revenue_growth: fundamentals["Revenue Growth (YoY)"] = format_percent(revenue_growth * 100)

    roe = format_percent(m.get("roeTTM"))
    if roe: fundamentals["ROE"] = roe

    div_yield = format_percent(m.get("dividendYieldIndicatedAnnual"))
    if div_yield: fundamentals["Dividend Yield"] = div_yield

    return fundamentals


def get_fundamentals_tool(inputs):
    ticker = inputs.get("ticker")
    data = get_fundamentals(ticker)

    if not data:
        return {
            "text": f"No fundamental data is currently available for {ticker.upper()}."
        }

    def escape_dollar(text):
        return text.replace("$", "\\$") if isinstance(text, str) else text

    # Escape values that contain dollar signs
    for k in data:
        data[k] = escape_dollar(data[k])

    parts = []

    if "Market Cap" in data:
        parts.append(f"{ticker.upper()} has a market capitalization of {data['Market Cap']}.")
    if "P/E Ratio" in data and "EPS" in data:
        parts.append(f"It has a P/E ratio of {data['P/E Ratio']} and earnings per share (EPS) of {data['EPS']}.")
    elif "P/E Ratio" in data:
        parts.append(f"It has a P/E ratio of {data['P/E Ratio']}.")
    elif "EPS" in data:
        parts.append(f"It has earnings per share (EPS) of {data['EPS']}.")
    if "ROE" in data:
        parts.append(f"It has a return on equity (ROE) of {data['ROE']}.")
    if "Dividend Yield" in data:
        parts.append(f"It has a dividend yield of {data['Dividend Yield']}.")

    summary = " ".join(parts)

    return {
        "text": summary
    }

stock_chart_spec = {
    "name": "get_stock_chart",
    "description": "Returns a candlestick chart of a stock.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {"type": "string"},
                "interval": {"type": "string"}
            },
            "required": ["ticker"]
        }
    }
}

fundamentals_spec = {
    "name": "get_fundamentals",
    "description": "Returns key fundamentals for a company from Finnhub.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"]
        }
    }
}

def compare_stocks_tool(inputs):
    import yfinance as yf
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    tickers = inputs.get("tickers")
    period = inputs.get("period", "1mo")
    interval = inputs.get("interval", "1d")

    if not tickers or not isinstance(tickers, list) or len(tickers) < 2 or len(tickers) > 5:
        return {"error": "Please provide between 2 and 5 tickers in a list."}

    price_data = {}
    try:
        for ticker in tickers:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            if hist.empty:
                return {"error": f"Price data not available for {ticker}"}
            hist['Pct Change'] = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
            price_data[ticker] = hist['Pct Change']

        df = pd.DataFrame(price_data).dropna()

        # Plot chart
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#00FFAA', '#FFD700', '#FF6347', '#1E90FF', '#ADFF2F']
        for i, ticker in enumerate(tickers):
            color = colors[i % len(colors)]
            ax.plot(df.index, df[ticker], label=ticker, color=color, linewidth=2.5)
            ax.text(df.index[-1], df[ticker].iloc[-1], f"{ticker}: {df[ticker].iloc[-1]:.2f}%", color=color, fontsize=9, va='center')

        ax.set_title(f"{', '.join(tickers)} Stock Comparison ({period})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Return (% from start)", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

        chart_file = f"compare_chart_{'_'.join(tickers)}.png"
        plt.savefig(chart_file, dpi=300)
        plt.close()

        # Build performance + volatility summary
        performance = {t: f"{df[t].iloc[-1]:.2f}%" for t in tickers}
        volatility = {t: f"{df[t].std():.2f}%" for t in tickers}
        summary = ", ".join([f"{t}: {performance[t]} (vol: {volatility[t]})" for t in tickers])

        return {
            "tickers": tickers,
            "performance": performance,
            "volatility": volatility,
            "chart_file": chart_file,
            "trend_summary": f"Over the past {period}, {summary}."
        }

    except Exception as e:
        return {"error": str(e)}
    
def summarize_sentiment_tool(inputs):
    headlines = inputs.get("headlines", [])
    
    if isinstance(headlines, str):
        headlines = [h.strip() for h in headlines.split(",") if h.strip()]
    
    if not headlines:
        return {
            "summary": "No headlines provided.",
            "headline_count": 0,
            "status": "complete",
            "importance": "Sentiment unavailable due to lack of headlines."
        }

    # Deduplicate and normalize
    normalized = sorted(set(h.lower().strip() for h in headlines))

    import json

    if isinstance(headlines, str) and headlines.startswith("["):
        try:
            headlines = json.loads(headlines)
        except json.JSONDecodeError:
            headlines = [headlines]

    return {
    "headlines_used": normalized,
    "headline_count": len(normalized),
    "status": "complete",
    "importance": ( "Analyze the following headlines to determine overall tone. "
        "Write 1–2 sentences that clearly state whether the sentiment is mostly positive, negative, or mixed, "
        "and briefly explain why based on headline content. "
        "Name at least 2 of the most relevant headlines in your response to support your summary."
        )
}

def get_recent_news_tool(inputs):
    ticker = inputs.get("ticker")
    max_articles = inputs.get("max_articles", 5)
    headlines = get_scored_headlines(ticker, max_articles)
    return {"headlines": headlines}

compare_stocks_spec = {
    "name": "compare_stocks",
    "description": (
        "Compares the stock performance of 2 to 5 companies over a given time period. "
        "Generates a percent-return chart, performance summary, and volatility for each ticker."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 5
                },
                "period": {
                    "type": "string",
                    "description": "Time period for the comparison, e.g. '1mo', '6mo', '1y'"
                },
                "interval": {
                    "type": "string",
                    "description": "Data interval (e.g., '1d', '1wk')",
                    "default": "1d"
                }
            },
            "required": ["tickers"]
        }
    }
}


summarize_sentiment_spec = {
    "name": "summarize_sentiment",
    "description": (
    "Returns a structured list of headlines for the LLM to analyze. You must decide the sentiment (positive, negative, or mixed). "
    "Use this only if the query relates to public sentiment, news tone, or recent headlines."
),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "headlines": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["headlines"]
        }
    }
}

get_recent_news_spec = {
    "name": "get_recent_news",
    "description": "Gets recent news headlines for a company.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "max_articles": {"type": "integer"}
            },
            "required": ["ticker"]
        }
    }
}

tool_config = {
    "tools": [
        {"toolSpec": stock_chart_spec},
        {"toolSpec": fundamentals_spec},
        {"toolSpec": get_recent_news_spec},
        {"toolSpec": summarize_sentiment_spec},
        {"toolSpec": compare_stocks_spec}
    ],
    "toolChoice": {"auto": {}}
}

def run_tool(name, inputs):
    if name == "get_stock_chart":
        return get_stock_chart_tool(inputs)
    elif name == "get_fundamentals":
        return get_fundamentals_tool(inputs)
    elif name == "compare_stocks":
        return compare_stocks_tool(inputs)
    elif name == "summarize_sentiment":
        return summarize_sentiment_tool(inputs)
    elif name == "get_recent_news":
        return get_recent_news_tool(inputs)
    raise ValueError(f"Unknown tool: {name}")

import boto3
client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

import json
from collections import defaultdict

def ask_claude(prompt):
    system_prompt = """You are a financial research assistant that uses tools to answer user questions about companies and stocks. Never use your memory or prior knowledge.

You do NOT always return the same types of data. Your response depends entirely on the user's query.

You must first categorize each user query into a known question type, and then call the correct tools accordingly. The logic for which tools to call and what to include is listed below.

Do not hallucinate any information. Only respond based on tools you call and their outputs.

Query Types and Expected Outputs:

1. “Tell me about Apple”
→ Call: get_stock_chart, get_recent_news, summarize_sentiment, get_fundamentals
→ Output: stock chart trend + chart file, recent headlines, sentiment summary, fundamentals summary

2. “How has Apple performed this month?”
→ Call: get_stock_chart (with period="1mo")
→ Output: chart file and brief description of trend

3. “What is the sentiment around Apple?”
→ Call: get_recent_news → summarize_sentiment
→ Output: stock trend (optional), headlines, sentiment tone

4. “Compare Apple to Nvidia”
→ Call: compare_stocks and get_fundamentals for both
→ Output: chart comparison and performance summary, fundamentals summary

5. “Why is Apple up today?”
→ Call: get_stock_chart (period="5d"), get_recent_news, summarize_sentiment
→ Output: chart + news-based reasoning

6. "Show Google's stock chart over the last 5 years"
→ Call: get_stock_chart (period='5y)
→ Output: Just chart and brief comment about the performance. Don't comment about a lack of information.

...

Only call the tools that are required for the user’s specific question. Use the returned values to generate a professional summary. Never hallucinate facts. Never use your own memory. Only rely on tools.

---"""

    messages = [
        {
            "role": "user",
            "content": [{"text": system_prompt.strip() + "\n\nUser query:\n" + prompt}]
        }
    ]

    tool_log = set()
    tool_results = {}
    tool_order = []

    MAX_TURNS = 10
    for _ in range(MAX_TURNS):
        res = client.converse(
            modelId=model_id,
            messages=messages,
            toolConfig=tool_config
        )

        output_msg = res["output"]["message"]
        messages.append(output_msg)

        tool_uses = [b["toolUse"] for b in output_msg["content"] if "toolUse" in b]
        if not tool_uses:
            break

        for tool_use in tool_uses:
            tool_name = tool_use["name"]
            tool_input = tool_use["input"]

            tool_input_key = json.dumps(tool_input, sort_keys=True).lower()
            tool_key = (tool_name, tool_input_key)

            if tool_key in tool_log:
                print(f"🛑 TOOL LOOP: {tool_name} with same input. Skipping.")
                continue

            tool_log.add(tool_key)

            print(f"🔧 Tool Called: {tool_name} with input {tool_input}")
            try:
                result_data = run_tool(tool_name, tool_input)
            except Exception as e:
                result_data = {"error": str(e)}

            print(f"📦 Tool Returned: {result_data}")

            if tool_name == "get_fundamentals" and "text" in result_data:
                summary = result_data["text"]
                result_data = {"text": summary}
                ticker = tool_input.get("ticker", "UNKNOWN")
                tool_results[f"get_fundamentals_{ticker}"] = {"text": summary}
            else:
                tool_results[tool_name] = result_data

            tool_result_msg = {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": result_data}],
                        "status": "success"
                    }
                }]
            }
            messages.append(tool_result_msg)

            tool_order.append(
                f"{tool_name}_{tool_input.get('ticker', '')}" if "ticker" in tool_input else tool_name
            )

    # ✅ Build result sections
    sections = []

    if "get_stock_chart" in tool_results:
        s = tool_results["get_stock_chart"]
        if s.get("trend_summary"):
            sections.append(s["trend_summary"])
        if s.get("chart_file"):
            sections.append(f"See attached chart: {s['chart_file']}")

    if "summarize_sentiment" in tool_results:
        s = tool_results["summarize_sentiment"]
        if s.get("importance"):
            sections.append(s["importance"])
        headlines = s.get("headlines_used", [])
        if headlines:
            sections.append("Recent Headlines:\n" + "\n".join(f"- {h}" for h in headlines))

    for k in tool_order:
        if k.startswith("get_fundamentals") and k in tool_results:
            summary = tool_results[k].get("text")
            if isinstance(summary, str) and summary.strip():
                sections.append(summary.strip())

    if "compare_stocks" in tool_results:
        s = tool_results["compare_stocks"]
        if s.get("trend_summary"):
            sections.append(s["trend_summary"])
        if s.get("chart_file"):
            sections.append(f"See attached comparison chart: {s['chart_file']}")

    # ✅ Construct one final user message instead of multiple role messages
    summary_prompt = (
        "Please summarize the following:\n\n" +
        "\n\n".join(f"- {s}" for s in sections if isinstance(s, str) and s.strip()) +
        "\n\n" +
        ("Summarize this company's stock performance and financials." if len([k for k in tool_results if k.startswith("get_fundamentals")]) <= 1
         else "Compare the companies mentioned. Discuss their performance and fundamentals.")
    )

    final_response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": summary_prompt.strip()}]
            }
        ]
    )

    response_text = next(
        (b["text"] for b in final_response["output"]["message"]["content"] if "text" in b),
        "No response generated."
    )

    return response_text, tool_results





st.set_page_config(page_title="FinanceBot", layout="wide")
st.title("FinanceBot: Stock Comparison Tool")

query = st.text_input("Enter your question (e.g., 'Compare NVDA and TSLA')")

if st.button("Run Analysis") and query:
    with st.spinner("Running financial analysis..."):
        response, tool_results = ask_claude(query)
        response = response.replace("$", "\\$")
        st.markdown("### Claude Response")
        st.markdown(response)

        # Show chart if mentioned in response
        import re
        import os

        # Also check tool output for attached chart files
        chart_file_keys = [key for key in ["get_stock_chart", "compare_stocks"] if key in tool_results]
        for key in chart_file_keys:
            chart_info = tool_results[key]
            chart_path = chart_info.get("chart_file")
            if chart_path and os.path.exists(chart_path):
                st.image(chart_path, caption=f"Chart: {chart_path}", use_container_width=True)



