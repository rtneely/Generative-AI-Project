import streamlit as st
import json
import boto3

from dotenv import load_dotenv
import os
import ast
load_dotenv()
from datetime import datetime, timedelta, timezone
import yfinance as yf
import mplfinance as mpf

def get_stock_chart(ticker, period='6mo', interval='1d'):
    import yfinance as yf
    import mplfinance as mpf
    import pandas as pd
    import os

    try:
        # Normalize period input
        period_map = {
            "1D": "1d", "5D": "5d", "1W": "5d", "1MO": "1mo", "3MO": "3mo",
            "6M": "6mo", "6MO": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y",
            "10Y": "10y", "YTD": "ytd", "MAX": "max"
        }
        period = period_map.get(period.upper(), period.lower())

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)

        # Retry with 1wk if empty
        if hist.empty or len(hist) < 2:
            print(f"No data for {ticker} with {period}/{interval}. Retrying with 1wk interval.")
            hist = stock.history(period=period, interval='1wk')

        # Check again if still unusable
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if hist.empty or not all(col in hist.columns for col in required_cols) or len(hist) < 2:
            raise ValueError("Insufficient historical data after retry")

        # Compute trend
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        pct_change = round((end_price - start_price) / start_price * 100, 2)
        trend = "increased" if pct_change > 0 else "decreased" if pct_change < 0 else "remained stable"
        trend_summary = f"{ticker.upper()} stock has {trend} by {abs(pct_change)}% over the past {period}."

        # Moving averages
        hist['MA10'] = hist['Close'].rolling(10).mean()
        hist['MA50'] = hist['Close'].rolling(50).mean()

        apds = []
        if hist['MA10'].notna().sum() >= 3:
            apds.append(mpf.make_addplot(hist['MA10'], color='orange', width=1))
        if hist['MA50'].notna().sum() >= 3:
            apds.append(mpf.make_addplot(hist['MA50'], color='blue', width=1))

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
            panel_ratios=(5, 1),
            figratio=(16, 9),
            figscale=1.5,
            savefig=filename
        )

        return {
            "chart_file": filename,
            "trend_summary": trend_summary
        }

    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")

        # Fallback using .info
        try:
            stock_info = stock.info
            price = stock_info.get("regularMarketPrice")
            prev = stock_info.get("regularMarketPreviousClose")
            if price and prev:
                change = round((price - prev) / prev * 100, 2)
                trend = "increased" if change > 0 else "decreased" if change < 0 else "remained stable"
                trend_summary = f"{ticker.upper()} stock has {trend} by {abs(change)}% today (based on fallback price info)."
            else:
                trend_summary = f"Could not retrieve price change for {ticker}."
        except:
            trend_summary = f"Could not retrieve price data for {ticker}."

        return {
            "chart_file": None,
            "trend_summary": trend_summary + " (Chart could not be generated.)"
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

Don't ever mention not having certain information. Example: If you don't use the get_fundamentals tool, don't say that there was no financial information provided.

Query Types and Expected Outputs:

1. “Tell me about Apple”
→ Call: get_stock_chart, get_recent_news, summarize_sentiment, get_fundamentals
→ Output: stock chart trend + chart file, recent headlines, sentiment summary, fundamentals summary

2. “How has Apple performed this month?”
→ Call: get_stock_chart (with period="1mo")
→ Output: chart file and brief description of trend. Do not mention fundamentals.

3. “What is the sentiment around Apple?”
→ Call: get_recent_news → summarize_sentiment
→ Output: stock trend (optional), headlines, sentiment tone

4. “Compare Apple to Nvidia” or any question comparing 2 to 5 companies
→ Call: compare_stocks and get_fundamentals for each company mentioned
→ Output: chart comparison and performance summary, fundamentals summary for each company

5. “Why is Apple up today?”
→ Call: get_stock_chart (period="5d"), get_recent_news, summarize_sentiment
→ Output: chart + news-based reasoning. Don't comment about lack of information. Don't mention fundamentals.

6. "Show Google's stock chart over the last 5 years"
→ Call: get_stock_chart (period='5y)
→ Output: Just show the stock chart. Don't comment about a lack of information or lack of chart. Do not mention fundamentals.

...

Only call the tools that are required for the user’s specific question. Use the returned values to generate a professional summary. Never hallucinate facts. Never use your own memory. Only rely on tools.

For compare questions involving more than 2 companies, call get_fundamentals for each company, unless otherwise instructed.

Only summarize the tools that were actually called and returned results. Do not speculate about additional financials, news, or analysis unless those tools were used.

Do not say things like "you did not provide enough information" — the system already determines what tools to run. If a tool result is brief (e.g., just a trend summary from get_stock_chart), treat that as the complete information for the answer.

Never recommend accessing additional data. Never mention other tools that were not used. The assistant’s role is to summarize tool outputs only — not to request more data or critique what's missing.

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

        # ✅ FIX: Insert dummy assistant message if toolUses are present
        if "tool_use" in output_msg:
            tool_calls = output_msg["tool_use"]
            messages.append({
                "role": "assistant",
                "content": [{"text": "Calling tools..."}],  # Avoid blank content error
                "toolCalls": tool_calls
            })


        tool_uses = [b["toolUse"] for b in output_msg["content"]
                     if isinstance(b, dict) and "toolUse" in b and "toolUseId" in b["toolUse"]]

        if not tool_uses:
            break

        tool_result_messages = []

        for tool_use in tool_uses:
            tool_name = tool_use.get("name")
            tool_input = tool_use.get("input")
            tool_use_id = tool_use.get("toolUseId")

            if not tool_use_id or not tool_name or not tool_input:
                print(f"⚠️ Skipping malformed tool_use: {tool_use}")
                continue

            # 🛠️ Fix compare_stocks malformed input
            if tool_name == "compare_stocks":
                tickers = tool_input.get("tickers")
                
                # Fix malformed string representation of list
                if isinstance(tickers, list) and len(tickers) == 1 and isinstance(tickers[0], str):
                    try:
                        parsed = ast.literal_eval(tickers[0])
                        if isinstance(parsed, list):
                            tool_input["tickers"] = [t.strip() for t in parsed]
                    except Exception as e:
                        print(f"❌ Failed to parse tickers list: {tickers} → {e}")
                        continue

                # Fallback: if it’s a comma-separated string
                elif isinstance(tickers, str):
                    tool_input["tickers"] = [t.strip() for t in tickers.split(",")]


            # 🚀 Avoid repeated tool calls
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

            # Safely extract ticker for tools that have it
            if tool_name in ["get_fundamentals", "get_recent_news", "summarize_sentiment"]:
                ticker = tool_input.get("ticker", "").upper()
            else:
                ticker = None

            # Save tool result
            if tool_name == "get_fundamentals" and "text" in result_data and ticker:
                tool_results[f"get_fundamentals_{ticker}"] = result_data
            elif tool_name == "get_recent_news" and "headlines" in result_data and ticker:
                tool_results[f"get_recent_news_{ticker}"] = result_data
            elif tool_name == "summarize_sentiment" and "importance" in result_data and ticker:
                tool_results[f"summarize_sentiment_{ticker}"] = result_data
            else:
                tool_results[tool_name] = result_data


            # ✅ Always return a toolResult to Claude using the correct toolUseId
            tool_result_messages.append({
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"json": result_data}],
                    "status": "success"
                }
            })
            print(f"📨 Appending toolResult for {tool_name} ID: {tool_use_id}")

        if tool_result_messages:
            messages.append({
                "role": "user",
                "content": tool_result_messages
            })



    # ✅ Build result sections
    sections = []

    if "get_stock_chart" in tool_results:
        s = tool_results["get_stock_chart"]
        if s.get("trend_summary"):
            sections.append(s["trend_summary"])

    # Don't mention chart file unless you're passing it to Claude — which you aren't
    # If needed for display in Streamlit, handle it separately in the UI


    for k, s in tool_results.items():
        if k.startswith("summarize_sentiment") and isinstance(s, dict):
            if s.get("importance"):
                sections.append(s["importance"])
            headlines = s.get("headlines_used", [])
            if headlines:
                sections.append("Recent Headlines:\n" + "\n".join(f"- {h}" for h in headlines))


    for k, v in tool_results.items():
        if k.startswith("get_fundamentals"):
            summary = v.get("text")
            if isinstance(summary, str) and summary.strip():
                sections.append(summary.strip())


    if "compare_stocks" in tool_results:
        s = tool_results["compare_stocks"]
        if s.get("trend_summary"):
            sections.append(s["trend_summary"])
        if s.get("chart_file"):
            sections.append(f"See attached comparison chart: {s['chart_file']}")

    # ✅ Construct one final user message instead of multiple role messages
    # 🔍 Determine if we should summarize or compare based on available fundamentals
    # Determine strategy prompt
    fundamental_keys = [k for k in tool_results if k.startswith("get_fundamentals")]
    actual_fundamentals = [
        k for k in fundamental_keys if tool_results.get(k) and "text" in tool_results[k]
    ]

    if len(actual_fundamentals) > 1:
        strategy_prompt = "Compare the companies mentioned. Discuss their performance and fundamentals."
    elif len(actual_fundamentals) == 1:
        strategy_prompt = "Summarize the company's stock performance and financials based on the available results."
    else:
        strategy_prompt = "Summarize the company's stock performance using only the available details below. Do not comment on what's missing."

    # Build section list
    valid_sections = [s for s in sections if isinstance(s, str) and s.strip()]
    joined_sections = "\n\n".join(f"- {s.strip()}" for s in valid_sections if s.strip())


    # Handle short fallback cases
    if not valid_sections:
        summary_prompt = "There are no valid results to summarize."
    elif len(valid_sections) == 1 and "trend" in valid_sections[0].lower():
        summary_prompt = f"Return this fact clearly and professionally:\n\n{valid_sections[0]}"
    else:
        summary_prompt = f"Please summarize the following:\n\n{joined_sections}\n\n{strategy_prompt}"

    # Final safety check
    # If summary_prompt ends up blank or malformed, patch it safely
    if not summary_prompt.strip() or all(line.strip() == "-" for line in joined_sections.splitlines()):
        summary_prompt = "Summarize the companies' stock performance and fundamentals based on the available outputs."


    final_response = client.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": [{"text": summary_prompt.strip()}]
        }]
    )


    response_text = next(
        (b["text"] for b in final_response["output"]["message"]["content"] if "text" in b),
        "No response generated."
    )

    return response_text, tool_results



st.set_page_config(page_title="Finance Agent", layout="wide")
st.title("Stock Information and Comparison Agent")


st.markdown("#### 🤖 What this app can do")
st.markdown("""
This tool uses real financial data and AI to answer questions like:
- *"Compare AAPL and NVDA over the past year"*
- *"Tell me about Tesla"*
- *"How is Amazon performing this month?"*
- *"What’s the sentiment around Microsoft?"*
- *"Show me how Apple has done in the last five days?"*

Simply type your question below and hit **Run Analysis**.

---

*Disclaimer: This app is for educational purposes only and does not constitute financial advice.*
""")

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



