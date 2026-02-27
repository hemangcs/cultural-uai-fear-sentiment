#!/usr/bin/env python3
"""
Script to download S&P 500 Technology Sector stock data using MarketPsych/LSEG Analytics API
Based on API credentials from Richard Peterson's email (Feb 5, 2026)
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

# ==========================
# API CONFIGURATION
# ==========================

# MarketPsych/LSEG Analytics API Credentials
API_KEY = "cus_B65QiWcxxfYlar"
USERNAME = "floridaintuni"
PASSWORD = "fiu&nc23$px49%xq17"

# API Endpoints
BASE_URL = "https://dataapi.marketpsych.com"
API_BASE = f"{BASE_URL}/rma/v4"  # RMA (Risk Management Analytics) not ESG
WEB_APP_URL = "https://www.marketpsych.com/app"

# ==========================
# S&P 500 TECHNOLOGY STOCKS
# ==========================
# Based on S&P 500 Information Technology Sector (as of February 2026)
# The Information Technology sector comprises ~70 stocks and represents 33.4% of S&P 500

SP500_TECH_STOCKS = [
    # Top Technology Stocks by Market Cap
    "NVDA",  # Nvidia Corp
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corp
    "AVGO",  # Broadcom Inc

    # Other Major Tech Companies
    "ORCL",  # Oracle Corporation
    "CSCO",  # Cisco Systems
    "ADBE",  # Adobe Inc.
    "CRM",   # Salesforce Inc.
    "INTC",  # Intel Corporation
    "AMD",   # Advanced Micro Devices
    "QCOM",  # Qualcomm Inc.
    "TXN",   # Texas Instruments
    "INTU",  # Intuit Inc.
    "AMAT",  # Applied Materials
    "MU",    # Micron Technology
    "LRCX",  # Lam Research
    "KLAC",  # KLA Corporation
    "SNPS",  # Synopsys Inc.
    "CDNS",  # Cadence Design Systems
    "PANW",  # Palo Alto Networks
    "ADSK",  # Autodesk Inc.
    "ADI",   # Analog Devices
    "NXPI",  # NXP Semiconductors
    "MCHP",  # Microchip Technology
    "FTNT",  # Fortinet Inc.
    "ANET",  # Arista Networks
    "CRWD",  # CrowdStrike Holdings
    "NOW",   # ServiceNow Inc.
    "PLTR",  # Palantir Technologies
    "SNOW",  # Snowflake Inc.
    "DDOG",  # Datadog Inc.
    "ZS",    # Zscaler Inc.
    "NET",   # Cloudflare Inc.
    "TEAM",  # Atlassian Corporation
    "WDAY",  # Workday Inc.
    "ANSS",  # ANSYS Inc.
    "ROP",   # Roper Technologies
    "KEYS",  # Keysight Technologies
    "MPWR",  # Monolithic Power Systems
    "ENPH",  # Enphase Energy (sometimes classified as Tech)
    "GDDY",  # GoDaddy Inc.
    "IT",    # Gartner Inc.
    "GLW",   # Corning Inc.
    "APH",   # Amphenol Corporation
    "TEL",   # TE Connectivity
    "ZBRA",  # Zebra Technologies
    "NTAP",  # NetApp Inc.
    "STX",   # Seagate Technology
    "WDC",   # Western Digital
    "HPQ",   # HP Inc.
    "DELL",  # Dell Technologies
    "HPE",   # Hewlett Packard Enterprise
    "FFIV",  # F5 Inc.
    "JNPR",  # Juniper Networks
    "AKAM",  # Akamai Technologies
    "CTSH",  # Cognizant Technology Solutions
    "EPAM",  # EPAM Systems
    "GEN",   # Gen Digital Inc.
    "VRSN",  # VeriSign Inc.
    "JKHY",  # Jack Henry & Associates
    "SWKS",  # Skyworks Solutions
    "QRVO",  # Qorvo Inc.
    "ON",    # ON Semiconductor
    "SMCI",  # Super Micro Computer
    "TER",   # Teradyne Inc.
    "TRMB",  # Trimble Inc.
    "FICO",  # Fair Isaac Corporation
    "TYL",   # Tyler Technologies
    "PTC",   # PTC Inc.
    "TTWO",  # Take-Two Interactive (sometimes classified as Tech)
    "EA",    # Electronic Arts (sometimes classified as Tech)
]

# Note: Some major "tech" companies are in other sectors:
# - AMZN, TSLA: Consumer Discretionary
# - GOOGL, GOOG, META: Communication Services


# ==========================
# API FUNCTIONS
# ==========================

class MarketPsychAPI:
    """
    MarketPsych/LSEG Analytics API Client (RMA - Risk Management Analytics)
    Documentation: https://dataapi.marketpsych.com/rma/v4
    """

    def __init__(self, api_key: str, username: str, password: str):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.base_url = "https://dataapi.marketpsych.com/rma/v4"
        self.asset_class = "equamer"  # Companies (Americas region)
        self.freq = "dai"  # Daily frequency
        self.ticker_to_assetcode = {}  # Will be populated from API

    def login(self) -> bool:
        """
        Authenticate with the MarketPsych API and load asset mappings
        Note: Pull API uses API key in query params, no login required
        """
        try:
            print(f"✓ API initialized with key: {self.api_key[:15]}...")

            # Load asset mappings (ticker to assetCode)
            print("Loading asset mappings from API...")
            url = f"{self.base_url}/{self.asset_class}/assets"
            params = {"apikey": self.api_key}

            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            assets_data = response.json()
            if "data" in assets_data:
                # Create ticker to assetCode mapping
                for asset in assets_data["data"]:
                    ticker = asset.get("Ticker")
                    asset_code = asset.get("assetCode")
                    if ticker and asset_code:
                        self.ticker_to_assetcode[ticker.upper()] = str(asset_code)

                print(f"✓ Loaded {len(self.ticker_to_assetcode)} ticker-to-assetCode mappings")
            else:
                print("⚠ Warning: No asset data received, will attempt ticker lookups")

            return True
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            return False

    def get_company_data(self, ticker: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Fetch company sentiment/analytics data for a specific ticker using Pull API

        Parameters:
        - ticker: Stock ticker symbol
        - start_date: Start date in YYYY-MM-DD format (inclusive)
        - end_date: End date in YYYY-MM-DD format (exclusive)

        Returns:
        - Dictionary containing status and data
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Get asset code for ticker
        asset_code = self.ticker_to_assetcode.get(ticker.upper())
        if not asset_code:
            print(f"✗ {ticker:6s} | NOT MAPPED | Ticker not found in asset mappings")
            return {
                "ticker": ticker,
                "status": "not_mapped",
                "error": "Ticker not found in asset mappings",
                "records": 0,
                "data": []
            }

        try:
            # Pull API endpoint: /data/<asset_class>/<freq>/<asset_code>
            url = f"{self.base_url}/data/{self.asset_class}/{self.freq}/{asset_code}"
            params = {
                "apikey": self.api_key,
                "start_date": start_date,
                "end_date": end_date,
                "format": "csv"  # CSV is more efficient for large datasets
            }

            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            # Parse CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            # Check if data was returned
            if len(df) > 0:
                print(f"✓ {ticker:6s} | SUCCESS | {len(df):6d} records | {start_date} to {end_date}")
                return {
                    "ticker": ticker,
                    "asset_code": asset_code,
                    "status": "success",
                    "records": len(df),
                    "data": df.to_dict('records'),
                    "start_date": start_date,
                    "end_date": end_date
                }
            else:
                print(f"⚠ {ticker:6s} | NO DATA | 0 records")
                return {
                    "ticker": ticker,
                    "asset_code": asset_code,
                    "status": "no_data",
                    "records": 0,
                    "data": [],
                    "start_date": start_date,
                    "end_date": end_date
                }

        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            error_detail = ""
            try:
                error_detail = e.response.text
            except:
                pass

            if e.response.status_code == 404:
                print(f"✗ {ticker:6s} | NOT FOUND | Asset not available in API")
            elif e.response.status_code == 401:
                print(f"✗ {ticker:6s} | AUTH ERROR | {error_detail[:100] if error_detail else 'Invalid API key'}")
            elif e.response.status_code == 403:
                print(f"✗ {ticker:6s} | FORBIDDEN | Not authorized for this asset")
            else:
                print(f"✗ {ticker:6s} | HTTP {e.response.status_code} | {str(e)}")
            return {
                "ticker": ticker,
                "status": "error",
                "error": str(e),
                "error_detail": error_detail,
                "records": 0,
                "data": []
            }
        except requests.exceptions.Timeout:
            print(f"✗ {ticker:6s} | TIMEOUT | Request took too long")
            return {
                "ticker": ticker,
                "status": "timeout",
                "error": "Request timeout",
                "records": 0,
                "data": []
            }
        except Exception as e:
            print(f"✗ {ticker:6s} | ERROR | {str(e)}")
            return {
                "ticker": ticker,
                "status": "error",
                "error": str(e),
                "records": 0,
                "data": []
            }

    def get_bulk_data(self, tickers: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch bulk data for multiple tickers with chunking for date range limits

        Parameters:
        - tickers: List of stock ticker symbols
        - start_date: Start date in YYYY-MM-DD format
        - end_date: End date in YYYY-MM-DD format

        Returns:
        - pandas DataFrame with combined data
        """
        all_data = []
        success_count = 0
        failed_count = 0
        no_data_count = 0

        # Calculate date chunks (15000 days max per query for daily data)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days

        chunk_size = 15000  # Max days per query for daily data
        date_chunks = []

        if total_days > chunk_size:
            current_date = start_dt
            while current_date < end_dt:
                chunk_end = min(current_date + timedelta(days=chunk_size), end_dt)
                date_chunks.append((current_date.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
                current_date = chunk_end
        else:
            date_chunks = [(start_date, end_date)]

        print(f"\nDate range split into {len(date_chunks)} chunk(s) for processing...")
        print(f"Total date range: {total_days} days\n")

        for ticker in tickers:
            ticker_data = []

            for chunk_start, chunk_end in date_chunks:
                result = self.get_company_data(ticker, chunk_start, chunk_end)

                if result["status"] == "success":
                    ticker_data.extend(result["data"])

                time.sleep(0.2)  # Rate limiting between chunks

            # Aggregate results
            if ticker_data:
                success_count += 1
                for record in ticker_data:
                    record["ticker"] = ticker
                    all_data.append(record)
            elif result["status"] == "no_data":
                no_data_count += 1
            else:
                failed_count += 1

            time.sleep(0.1)  # Rate limiting between tickers

        print(f"\n{'='*60}")
        print(f"Download Summary:")
        print(f"  ✓ Success: {success_count}/{len(tickers)} stocks")
        print(f"  ⚠ No data: {no_data_count}/{len(tickers)} stocks")
        print(f"  ✗ Failed:  {failed_count}/{len(tickers)} stocks")
        print(f"  📊 Total records: {len(all_data)}")
        print(f"{'='*60}\n")

        if all_data:
            df = pd.DataFrame(all_data)
            return df
        else:
            return pd.DataFrame()


# ==========================
# MAIN EXECUTION
# ==========================

def main():
    """
    Main function to download S&P 500 Technology sector stock data
    """
    print("="*60)
    print("S&P 500 Technology Sector Data Downloader")
    print("MarketPsych/LSEG Analytics Pull API")
    print("="*60)
    print()

    # Initialize API client
    print("Initializing API connection...")
    api = MarketPsychAPI(API_KEY, USERNAME, PASSWORD)

    # Login
    if not api.login():
        print("Failed to authenticate. Please check credentials.")
        return

    print()
    print(f"Number of Technology stocks to download: {len(SP500_TECH_STOCKS)}")
    print(f"Asset class: {api.asset_class}")
    print(f"Frequency: {api.freq} (daily)")
    print()

    # Set date range (1998-01-01 to 2026-02-01)
    start_date = "1998-01-01"
    end_date = "2026-02-01"

    # Calculate date range info
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    print(f"Date range: {start_date} to {end_date} ({total_days} days)")
    print()

    # Download data
    print("Downloading data stock-by-stock...")
    print("-" * 60)
    print(f"{'Ticker':<6s} | {'Status':<10s} | {'Records':<6s} | Details")
    print("-" * 60)

    df = api.get_bulk_data(SP500_TECH_STOCKS, start_date, end_date)

    if not df.empty:
        # Create Downloads folder if it doesn't exist
        import os
        downloads_dir = "Downloads"
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)
            print(f"✓ Created Downloads directory")

        # Save to CSV
        output_file = os.path.join(downloads_dir, f"sp500_tech_stocks_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print(f"✓ Data saved to: {output_file}")
        print(f"✓ Total records: {len(df):,}")
        print(f"✓ Unique stocks: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")
        print(f"✓ Data columns: {len(df.columns)}")
        print(f"{'='*60}\n")

        print("Sample data (first 5 rows):")
        print(df.head().to_string())
        print()

        if 'ticker' in df.columns:
            print("Records per stock:")
            stock_counts = df['ticker'].value_counts().sort_index()
            for ticker, count in stock_counts.items():
                print(f"  {ticker}: {count:,} records")
    else:
        print()
        print("✗ No data downloaded. Please check:")
        print("  1. API credentials are correct")
        print("  2. API endpoint URLs are correct")
        print("  3. Internet connection is working")
        print("  4. API subscription is active (expires May 5, 2026)")
        print("  5. Date range is within permissioned dates")
        print()
        print("For more information, visit:")
        print(f"  - Web App: {WEB_APP_URL}")
        print(f"  - API Docs: {API_BASE}")

    print()
    print("="*60)


if __name__ == "__main__":
    main()


# ==========================
# ADDITIONAL NOTES
# ==========================
"""
IMPORTANT NOTES:

1. API ENDPOINTS:
   The actual API endpoints need to be verified from the official documentation:
   https://www.marketpsych.com/esg4/apidocs/overview

   The attached Jupyter notebook (API_Access_LMA_Companies.ipynb) in the email
   contains working examples of API calls.

2. AUTHENTICATION:
   The authentication method used here is a placeholder. Check the API docs
   for the correct authentication flow (might use OAuth, JWT, or API key in headers).

3. DATA FIELDS:
   MarketPsych provides sentiment and analytics data. Available fields may include:
   - Sentiment scores
   - Buzz/volume metrics
   - ESG scores
   - News analytics
   - Social media metrics

   Refer to the API documentation for complete field descriptions.

4. RATE LIMITING:
   Be aware of API rate limits. Adjust the sleep time in get_bulk_data() if needed.

5. ALTERNATIVE DATA SOURCES:
   The email mentions you can also download data via:
   - Bulk Files: https://www.marketpsych.com/ma4/download/files
   - Coverage List: For mapping files

6. ADDING MORE STOCKS:
   To add more technology stocks, simply append ticker symbols to SP500_TECH_STOCKS list.

7. API EXPIRATION:
   API key expires: May 5, 2026
   Contact support@marketpsych.com for renewal or issues.

8. NEXT STEPS:
   - Review the Jupyter notebook attachment for working code examples
   - Test API endpoints with a single stock first
   - Consult with David Guarin (davidg@marketpsychdata.com) for technical support
   - Schedule follow-up call as recommended in the welcome email
"""
