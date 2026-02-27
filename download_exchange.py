#!/usr/bin/env python3
"""
Downloads MarketPsych sentiment data for a given exchange's stocks.
Usage: python3 download_exchange.py <exchange_label>
Example: python3 download_exchange.py US_NASDAQ_GS
"""

import requests
import pandas as pd
import json
import sys
import os
import time
from io import StringIO
from datetime import datetime

API_KEY = "cus_B65QiWcxxfYlar"
BASE_URL = "https://dataapi.marketpsych.com/rma/v4"
ASSET_CLASS = "equamer"
FREQ = "dai"
START_DATE = "1998-01-01"
END_DATE = "2026-02-01"
BASE_DIR = "/Volumes/TOSHIBA EXT/refinitiv-Data/IntnlData"


def download_stock(asset_code, ticker, name, exchange_label):
    """Download sentiment data for a single stock."""
    url = f"{BASE_URL}/data/{ASSET_CLASS}/{FREQ}/{asset_code}"
    params = {
        "apikey": API_KEY,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "format": "csv"
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        if len(df) > 0:
            # Save to exchange folder
            output_dir = os.path.join(BASE_DIR, exchange_label)
            safe_ticker = ticker.replace("/", "_").replace(".", "_")
            output_file = os.path.join(output_dir, f"{safe_ticker}_sentiment.csv")
            df.to_csv(output_file, index=False)

            return {
                "ticker": ticker,
                "name": name,
                "asset_code": asset_code,
                "status": "SUCCESS",
                "records": len(df),
                "file": output_file
            }
        else:
            return {
                "ticker": ticker,
                "name": name,
                "asset_code": asset_code,
                "status": "NO_DATA",
                "records": 0,
                "file": None
            }

    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.response.text[:100]
        except:
            pass
        return {
            "ticker": ticker,
            "name": name,
            "asset_code": asset_code,
            "status": f"HTTP_{e.response.status_code}",
            "records": 0,
            "error": error_detail,
            "file": None
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "name": name,
            "asset_code": asset_code,
            "status": "ERROR",
            "records": 0,
            "error": str(e)[:100],
            "file": None
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 download_exchange.py <exchange_label>")
        sys.exit(1)

    exchange_label = sys.argv[1]

    # Load selection
    selection_file = "/Volumes/TOSHIBA EXT/refinitiv-Data/Downloads/final_exchange_selection.json"
    with open(selection_file, "r") as f:
        selection = json.load(f)

    if exchange_label not in selection:
        print(f"ERROR: Exchange '{exchange_label}' not found in selection")
        sys.exit(1)

    stocks = selection[exchange_label]

    print(f"{'='*60}")
    print(f"Exchange: {exchange_label}")
    print(f"Stocks to download: {len(stocks)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"{'='*60}")
    print()
    print(f"{'Ticker':<12s} | {'Status':<10s} | {'Records':>8s} | Name")
    print("-" * 60)

    results = []
    success_count = 0
    total_records = 0

    for stock in stocks:
        asset_code = str(stock["assetCode"])
        ticker = stock["Ticker"]
        name = stock["name"]

        result = download_stock(asset_code, ticker, name, exchange_label)
        results.append(result)

        status = result["status"]
        records = result["records"]
        print(f"{ticker:<12s} | {status:<10s} | {records:>8d} | {name[:35]}")

        if status == "SUCCESS":
            success_count += 1
            total_records += records

        time.sleep(0.3)  # Rate limiting

    # Save summary
    output_dir = os.path.join(BASE_DIR, exchange_label)
    summary_file = os.path.join(output_dir, "_download_summary.json")
    summary = {
        "exchange": exchange_label,
        "download_time": datetime.now().isoformat(),
        "total_stocks": len(stocks),
        "successful": success_count,
        "total_records": total_records,
        "date_range": f"{START_DATE} to {END_DATE}",
        "results": results
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"{'='*60}")
    print(f"SUMMARY for {exchange_label}:")
    print(f"  Success: {success_count}/{len(stocks)} stocks")
    print(f"  Total records: {total_records:,}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
