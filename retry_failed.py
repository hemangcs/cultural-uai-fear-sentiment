#!/usr/bin/env python3
"""Retries downloading stocks that failed with HTTP 429 rate limiting."""

import requests
import pandas as pd
import json
import os
import time
import glob
from io import StringIO
from datetime import datetime

API_KEY = "cus_B65QiWcxxfYlar"
BASE_URL = "https://dataapi.marketpsych.com/rma/v4"
ASSET_CLASS = "equamer"
FREQ = "dai"
START_DATE = "2016-02-01"  # Last 10 years only for retries
END_DATE = "2026-02-01"
BASE_DIR = "/Volumes/TOSHIBA EXT/refinitiv-Data/IntnlData"


def download_stock(asset_code, ticker, name, exchange_label):
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
            output_dir = os.path.join(BASE_DIR, exchange_label)
            safe_ticker = ticker.replace("/", "_").replace(".", "_")
            output_file = os.path.join(output_dir, f"{safe_ticker}_sentiment.csv")
            df.to_csv(output_file, index=False)
            return "SUCCESS", len(df)
        else:
            return "NO_DATA", 0

    except requests.exceptions.HTTPError as e:
        return f"HTTP_{e.response.status_code}", 0
    except Exception as e:
        return "ERROR", 0


def main():
    # Collect all failed stocks from summaries
    failed = []
    for summary_file in sorted(glob.glob(os.path.join(BASE_DIR, "*", "_download_summary.json"))):
        with open(summary_file) as f:
            data = json.load(f)
        exchange = data["exchange"]
        for r in data["results"]:
            if r["status"].startswith("HTTP_429"):
                failed.append({
                    "exchange": exchange,
                    "ticker": r["ticker"],
                    "name": r["name"],
                    "asset_code": r["asset_code"]
                })

    print(f"Retrying {len(failed)} rate-limited stocks...")
    print(f"{'Ticker':<15s} | {'Exchange':<20s} | {'Status':<10s} | {'Records':>8s}")
    print("-" * 65)

    success_count = 0
    total_records = 0
    retry_results = {}

    for i, stock in enumerate(failed):
        ticker = stock["ticker"]
        exchange = stock["exchange"]
        asset_code = stock["asset_code"]
        name = stock["name"]

        status, records = download_stock(asset_code, ticker, name, exchange)
        print(f"{ticker:<15s} | {exchange:<20s} | {status:<10s} | {records:>8d}")

        if status == "SUCCESS":
            success_count += 1
            total_records += records

        # Track results for summary updates
        if exchange not in retry_results:
            retry_results[exchange] = []
        retry_results[exchange].append({
            "ticker": ticker,
            "name": name,
            "asset_code": asset_code,
            "status": status,
            "records": records,
            "file": os.path.join(BASE_DIR, exchange, f"{ticker.replace('/', '_').replace('.', '_')}_sentiment.csv") if status == "SUCCESS" else None
        })

        # Longer delay between requests to avoid rate limiting
        if i < len(failed) - 1:
            time.sleep(2.0)

    # Update summary files
    for exchange, results in retry_results.items():
        summary_file = os.path.join(BASE_DIR, exchange, "_download_summary.json")
        with open(summary_file) as f:
            summary = json.load(f)

        # Update results for retried stocks
        for retry_r in results:
            for j, orig_r in enumerate(summary["results"]):
                if orig_r["asset_code"] == retry_r["asset_code"]:
                    summary["results"][j]["status"] = retry_r["status"]
                    summary["results"][j]["records"] = retry_r["records"]
                    summary["results"][j]["file"] = retry_r["file"]
                    break

        # Recalculate totals
        summary["successful"] = sum(1 for r in summary["results"] if r["status"] == "SUCCESS")
        summary["total_records"] = sum(r["records"] for r in summary["results"])
        summary["retry_time"] = datetime.now().isoformat()

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    print()
    print(f"{'='*65}")
    print(f"RETRY SUMMARY:")
    print(f"  Retried: {len(failed)} stocks")
    print(f"  Success: {success_count}/{len(failed)}")
    print(f"  Total new records: {total_records:,}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
