#!/usr/bin/env python3
"""
RQ1 Analysis: Cultural Uncertainty Avoidance and Investor Fear Contagion
in AI/Emerging Technology Companies

Research Question: Does Hofstede's uncertainty avoidance (UAI) amplify
fear-sentiment spillovers among AI/quantum computing stocks during
technology disruption events?

Data Sources:
- MarketPsych/LSEG RMA daily sentiment (1998-2026) for 165 tech stocks
- Hofstede cultural dimensions (6D)
- Schwartz cultural value orientations (7D)
- GLOBE project dimensions (18 measures)
- Gelfand cultural tightness-looseness
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "/Volumes/TOSHIBA EXT/refinitiv-Data/IntnlData"
OUTPUT_DIR = "/Volumes/TOSHIBA EXT/refinitiv-Data/RQ1_Analysis"

# Country-exchange mapping
COUNTRY_EXCHANGE = {
    'AU_ASX': 'Australia',
    'BR_B3_BOVESPA': 'Brazil',
    'CA_TSX': 'Canada', 'CA_CSE': 'Canada', 'CA_TSX_V': 'Canada',
    'CL_BOLSA_SANTIAGO': 'Chile',
    'DE_XETRA': 'Germany',
    'GB_LSE_AIM': 'United Kingdom',
    'MX_BMV': 'Mexico',
    'TW_TPEx': 'Taiwan',
    'US_NASDAQ_GS': 'United States', 'US_NASDAQ_CM': 'United States',
    'US_NYSE': 'United States', 'US_NASDAQ_GM': 'United States',
    'US_NYSE_AMER': 'United States'
}

# Hofstede UAI scores
UAI_SCORES = {
    'Australia': 51, 'Brazil': 76, 'Canada': 48, 'Chile': 86,
    'Germany': 65, 'Mexico': 82, 'Taiwan': 69,
    'United Kingdom': 35, 'United States': 46
}

# Full Hofstede dimensions
HOFSTEDE = {
    'Australia':      {'pdi': 38, 'idv': 90, 'mas': 61, 'uai': 51, 'lto': 21, 'ivr': 71},
    'Brazil':         {'pdi': 69, 'idv': 38, 'mas': 49, 'uai': 76, 'lto': 44, 'ivr': 59},
    'Canada':         {'pdi': 39, 'idv': 80, 'mas': 52, 'uai': 48, 'lto': 36, 'ivr': 68},
    'Chile':          {'pdi': 63, 'idv': 23, 'mas': 28, 'uai': 86, 'lto': 31, 'ivr': 68},
    'Germany':        {'pdi': 35, 'idv': 67, 'mas': 66, 'uai': 65, 'lto': 83, 'ivr': 40},
    'Mexico':         {'pdi': 81, 'idv': 30, 'mas': 69, 'uai': 82, 'lto': 24, 'ivr': 97},
    'Taiwan':         {'pdi': 58, 'idv': 17, 'mas': 45, 'uai': 69, 'lto': 93, 'ivr': 49},
    'United Kingdom': {'pdi': 35, 'idv': 89, 'mas': 66, 'uai': 35, 'lto': 51, 'ivr': 69},
    'United States':  {'pdi': 40, 'idv': 91, 'mas': 62, 'uai': 46, 'lto': 26, 'ivr': 68},
}

# UAI groups for analysis
UAI_HIGH = ['Chile', 'Mexico', 'Brazil']        # UAI >= 76
UAI_MEDIUM = ['Taiwan', 'Germany']               # UAI 65-69
UAI_LOW = ['Australia', 'Canada', 'United States', 'United Kingdom']  # UAI <= 51

# Key AI disruption events
AI_EVENTS = {
    '2022-11-30': 'ChatGPT Launch',
    '2023-01-23': 'Microsoft-OpenAI $10B Investment',
    '2023-03-14': 'GPT-4 Release',
    '2023-05-25': 'NVIDIA Q1 2024 Earnings Surge',
    '2023-07-18': 'EU AI Act Vote',
    '2023-10-30': 'Biden AI Executive Order',
    '2024-01-25': 'DeepSeek R1 Announcement',
    '2024-03-04': 'Claude 3 Launch',
    '2024-06-18': 'NVIDIA Becomes Most Valuable Company',
    '2024-09-10': 'OpenAI o1 Release',
    '2025-01-20': 'DeepSeek V3 Disruption',
    '2025-01-27': 'DeepSeek R1 Market Crash',
}

# Sentiment columns of interest
FEAR_COLS = ['fear', 'uncertainty', 'stress', 'gloom', 'anger']
SENTIMENT_COLS = ['sentiment', 'positive', 'negative', 'optimism', 'pessimism',
                  'joy', 'trust', 'fear', 'uncertainty', 'stress', 'gloom',
                  'anger', 'surprise', 'disagreement', 'volatility',
                  'emotionVsFact', 'buzz', 'mentions']


def load_all_sentiment_data():
    """Load all sentiment CSVs and tag with country."""
    print("Loading sentiment data from all exchanges...")
    all_dfs = []

    for exchange_dir in sorted(glob.glob(os.path.join(BASE_DIR, "*"))):
        if not os.path.isdir(exchange_dir):
            continue
        exchange = os.path.basename(exchange_dir)
        if exchange not in COUNTRY_EXCHANGE:
            continue
        country = COUNTRY_EXCHANGE[exchange]

        csv_files = glob.glob(os.path.join(exchange_dir, "*_sentiment.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, parse_dates=['windowTimestamp'])
                df['exchange'] = exchange
                df['country'] = country
                df['uai'] = UAI_SCORES[country]

                # Add all Hofstede dimensions
                for dim, val in HOFSTEDE[country].items():
                    df[f'h_{dim}'] = val

                # UAI group
                if country in UAI_HIGH:
                    df['uai_group'] = 'High'
                elif country in UAI_MEDIUM:
                    df['uai_group'] = 'Medium'
                else:
                    df['uai_group'] = 'Low'

                all_dfs.append(df)
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['date'] = combined['windowTimestamp'].dt.date
    combined['date'] = pd.to_datetime(combined['date'])
    combined['year'] = combined['date'].dt.year
    combined['month'] = combined['date'].dt.to_period('M')

    print(f"  Total records: {len(combined):,}")
    print(f"  Unique tickers: {combined['ticker'].nunique()}")
    print(f"  Countries: {combined['country'].nunique()}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")

    return combined


def compute_fear_index(df):
    """Create composite fear index from multiple fear-related variables."""
    fear_components = []
    for col in FEAR_COLS:
        if col in df.columns:
            # Standardize each component within the full sample
            series = df[col].copy()
            if series.notna().sum() > 0:
                fear_components.append(series)

    if fear_components:
        # Simple average of available fear components
        fear_df = pd.DataFrame(fear_components).T
        df['fear_index'] = fear_df.mean(axis=1)
    else:
        df['fear_index'] = np.nan

    return df


def create_panel_dataset(df):
    """Create firm-day panel with country-level cultural variables."""
    print("\nCreating panel dataset...")

    # Compute fear index
    df = compute_fear_index(df)

    # Create firm identifier (ensure string types)
    df['ticker'] = df['ticker'].astype(str)
    df['exchange'] = df['exchange'].astype(str)
    df['firm_id'] = df['ticker'] + '_' + df['exchange']

    # Select relevant columns
    panel_cols = ['date', 'firm_id', 'ticker', 'exchange', 'country',
                  'uai', 'uai_group', 'h_pdi', 'h_idv', 'h_mas', 'h_uai',
                  'h_lto', 'h_ivr', 'year'] + SENTIMENT_COLS + ['fear_index']

    existing_cols = [c for c in panel_cols if c in df.columns]
    panel = df[existing_cols].copy()

    # Drop rows where key sentiment vars are all NaN
    panel = panel.dropna(subset=['sentiment'], how='all')

    print(f"  Panel observations: {len(panel):,}")
    print(f"  Firms: {panel['firm_id'].nunique()}")
    print(f"  Countries: {panel['country'].nunique()}")

    return panel


def descriptive_statistics(panel):
    """Generate descriptive statistics tables."""
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # Table 1: Country-level summary
    print("\nTable 1: Sample Distribution by Country and UAI Group")
    print("-" * 70)
    country_stats = panel.groupby(['country', 'uai', 'uai_group']).agg(
        firms=('firm_id', 'nunique'),
        observations=('sentiment', 'count'),
        avg_sentiment=('sentiment', 'mean'),
        avg_fear=('fear', 'mean'),
        avg_uncertainty=('uncertainty', 'mean'),
        avg_buzz=('buzz', 'mean')
    ).round(4)
    print(country_stats.to_string())

    # Table 2: Sentiment variable statistics by UAI group
    print("\n\nTable 2: Sentiment Variables by UAI Group")
    print("-" * 70)
    sent_vars = ['sentiment', 'fear', 'uncertainty', 'stress', 'gloom',
                 'fear_index', 'volatility', 'disagreement', 'buzz']
    existing_vars = [v for v in sent_vars if v in panel.columns]

    uai_stats = panel.groupby('uai_group')[existing_vars].agg(['mean', 'std', 'count'])
    print(uai_stats.to_string())

    # Save to CSV
    country_stats.to_csv(os.path.join(OUTPUT_DIR, 'table1_country_summary.csv'))
    uai_stats.to_csv(os.path.join(OUTPUT_DIR, 'table2_uai_group_stats.csv'))

    return country_stats, uai_stats


def correlation_analysis(panel):
    """Correlation between UAI and fear-related sentiment variables."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Country-level aggregation for correlation
    country_agg = panel.groupby('country').agg(
        uai=('uai', 'first'),
        mean_sentiment=('sentiment', 'mean'),
        mean_fear=('fear', 'mean'),
        mean_uncertainty=('uncertainty', 'mean'),
        mean_stress=('stress', 'mean'),
        mean_fear_index=('fear_index', 'mean'),
        std_sentiment=('sentiment', 'std'),
        std_fear=('fear', 'std'),
        mean_volatility=('volatility', 'mean'),
        mean_disagreement=('disagreement', 'mean'),
    ).dropna()

    print("\nCountry-Level Correlations with UAI:")
    for col in country_agg.columns:
        if col != 'uai' and country_agg[col].notna().sum() >= 3:
            corr = country_agg['uai'].corr(country_agg[col])
            print(f"  UAI x {col}: r = {corr:.4f}")

    country_agg.to_csv(os.path.join(OUTPUT_DIR, 'country_aggregates.csv'))
    return country_agg


def panel_regression(panel):
    """Run panel regressions: Fear sentiment ~ UAI + controls."""
    print("\n" + "=" * 70)
    print("PANEL REGRESSION ANALYSIS")
    print("=" * 70)

    from linearmodels.panel import PanelOLS, RandomEffects
    import statsmodels.api as sm

    # Prepare panel data
    reg_data = panel[['date', 'firm_id', 'country', 'uai', 'uai_group',
                       'h_pdi', 'h_idv', 'h_mas', 'h_lto', 'h_ivr',
                       'fear', 'uncertainty', 'stress', 'sentiment',
                       'fear_index', 'buzz', 'volatility', 'disagreement',
                       'emotionVsFact']].copy()

    reg_data = reg_data.dropna(subset=['fear_index', 'uai', 'buzz'])

    # Standardize continuous variables
    for col in ['fear_index', 'fear', 'uncertainty', 'stress', 'sentiment',
                'buzz', 'volatility', 'disagreement', 'emotionVsFact']:
        if col in reg_data.columns and reg_data[col].notna().sum() > 0:
            mean_val = reg_data[col].mean()
            std_val = reg_data[col].std()
            if std_val > 0:
                reg_data[f'{col}_z'] = (reg_data[col] - mean_val) / std_val

    # Standardize UAI
    uai_mean = reg_data['uai'].mean()
    uai_std = reg_data['uai'].std()
    reg_data['uai_z'] = (reg_data['uai'] - uai_mean) / uai_std

    # Create interaction term
    reg_data['uai_x_buzz'] = reg_data['uai_z'] * reg_data['buzz_z']

    # Create dummy variables for UAI groups
    reg_data['uai_high'] = (reg_data['uai_group'] == 'High').astype(int)
    reg_data['uai_medium'] = (reg_data['uai_group'] == 'Medium').astype(int)

    # ---- Model 1: OLS with clustered standard errors ----
    print("\nModel 1: OLS - Fear Index ~ UAI + Controls")
    y = reg_data['fear_index_z'].dropna()
    X_cols = ['uai_z', 'buzz_z']
    X_cols = [c for c in X_cols if c in reg_data.columns]
    X = reg_data.loc[y.index, X_cols]
    X = sm.add_constant(X).dropna()
    y = y.loc[X.index]

    model1 = sm.OLS(y, X).fit(cov_type='cluster',
                               cov_kwds={'groups': reg_data.loc[X.index, 'country']})
    print(model1.summary2().tables[1].to_string())

    # ---- Model 2: With UAI group dummies ----
    print("\nModel 2: OLS - Fear Index ~ UAI Groups + Controls")
    X_cols2 = ['uai_high', 'uai_medium', 'buzz_z']
    X_cols2 = [c for c in X_cols2 if c in reg_data.columns]
    X2 = reg_data.loc[y.index, X_cols2]
    X2 = sm.add_constant(X2).dropna()
    y2 = y.loc[X2.index]

    model2 = sm.OLS(y2, X2).fit(cov_type='cluster',
                                 cov_kwds={'groups': reg_data.loc[X2.index, 'country']})
    print(model2.summary2().tables[1].to_string())

    # ---- Model 3: With interaction term (UAI × Buzz) ----
    print("\nModel 3: OLS - Fear Index ~ UAI + Buzz + UAI×Buzz + Controls")
    X_cols3 = ['uai_z', 'buzz_z', 'uai_x_buzz']
    X3 = reg_data.loc[y.index, X_cols3]
    X3 = sm.add_constant(X3).dropna()
    y3 = y.loc[X3.index]

    model3 = sm.OLS(y3, X3).fit(cov_type='cluster',
                                 cov_kwds={'groups': reg_data.loc[X3.index, 'country']})
    print(model3.summary2().tables[1].to_string())

    # ---- Model 4: Full model with all Hofstede controls ----
    print("\nModel 4: Full Model - Fear Index ~ UAI + Other Hofstede + Buzz + Interaction")
    # Standardize other Hofstede dimensions
    for dim in ['h_pdi', 'h_idv', 'h_mas', 'h_lto', 'h_ivr']:
        m = reg_data[dim].mean()
        s = reg_data[dim].std()
        if s > 0:
            reg_data[f'{dim}_z'] = (reg_data[dim] - m) / s

    X_cols4 = ['uai_z', 'buzz_z', 'uai_x_buzz', 'h_pdi_z', 'h_idv_z', 'h_mas_z', 'h_lto_z', 'h_ivr_z']
    X_cols4 = [c for c in X_cols4 if c in reg_data.columns]
    X4 = reg_data.loc[y.index, X_cols4]
    X4 = sm.add_constant(X4).dropna()
    y4 = y.loc[X4.index]

    model4 = sm.OLS(y4, X4).fit(cov_type='cluster',
                                 cov_kwds={'groups': reg_data.loc[X4.index, 'country']})
    print(model4.summary2().tables[1].to_string())

    # Save regression results
    results = {
        'Model 1 (UAI only)': model1,
        'Model 2 (UAI groups)': model2,
        'Model 3 (Interaction)': model3,
        'Model 4 (Full)': model4
    }

    # Create consolidated results table
    print("\n\nCONSOLIDATED REGRESSION TABLE")
    print("=" * 90)
    print(f"{'Variable':<25s} {'Model 1':>12s} {'Model 2':>12s} {'Model 3':>12s} {'Model 4':>12s}")
    print("-" * 90)

    all_vars = set()
    for m in results.values():
        all_vars.update(m.params.index)

    for var in sorted(all_vars):
        row = f"{var:<25s}"
        for name, m in results.items():
            if var in m.params.index:
                coef = m.params[var]
                pval = m.pvalues[var]
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row += f" {coef:>8.4f}{stars:<3s}"
            else:
                row += f" {'':>12s}"
        print(row)

    print("-" * 90)
    for name, m in results.items():
        print(f"  {name}: R² = {m.rsquared:.4f}, N = {int(m.nobs):,}")

    return results


def event_study(panel):
    """Event study: Fear sentiment around AI disruption events."""
    print("\n" + "=" * 70)
    print("EVENT STUDY ANALYSIS")
    print("=" * 70)

    event_window = 10  # days before/after event

    results = []

    for event_date_str, event_name in AI_EVENTS.items():
        event_date = pd.to_datetime(event_date_str)

        # Check if we have data around this event
        window_start = event_date - pd.Timedelta(days=event_window + 5)
        window_end = event_date + pd.Timedelta(days=event_window + 5)

        event_data = panel[(panel['date'] >= window_start) &
                           (panel['date'] <= window_end)].copy()

        if len(event_data) < 50:
            continue

        # Calculate days relative to event
        event_data['event_day'] = (event_data['date'] - event_date).dt.days
        event_data = event_data[(event_data['event_day'] >= -event_window) &
                                 (event_data['event_day'] <= event_window)]

        if len(event_data) < 30:
            continue

        # Pre-event baseline (days -10 to -3)
        pre_event = event_data[event_data['event_day'].between(-event_window, -3)]

        # Post-event period (days 0 to +5)
        post_event = event_data[event_data['event_day'].between(0, 5)]

        for uai_group in ['High', 'Medium', 'Low']:
            pre = pre_event[pre_event['uai_group'] == uai_group]
            post = post_event[post_event['uai_group'] == uai_group]

            if len(pre) > 5 and len(post) > 5:
                for var in ['fear', 'uncertainty', 'stress', 'sentiment', 'fear_index']:
                    if var in pre.columns:
                        pre_mean = pre[var].mean()
                        post_mean = post[var].mean()
                        change = post_mean - pre_mean

                        # Abnormal fear = post - pre
                        results.append({
                            'event': event_name,
                            'event_date': event_date_str,
                            'uai_group': uai_group,
                            'variable': var,
                            'pre_mean': pre_mean,
                            'post_mean': post_mean,
                            'change': change,
                            'pre_n': pre[var].notna().sum(),
                            'post_n': post[var].notna().sum(),
                        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Display event study results
        print("\nAbnormal Fear Sentiment Around AI Disruption Events")
        print("-" * 80)

        # Focus on fear_index
        fear_results = results_df[results_df['variable'] == 'fear_index']

        if len(fear_results) > 0:
            pivot = fear_results.pivot_table(
                values='change',
                index=['event', 'event_date'],
                columns='uai_group',
                aggfunc='mean'
            )

            # Reorder columns
            col_order = [c for c in ['High', 'Medium', 'Low'] if c in pivot.columns]
            pivot = pivot[col_order]

            print(pivot.round(4).to_string())

            # Average across all events
            print("\n\nAverage Abnormal Fear Index Change by UAI Group:")
            avg_change = fear_results.groupby('uai_group')['change'].agg(['mean', 'std', 'count'])
            print(avg_change.round(4).to_string())

            # Statistical test: High UAI vs Low UAI
            from scipy import stats
            high_changes = fear_results[fear_results['uai_group'] == 'High']['change']
            low_changes = fear_results[fear_results['uai_group'] == 'Low']['change']

            if len(high_changes) > 2 and len(low_changes) > 2:
                t_stat, p_val = stats.ttest_ind(high_changes, low_changes)
                print(f"\n  t-test (High UAI vs Low UAI fear change): t={t_stat:.3f}, p={p_val:.4f}")

            pivot.to_csv(os.path.join(OUTPUT_DIR, 'event_study_results.csv'))

        # Also analyze sentiment
        sent_results = results_df[results_df['variable'] == 'sentiment']
        if len(sent_results) > 0:
            print("\n\nAbnormal Sentiment Change by UAI Group:")
            pivot_s = sent_results.pivot_table(
                values='change',
                index=['event', 'event_date'],
                columns='uai_group',
                aggfunc='mean'
            )
            col_order = [c for c in ['High', 'Medium', 'Low'] if c in pivot_s.columns]
            pivot_s = pivot_s[col_order]
            print(pivot_s.round(4).to_string())
            pivot_s.to_csv(os.path.join(OUTPUT_DIR, 'event_study_sentiment.csv'))

    results_df.to_csv(os.path.join(OUTPUT_DIR, 'event_study_full.csv'), index=False)
    return results_df


def time_series_analysis(panel):
    """Granger causality and fear volatility clustering by UAI group."""
    print("\n" + "=" * 70)
    print("TIME SERIES ANALYSIS: Fear Sentiment Volatility by UAI Group")
    print("=" * 70)

    # Create month column if not present
    panel['month'] = panel['date'].dt.to_period('M')

    # Monthly aggregation by UAI group
    monthly = panel.groupby(['month', 'uai_group']).agg(
        mean_fear=('fear', 'mean'),
        std_fear=('fear', 'std'),
        mean_uncertainty=('uncertainty', 'mean'),
        mean_stress=('stress', 'mean'),
        mean_sentiment=('sentiment', 'mean'),
        std_sentiment=('sentiment', 'std'),
        mean_fear_index=('fear_index', 'mean'),
        std_fear_index=('fear_index', 'std'),
        observations=('sentiment', 'count'),
    ).reset_index()

    monthly['month_dt'] = monthly['month'].dt.to_timestamp()

    # Focus on post-2020 period (AI era)
    ai_era = monthly[monthly['month_dt'] >= '2020-01-01']

    print("\nFear Sentiment Volatility (Std Dev) by UAI Group - AI Era (2020+):")
    for group in ['High', 'Medium', 'Low']:
        g_data = ai_era[ai_era['uai_group'] == group]
        if len(g_data) > 0:
            avg_std_fear = g_data['std_fear'].mean()
            avg_std_sent = g_data['std_sentiment'].mean()
            avg_mean_fear = g_data['mean_fear'].mean()
            print(f"  {group:<8s} UAI: Avg Fear={avg_mean_fear:.4f}, "
                  f"Fear Volatility={avg_std_fear:.4f}, "
                  f"Sentiment Volatility={avg_std_sent:.4f}")

    monthly.to_csv(os.path.join(OUTPUT_DIR, 'monthly_uai_timeseries.csv'), index=False)
    return monthly


def generate_visualizations(panel, monthly_ts, event_results):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
    })

    # ---- Figure 1: Fear Sentiment Over Time by UAI Group ----
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}

    # Use full date range and monthly averages with fear_index for better coverage
    # (raw 'fear' has only 4 obs for Medium/Taiwan; fear_index has 56)
    panel_ts = panel.copy()
    panel_ts['yearmonth'] = panel_ts['date'].dt.to_period('M')

    monthly_fig = panel_ts.groupby(['yearmonth', 'uai_group']).agg(
        mean_fear=('fear_index', 'mean'),
        obs=('fear_index', 'count'),
    ).reset_index()
    monthly_fig['month_dt'] = monthly_fig['yearmonth'].dt.to_timestamp()

    # Different min-obs thresholds: generous for Medium (Taiwan, sparse data)
    min_obs = {'High': 5, 'Medium': 1, 'Low': 5}

    for group in ['High', 'Medium', 'Low']:
        g = monthly_fig[(monthly_fig['uai_group'] == group) &
                        (monthly_fig['obs'] >= min_obs[group])].sort_values('month_dt')
        if len(g) > 0:
            g = g.copy()
            if group == 'Medium':
                # Sparse data: show individual points with connecting dashed line
                ax.scatter(g['month_dt'], g['mean_fear'],
                           color=colors[group], s=25, alpha=0.7, zorder=5,
                           label=f'{group} UAI - Taiwan (n={g["obs"].sum()} obs)')
                ax.plot(g['month_dt'], g['mean_fear'],
                        color=colors[group], linewidth=1.0, alpha=0.4, linestyle='--')
            else:
                # Dense data: rolling 3-month average for smoother lines
                g['smooth_fear'] = g['mean_fear'].rolling(window=3, min_periods=1, center=True).mean()
                ax.plot(g['month_dt'], g['smooth_fear'],
                        label=f'{group} UAI (n={g["obs"].sum():,} obs)',
                        color=colors[group], linewidth=1.5)

    # Add event markers with labels
    event_labels = {
        '2022-11-30': 'ChatGPT', '2023-01-23': 'MS-OpenAI',
        '2023-05-25': 'NVIDIA Earnings', '2024-06-18': 'NVIDIA #1',
        '2025-01-27': 'DeepSeek',
    }
    for event_date_str, event_name in AI_EVENTS.items():
        event_date = pd.to_datetime(event_date_str)
        ax.axvline(x=event_date, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        if event_date_str in event_labels:
            ymin, ymax = ax.get_ylim()
            ax.text(event_date, ymax * 0.95, event_labels[event_date_str],
                    rotation=90, fontsize=7, ha='right', va='top', color='gray', alpha=0.7)

    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Fear Index (composite)')
    ax.set_title('Fear Sentiment by UAI Group Over Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_fear_by_uai.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_fear_by_uai.png")

    # ---- Figure 2: Event Study - Abnormal Fear Around Key Events ----
    if len(event_results) > 0:
        fear_events = event_results[event_results['variable'] == 'fear_index']

        if len(fear_events) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))

            # Get events that have all 3 UAI groups
            event_counts = fear_events.groupby('event')['uai_group'].nunique()
            complete_events = event_counts[event_counts >= 2].index.tolist()

            if complete_events:
                plot_data = fear_events[fear_events['event'].isin(complete_events)]

                events_list = sorted(plot_data['event_date'].unique())[-6:]  # Last 6 events
                plot_data = plot_data[plot_data['event_date'].isin(events_list)]

                x_labels = []
                for ed in events_list:
                    name = AI_EVENTS.get(ed, ed)
                    short_name = name[:20] + '...' if len(name) > 20 else name
                    x_labels.append(short_name)

                x = np.arange(len(events_list))
                width = 0.25

                for i, group in enumerate(['High', 'Medium', 'Low']):
                    vals = []
                    for ed in events_list:
                        g_data = plot_data[(plot_data['event_date'] == ed) &
                                           (plot_data['uai_group'] == group)]
                        vals.append(g_data['change'].mean() if len(g_data) > 0 else 0)

                    offset = (i - 1) * width
                    ax.bar(x + offset, vals, width, label=f'{group} UAI', color=colors[group])

                ax.set_xlabel('Event')
                ax.set_ylabel('Abnormal Fear Index Change')
                ax.set_title('Abnormal Fear Sentiment Around AI Disruption Events')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
                ax.legend()
                ax.axhline(y=0, color='black', linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_event_study.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("  Saved fig2_event_study.png")

    # ---- Figure 3: UAI Score vs. Average Fear Sentiment (Country Level) ----
    fig, ax = plt.subplots(figsize=(6, 4))

    country_agg = panel.groupby('country').agg(
        uai=('uai', 'first'),
        mean_fear=('fear', 'mean'),
        uai_group=('uai_group', 'first')
    ).dropna()

    for _, row in country_agg.iterrows():
        color = colors.get(row['uai_group'], 'gray')
        ax.scatter(row['uai'], row['mean_fear'], color=color, s=80, zorder=5)
        ax.annotate(row.name, (row['uai'], row['mean_fear']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Trend line
    from numpy.polynomial.polynomial import polyfit
    valid = country_agg.dropna(subset=['uai', 'mean_fear'])
    if len(valid) >= 3:
        z = np.polyfit(valid['uai'], valid['mean_fear'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['uai'].min() - 5, valid['uai'].max() + 5, 100)
        ax.plot(x_line, p(x_line), "--", color='gray', alpha=0.6, linewidth=1)

        from scipy import stats
        r, p_val = stats.pearsonr(valid['uai'], valid['mean_fear'])
        ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p_val:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top')

    ax.set_xlabel('Hofstede Uncertainty Avoidance Index (UAI)')
    ax.set_ylabel('Mean Fear Sentiment')
    ax.set_title('UAI vs. Average Fear Sentiment Across Countries')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_uai_vs_fear.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_uai_vs_fear.png")


def run_robustness_checks(panel):
    """Additional robustness: uncertainty and stress as alternative DVs."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECKS")
    print("=" * 70)

    import statsmodels.api as sm

    for dv in ['uncertainty', 'stress', 'gloom']:
        if dv not in panel.columns:
            continue

        reg_data = panel[['uai', 'buzz', dv, 'country']].dropna()
        if len(reg_data) < 100:
            continue

        # Standardize
        for col in [dv, 'buzz']:
            m = reg_data[col].mean()
            s = reg_data[col].std()
            if s > 0:
                reg_data[f'{col}_z'] = (reg_data[col] - m) / s

        uai_m = reg_data['uai'].mean()
        uai_s = reg_data['uai'].std()
        reg_data['uai_z'] = (reg_data['uai'] - uai_m) / uai_s

        y = reg_data[f'{dv}_z']
        X = sm.add_constant(reg_data[['uai_z', 'buzz_z']])

        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': reg_data['country']})

        uai_coef = model.params['uai_z']
        uai_pval = model.pvalues['uai_z']
        stars = '***' if uai_pval < 0.01 else '**' if uai_pval < 0.05 else '*' if uai_pval < 0.1 else ''

        print(f"  DV={dv:<15s}: UAI coef={uai_coef:>8.4f}{stars}, "
              f"p={uai_pval:.4f}, R²={model.rsquared:.4f}, N={int(model.nobs):,}")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RQ1: Cultural Uncertainty Avoidance and Fear Contagion")
    print("     in AI/Emerging Technology Companies")
    print("=" * 70)

    # Step 1: Load data
    raw_data = load_all_sentiment_data()

    # Step 2: Create panel
    panel = create_panel_dataset(raw_data)
    panel.to_pickle(os.path.join(OUTPUT_DIR, 'panel_data.pkl'))

    # Step 3: Descriptive statistics
    country_stats, uai_stats = descriptive_statistics(panel)

    # Step 4: Correlation analysis
    country_agg = correlation_analysis(panel)

    # Step 5: Panel regression
    reg_results = panel_regression(panel)

    # Step 6: Event study
    event_results = event_study(panel)

    # Step 7: Time series analysis
    monthly_ts = time_series_analysis(panel)

    # Step 8: Visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(panel, monthly_ts, event_results)

    # Step 9: Robustness checks
    run_robustness_checks(panel)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
