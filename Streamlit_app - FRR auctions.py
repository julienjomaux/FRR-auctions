import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="AFRR Bids Overview", layout="wide")

st.title("AFRR Auction Results Explorer")

# --- Delivery date select ---
selected_date = st.date_input("Select delivery date:", help="Pick the date you want to analyze.")

@st.cache_data(show_spinner=True)
def fetch(dataset, d):
    url = "https://opendata.elia.be/api/records/1.0/search/"
    params = {"dataset": dataset, "rows": 10000, "refine.deliverydate": d}
    r = requests.get(url, params=params)
    r.raise_for_status()
    records = r.json().get("records", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([rec['fields'] for rec in records])

if selected_date:
    date_str = str(selected_date)

    df = fetch("ods125", date_str)

    if df.empty:
        st.warning("No data found for that date.")
        st.stop()

    # --- Results Calculation ---
    result_str = ""

    # ALL-CCTU: period 0-24, show for selected bids
    allcctu_mask = (df['capacitybiddeliveryperiod'].astype(str) == '0 - 24')
    allcctu_df = df[allcctu_mask].copy()
    if not allcctu_df.empty:
        selected = allcctu_df[allcctu_df['selectedbyoptimizer'].astype(str).str.lower() == "true"]
        if not selected.empty:
            result_str += "All-CCTU ('0-24'): Selected Bids\n"
            result_str += "{:<8} {:<12} {:<15} {:<16} {:<20}\n".format(
                'Index', 'PriceDown', 'PriceUp', 'OfferedDown', 'AwardedUp')
            for i, r in selected.iterrows():
                result_str += "{:<8} {:<12} {:<15} {:<16} {:<20}\n".format(
                    r.get('index', ''),
                    r.get('pricedownmwh', ''),
                    r.get('priceupmwh', ''),
                    r.get('afrrofferedvolumedownmw', ''),
                    r.get('afrrawardedvolumeupmw', '')
                )
            result_str += "\n"

    # All other periods summary
    periods = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']
    for period in periods:
        per_df = df[df['capacitybiddeliveryperiod'].astype(str) == period].copy()
        if per_df.empty:
            continue

        # UP DIRECTION
        up_sub = per_df.copy()
        up_sub = up_sub[pd.to_numeric(up_sub['afrrofferedvolumeupmw'], errors='coerce').notnull()]
        up_sub['afrrofferedvolumeupmw'] = up_sub['afrrofferedvolumeupmw'].astype(float)
        up_sub['afrrawardedvolumeupmw'] = pd.to_numeric(up_sub['afrrawardedvolumeupmw'], errors='coerce').fillna(0)
        up_sub['priceupmwh'] = pd.to_numeric(up_sub['priceupmwh'], errors='coerce').fillna(0)
        tot_vol_up = up_sub['afrrofferedvolumeupmw'].sum()
        sel_mask_up = up_sub['selectedbyoptimizer'].astype(str).str.lower() == "true"
        sel_up = up_sub[sel_mask_up]
        tot_sel_up = sel_up['afrrawardedvolumeupmw'].sum()
        avg_price_up = (sel_up['priceupmwh'] * sel_up['afrrawardedvolumeupmw']).sum() / tot_sel_up if tot_sel_up > 0 else 0
        marg_price_up = sel_up['priceupmwh'].max() if not sel_up.empty else 0

        # DOWN DIRECTION
        down_sub = per_df.copy()
        down_sub = down_sub[pd.to_numeric(down_sub['afrrofferedvolumedownmw'], errors='coerce').notnull()]
        down_sub['afrrofferedvolumedownmw'] = down_sub['afrrofferedvolumedownmw'].astype(float)
        down_sub['afrrawardedvolumedownmw'] = pd.to_numeric(down_sub['afrrawardedvolumedownmw'], errors='coerce').fillna(0)
        down_sub['pricedownmwh'] = pd.to_numeric(down_sub['pricedownmwh'], errors='coerce').fillna(0)
        tot_vol_down = down_sub['afrrofferedvolumedownmw'].sum()
        sel_mask_down = down_sub['selectedbyoptimizer'].astype(str).str.lower() == "true"
        sel_down = down_sub[sel_mask_down]
        tot_sel_down = sel_down['afrrawardedvolumedownmw'].sum()
        avg_price_down = (sel_down['pricedownmwh'] * sel_down['afrrawardedvolumedownmw']).sum() / tot_sel_down if tot_sel_down > 0 else 0
        marg_price_down = sel_down['pricedownmwh'].max() if not sel_down.empty else 0

        result_str += f"Period: {period}\n"
        result_str += " UP: total_submitted=%.1f, total_selected=%.1f, weighted_avg_price=%.2f, marginal_price=%.2f\n" % (
            tot_vol_up, tot_sel_up, avg_price_up, marg_price_up
        )
        result_str += "DOWN: total_submitted=%.1f, total_selected=%.1f, weighted_avg_price=%.2f, marginal_price=%.2f\n\n" % (
            tot_vol_down, tot_sel_down, avg_price_down, marg_price_down
        )

    # --- Results Section ---
    st.header("Auction Results")
    st.text(result_str.strip() if result_str else "No results found for this date.")

    # --- Main Scatter Plot Section ---
    st.header("AFRR Offered Volume Up vs Price Up (Period: 0 - 24, Offered Volume Down = 0)")
    period_mask = df['capacitybiddeliveryperiod'].astype(str) == '0 - 24'
    down_mask = df['afrrofferedvolumedownmw'].astype(float) == 0
    plot_df = df[period_mask & down_mask].copy()
    if plot_df.empty:
        st.info("No data to plot for this combination.")
    else:
        x_main = plot_df['afrrofferedvolumeupmw'].astype(float)
        y_main = plot_df['priceupmwh'].astype(float)
        optimizer_mask = plot_df['selectedbyoptimizer'].astype(str).str.lower() == "true"
        x_sel = x_main[optimizer_mask]
        y_sel = y_main[optimizer_mask]
        x_not_sel = x_main[~optimizer_mask]
        y_not_sel = y_main[~optimizer_mask]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_not_sel, y_not_sel, color='yellow', edgecolor='black', label='Not selected by Optimizer', s=50)
        ax.scatter(x_sel, y_sel, color='red', edgecolor='black', label='Selected by Optimizer', s=50)
        ax.set_xlabel('AFRR Offered Volume Up (MW)')
        ax.set_ylabel('Price Up (€/MWh)')
        ax.set_title("AFRR Offered Volume Up vs Price Up (0-24 period, Offered Down = 0)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # --- Merit order subplots ---
    st.header("AFRR Up Merit Order by Delivery Period")
    periods = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']
    xmax, ymax, xmin, ymin = 0, 0, 1e20, 1e20
    subplot_data = []
    for period in periods:
        subdf = df[df['capacitybiddeliveryperiod'].astype(str) == period].copy()
        valid_mask = pd.to_numeric(subdf['afrrofferedvolumeupmw'], errors='coerce').notnull() & \
                     pd.to_numeric(subdf['priceupmwh'], errors='coerce').notnull()
        subdf = subdf[valid_mask].copy()
        if subdf.empty:
            subplot_data.append(([], [], [], [], [], [], period))
            continue
        subdf['afrrofferedvolumeupmw'] = subdf['afrrofferedvolumeupmw'].astype(float)
        subdf['priceupmwh'] = subdf['priceupmwh'].astype(float)
        subdf['selectedbyoptimizer'] = subdf['selectedbyoptimizer'].astype(str).str.lower() == "true"
        subdf = subdf.sort_values(by='priceupmwh').reset_index(drop=True)
        cum_vol = subdf['afrrofferedvolumeupmw'].cumsum()
        prices = subdf['priceupmwh']
        sel_mask = subdf['selectedbyoptimizer']
        cum_vol_sel = cum_vol[sel_mask]
        prices_sel = prices[sel_mask]
        cum_vol_not_sel = cum_vol[~sel_mask]
        prices_not_sel = prices[~sel_mask]
        if not cum_vol.empty:
            xmax = max(xmax, cum_vol.max())
            ymax = max(ymax, prices.max())
            xmin = min(xmin, cum_vol.min())
            ymin = min(ymin, prices.min())
        subplot_data.append((cum_vol, prices, cum_vol_sel, prices_sel, cum_vol_not_sel, prices_not_sel, period))
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    axs = axs.flatten()
    for i, (cum_vol, prices, cum_vol_sel, prices_sel, cum_vol_not_sel, prices_not_sel, period) in enumerate(subplot_data):
        ax = axs[i]
        if len(cum_vol) > 0:
            ax.step(cum_vol, prices, where='post', color='gray', linewidth=1, label='_nolegend_')
        if len(cum_vol_not_sel) > 0:
            ax.scatter(cum_vol_not_sel, prices_not_sel, color='yellow', edgecolor='black', s=40, label='Not selected by Optimizer')
        if len(cum_vol_sel) > 0:
            ax.scatter(cum_vol_sel, prices_sel, color='red', edgecolor='black', s=40, label='Selected by Optimizer')
        ax.set_title(f"Period {period}")
        ax.set_xlabel("Cumulative offered volume up (MW)")
        ax.set_ylabel("Price up (€/MWh)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())
    for j in range(len(periods), len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.suptitle("AFRR Up Merit Order by Delivery Period", fontsize=16, y=1.02)
    st.pyplot(fig)
