import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

st.title("AFRR Data Visualizer")

# === Delivery date selector ===
selected_date = st.date_input(
    "Delivery date",
    value=date.today(),
    min_value=date(2022, 1, 1),
    max_value=date.today(),
    format="YYYY-MM-DD"
)
date_str = selected_date.strftime("%Y-%m-%d")

@st.cache_data(show_spinner=False)
def fetch(dataset, d):
    url = "https://opendata.elia.be/api/records/1.0/search/"
    params = {"dataset": dataset, "rows": 10000, "refine.deliverydate": d}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return pd.DataFrame([rec['fields'] for rec in r.json().get("records", [])])

try:
    df = fetch("ods125", date_str)
    if df.empty:
        st.warning("No data found for that date.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Period definitions
periods = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']
directions = ['afrrofferedvolumeupmw', 'afrrofferedvolumedownmw']

# ----------- Auction results summary block ------------
st.markdown("### Auction Results")

# --- All-CCTU (0-24) selected bids
df_0_24 = df[df['capacitybiddeliveryperiod'].astype(str) == '0 - 24'].copy()
sel_mask_cctu = df_0_24.get('selectedbyoptimizer', pd.Series("false")).astype(str).str.lower() == "true"
selected_cctu = df_0_24[sel_mask_cctu]

if not selected_cctu.empty:
    st.markdown("**All-CCTU (0-24), selected bids:**")
    st.dataframe(
        selected_cctu[[
            'balancingserviceprovidercode',
            'afrrofferedvolumeupmw',
            'afrrofferedvolumedownmw',
            'priceupmwh',
            'pricedownmwh'
        ]].rename(columns={
            'balancingserviceprovidercode': 'Provider',
            'afrrofferedvolumeupmw': 'Up Volume (MW)',
            'afrrofferedvolumedownmw': 'Down Volume (MW)',
            'priceupmwh': 'Up Price (€/MWh)',
            'pricedownmwh': 'Down Price (€/MWh)'
        }),
        hide_index=True
    )
else:
    st.info("No bids were selected for All-CCTU (period 0 to 24).")

# --- Selected period statistics for other periods
results_rows = []
for per in periods:
    for direction in ['up', 'down']:
        field = f'afrrofferedvolume{direction}mw'
        price = 'priceupmwh' if direction == 'up' else 'pricedownmwh'
        sel_mask = df['capacitybiddeliveryperiod'].astype(str) == per
        subdf = df[sel_mask].copy()
        subdf[field] = pd.to_numeric(subdf[field], errors='coerce')
        subdf[price] = pd.to_numeric(subdf[price], errors='coerce')
        # Submitted
        total_submitted_vol = subdf[field].sum()
        # Selected
        sel_opt_mask = subdf.get('selectedbyoptimizer', pd.Series("false")).astype(str).str.lower() == "true"
        sel_bids = subdf[sel_opt_mask]
        total_selected_vol = sel_bids[field].sum()
        avg_price = np.nan
        if total_selected_vol > 0:
            avg_price = (sel_bids[field] * sel_bids[price]).sum() / total_selected_vol
        marginal_price = sel_bids[price].max(skipna=True)
        results_rows.append({
            'Period': per,
            'Direction': direction,
            'Total submitted (MW)': total_submitted_vol,
            'Total selected (MW)': total_selected_vol,
            'Avg selected price (€/MWh)': avg_price,
            'Marginal price (€/MWh)': marginal_price
        })
if results_rows:
    st.dataframe(
        pd.DataFrame(results_rows),
        hide_index=True
    )
else:
    st.info("No statistics could be calculated for the periods.")

# ==== All-CCTU up visualizer ====
st.markdown("## All-CCTU up visualizer")

# --- Offered down volume selectbox, after the title
unique_down_volumes = df_0_24['afrrofferedvolumedownmw'].dropna().unique()
unique_down_volumes = np.sort(unique_down_volumes.astype(float))
if len(unique_down_volumes) == 0:
    st.warning("No offered down volumes for '0 - 24' period.")
    st.stop()

down_volume = st.selectbox("Select a value for offered down volume (MW):", unique_down_volumes, index=0)

# --- Y-axis max below this element
max_price_main = st.number_input(
    "Set maximum for price Y-axis for all-CCTU chart (0 for auto):",
    min_value=0.0, value=0.0, step=1.0, format="%.2f", help="Set Y-axis max for main graph. Zero for auto."
)

# -------- Graph 1: All-CCTU up visualizer ---------
main_mask = (df_0_24['afrrofferedvolumedownmw'].astype(float) == down_volume)
plot_df = df_0_24[main_mask].copy()

if plot_df.empty:
    st.info("No data to plot for '0 - 24' at offered down volume = {}".format(down_volume))
else:
    x_main = plot_df.get('afrrofferedvolumeupmw', pd.Series(dtype=float)).astype(float)
    y_main = plot_df.get('priceupmwh', pd.Series(dtype=float)).astype(float)
    optimizer_mask = plot_df.get('selectedbyoptimizer', pd.Series("false")).astype(str).str.lower() == "true"
    x_sel = x_main[optimizer_mask]
    y_sel = y_main[optimizer_mask]
    x_not_sel = x_main[~optimizer_mask]
    y_not_sel = y_main[~optimizer_mask]

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(x_not_sel, y_not_sel, color='yellow', edgecolor='black', label='Not selected by Optimizer', s=50)
    ax1.scatter(x_sel, y_sel, color='red', edgecolor='black', label='Selected by Optimizer', s=50)
    ax1.set_xlabel('AFRR Offered Volume Up (MW)')
    ax1.set_ylabel('Price Up (€/MWh)')
    ax1.set_title(f"AFRR Offered Volume Up vs Price Up\nPeriod: 0 - 24 | Offered Down = {down_volume} MW")
    ax1.legend()
    ax1.grid(True)
    if max_price_main > 0:
        ax1.set_ylim(top=max_price_main)
    plt.tight_layout()
    st.pyplot(fig1)

# ==== Single CCTU aFRR up ====
st.markdown("## Single CCTU aFRR up")
max_price_single = st.number_input(
    "Set maximum for price Y-axis for single CCTU chart (0 for auto):",
    min_value=0.0, value=0.0, step=1.0, format="%.2f", help="Set Y-axis max for single CCTU graph. Zero for auto."
)

subplot_data = []
xmax, ymax, xmin, ymin = 0, 0, 0, 0
for period in periods:
    subdf = df[df['capacitybiddeliveryperiod'].astype(str) == period].copy()
    valid_mask = pd.to_numeric(subdf.get('afrrofferedvolumeupmw'), errors='coerce').notnull() & \
                    pd.to_numeric(subdf.get('priceupmwh'), errors='coerce').notnull()
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

fig2, axs = plt.subplots(3, 2, figsize=(12, 15))
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
    ax.set_ylim(ymin, max_price_single if max_price_single > 0 else ymax)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())
for j in range(len(periods), len(axs)):
    fig2.delaxes(axs[j])
plt.tight_layout()
plt.suptitle("AFRR Up Merit Order by Delivery Period", fontsize=16, y=1.02)
st.pyplot(fig2)
