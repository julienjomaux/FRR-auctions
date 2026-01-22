import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

st.title("AFRR Data Visualizer")

# User selects delivery date
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

# --- Select offered down volume for period 0-24 ---
df_0_24 = df[df['capacitybiddeliveryperiod'].astype(str) == '0 - 24'].copy()
unique_down_volumes = df_0_24['afrrofferedvolumedownmw'].dropna().unique()
unique_down_volumes = np.sort(unique_down_volumes.astype(float))
if len(unique_down_volumes) == 0:
    st.warning("No offered down volumes for '0 - 24' period.")
    st.stop()
down_volume = st.selectbox("Select a value for offered down volume (MW) (all-CCTU):", unique_down_volumes, index=0)

# Option to set maximum price Y-axis for main chart
max_price_main = st.number_input(
    "Set maximum for price Y-axis for all-CCTU chart (blank for auto):",
    min_value=0.0, value=0.0, step=1.0, format="%.2f", help="Set Y-axis max for main graph. Zero for auto."
)

# --- Main Scatter Plot: All-CCTU up visualizer ---
st.markdown("## All-CCTU up visualizer")
main_mask = (df_0_24['afrrofferedvolumedownmw'].astype(float) == down_volume)
plot_df = df_0_24[main_mask].copy()

if plot_df.empty:
    st.info("No data to plot: no records for '0 - 24' with offered volume down = {}".format(down_volume))
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
    ax1.set_title(f"AFRR Offered Volume Up vs Price Up\nPeriod: 0 - 24, Offered Volume Down = {down_volume}")
    ax1.legend()
    ax1.grid(True)
    if max_price_main > 0:
        ax1.set_ylim(top=max_price_main)
    plt.tight_layout()
    st.pyplot(fig1)

# Option to set maximum price Y-axis for single CCTU chart
max_price_single = st.number_input(
    "Set maximum for price Y-axis for single CCTU chart (blank for auto):",
    min_value=0.0, value=0.0, step=1.0, format="%.2f", help="Set Y-axis max for single CCTU graph. Zero for auto."
)

st.markdown("## Single CCTU aFRR up")

periods = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']

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
