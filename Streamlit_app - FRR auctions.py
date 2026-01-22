import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.title("AFRR Auction Results Explorer")
date = st.date_input("Select delivery date:", help="Pick the date you want to analyze.")

@st.cache_data(show_spinner=True)
def fetch(dataset, d):
    url = "https://opendata.elia.be/api/records/1.0/search/"
    params = {"dataset": dataset, "rows": 10000, "refine.deliverydate": d}
    r = requests.get(url, params=params)
    return pd.DataFrame([rec['fields'] for rec in r.json().get("records", [])])

if date:
    df = fetch("ods125", str(date))
    if df.empty:
        st.warning("No data found for that date.")
        st.stop()

    # --- All-CCTU Table ---
    allcctu = df[df['capacitybiddeliveryperiod'].astype(str) == '0 - 24']
    sel = allcctu[allcctu['selectedbyoptimizer'].astype(str).str.lower() == "true"]
    if not sel.empty:
        t = sel[['index', 'pricedownmwh', 'priceupmwh', 'afrrofferedvolumedownmw', 'afrrawardedvolumeupmw']].copy()
        t.columns = ["Index", "Price Down", "Price Up", "Offered Down", "Awarded Up"]
        st.markdown("### All-CCTU (0-24): Selected Bids")
        st.table(t.round(2))
    else:
        st.info("No All-CCTU selected bids for this date.")

    # --- Summary Table ---
    periods = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']
    rows = []
    for p in periods:
        dper = df[df['capacitybiddeliveryperiod'].astype(str) == p]
        for dir_, vol_col, awd_col, prc_col in [
            ("UP", "afrrofferedvolumeupmw", "afrrawardedvolumeupmw", "priceupmwh"),
            ("DOWN", "afrrofferedvolumedownmw", "afrrawardedvolumedownmw", "pricedownmwh"),
        ]:
            valids = pd.to_numeric(dper[vol_col], errors='coerce').notnull()
            sub = dper[valids].copy()
            sub[vol_col] = sub[vol_col].astype(float)
            sub[awd_col] = pd.to_numeric(sub[awd_col], errors='coerce').fillna(0)
            sub[prc_col] = pd.to_numeric(sub[prc_col], errors='coerce').fillna(0)
            tot_vol = sub[vol_col].sum()
            sel = sub[sub['selectedbyoptimizer'].astype(str).str.lower() == "true"]
            tot_sel = sel[awd_col].sum()
            avg_price = (sel[prc_col] * sel[awd_col]).sum() / tot_sel if tot_sel > 0 else 0
            marg = sel[prc_col].max() if not sel.empty else 0
            rows.append([p, dir_, round(tot_sel,2), round(avg_price,2), round(marg,2), round(tot_vol,2)])
    summary = pd.DataFrame(rows, columns=[
        "Period", "Direction", "Total Awarded Volume (MW)", "Average Price (€/MWh)",
        "Marginal Price (€/MWh)", "Total Submitted Volume (MW)"
    ])
    st.markdown("### Period Results Summary")
    st.dataframe(summary, use_container_width=True)

    # --- Scatter Plot ---
    mask0 = (df['capacitybiddeliveryperiod'].astype(str) == '0 - 24') & (df['afrrofferedvolumedownmw'].astype(float) == 0)
    p0 = df[mask0]
    if not p0.empty:
        x = p0['afrrofferedvolumeupmw'].astype(float)
        y = p0['priceupmwh'].astype(float)
        sel = p0['selectedbyoptimizer'].astype(str).str.lower() == "true"
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(x[~sel], y[~sel], color='yellow', edgecolor='black', label='Not selected', s=40)
        ax.scatter(x[sel], y[sel], color='red', edgecolor='black', label='Selected', s=40)
        ax.set(xlabel='AFRR Offered Volume Up (MW)', ylabel='Price Up (€/MWh)', title="Offered Vol Up vs Price Up")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No data to plot for this combination.")

    # --- Merit Order Plots ---
    st.markdown("### AFRR Up Merit Order by Delivery Period")
    xmax = ymax = 0
    xmin = ymin = 1e20
    data = []
    for p in periods:
        sub = df[df['capacitybiddeliveryperiod'].astype(str) == p].copy()
        valid = pd.to_numeric(sub['afrrofferedvolumeupmw'], errors='coerce').notnull() & \
                pd.to_numeric(sub['priceupmwh'], errors='coerce').notnull()
        sub = sub[valid]
        if sub.empty:
            data.append(([], [], [], [], [], [], p))
            continue
        sub['afrrofferedvolumeupmw'] = sub['afrrofferedvolumeupmw'].astype(float)
        sub['priceupmwh'] = sub['priceupmwh'].astype(float)
        sub['selectedbyoptimizer'] = sub['selectedbyoptimizer'].astype(str).str.lower() == "true"
        sub = sub.sort_values(by='priceupmwh')
        cum = sub['afrrofferedvolumeupmw'].cumsum()
        prc = sub['priceupmwh']
        sel = sub['selectedbyoptimizer']
        data.append((
            cum, prc, cum[sel], prc[sel], cum[~sel], prc[~sel], p
        ))
        if not cum.empty:
            xmax, ymax = max(xmax, cum.max()), max(ymax, prc.max())
            xmin, ymin = min(xmin, cum.min()), min(ymin, prc.min())
    fig, axs = plt.subplots(3, 2, figsize=(9, 9))
    axs = axs.flatten()
    for i, (cum, prc, cs, ps, cnc, pnc, p) in enumerate(data):
        ax = axs[i]
        if len(cum): ax.step(cum, prc, where='post', color='gray', linewidth=1)
        if len(cnc): ax.scatter(cnc, pnc, color='yellow', edgecolor='black', s=20, label='Not selected')
        if len(cs): ax.scatter(cs, ps, color='red', edgecolor='black', s=20, label='Selected')
        ax.set_title(f"Period {p}")
        ax.set(xlabel="Cum. offered volume up (MW)", ylabel="Price up (€/MWh)", xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.grid(True)
        h, l = ax.get_legend_handles_labels()
        if l: ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)
    for j in range(len(periods), len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.suptitle("AFRR Up Merit Order by Delivery Period", fontsize=14, y=1.05)
    st.pyplot(fig)


