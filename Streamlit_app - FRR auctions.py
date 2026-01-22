import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="AFRR Bids Explorer", layout="wide")
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

    # Center content: empty–main–empty columns
    col_left, col_main, col_right = st.columns([0.15, 0.7, 0.15])
    with col_main:

        # ---------------- All-CCTU Table ----------------
        allcctu = df[df['capacitybiddeliveryperiod'].astype(str) == '0 - 24']
        sel = allcctu[allcctu['selectedbyoptimizer'].astype(str).str.lower() == "true"]
        st.markdown("### All-CCTU (0-24): Selected Bids")
        st.markdown("Selected aFRR bids for the All-CCTU (0-24h) period.")
        if not sel.empty:
            t = sel[['index', 'pricedownmwh', 'priceupmwh', 'afrrofferedvolumedownmw', 'afrrawardedvolumeupmw']].copy()
            t.columns = ["Index", "Price Down", "Price Up", "Offered Down", "Awarded Up"]
            st.table(t.round(2))
        else:
            st.info("No All-CCTU selected bids for this date.")

        # ---------------- Summary Table ----------------
        st.markdown("### Period Results Summary")
        st.markdown("This summary table shows total volumes and prices by direction and period.")
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
                selx = sub[sub['selectedbyoptimizer'].astype(str).str.lower() == "true"]
                tot_sel = selx[awd_col].sum()
                avg_price = (selx[prc_col] * selx[awd_col]).sum() / tot_sel if tot_sel > 0 else 0
                marg = selx[prc_col].max() if not selx.empty else 0
                rows.append([p, dir_, round(tot_sel,2), round(avg_price,2), round(marg,2), round(tot_vol,2)])
        summary = pd.DataFrame(rows, columns=[
            "Period", "Direction", "Total Awarded Volume (MW)", "Average Price (€/MWh)",
            "Marginal Price (€/MWh)", "Total Submitted Volume (MW)"
        ])
        st.dataframe(summary, use_container_width=True)

        # ---------------- First Graph ----------------
        st.markdown("---")
        st.markdown("## Scatter Plot: Offered UP vs Upward Price (All-CCTU 0-24)")
        st.write("This scatter plot displays the offered upward volumes versus their respective upward prices "
                 "for the All-CCTU (0-24h) period. Use the droplist to filter bids by a specific 'Offered Volume Down' value.")

        vals = allcctu['afrrofferedvolumedownmw'].dropna().unique()
        vals = sorted(list({float(v) for v in vals}))
        selected_value = st.selectbox(
            "Select the 'Offered Volume Down' for filtering:", vals, index=0)
        mask0 = (allcctu['afrrofferedvolumedownmw'].astype(float) == selected_value)
        p0 = allcctu[mask0]
        if not p0.empty:
            x = p0['afrrofferedvolumeupmw'].astype(float)
            y = p0['priceupmwh'].astype(float)
            sel = p0['selectedbyoptimizer'].astype(str).str.lower() == "true"
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(x[~sel], y[~sel], color='yellow', edgecolor='black', label='Not selected', s=40)
            ax.scatter(x[sel], y[sel], color='red', edgecolor='black', label='Selected', s=40)
            ax.set(xlabel='AFRR Offered Volume Up (MW)', ylabel='Price Up (€/MWh)')
            ax.set_title("Offered Upward Volume vs Price Up", fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(fontsize=8)
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("No data to plot for this combination.")

        # ---------------- aFRR Up Merit Order Section ----------------
        st.markdown("---")
        st.markdown("## AFRR Up Merit Order by Delivery Period")
        st.write("These charts show the cumulative upward offered volumes sorted by price for each 4-hour delivery period. "
                 "Selected bids are highlighted. You can set the maximum Y (price) value for all plots below.")
        default_ymax_up = 500  # adjust as needed
        ymax_up = st.number_input("Set y-max (€/MWh) for AFRR Up Merit Order plots:", min_value=10, max_value=2000, value=default_ymax_up, step=10)

        # Prepare up merit order plots for all periods
        xmax = ymax = 0
        xmin = ymin = 1e20
        data_up = []
        for p in periods:
            sub = df[df['capacitybiddeliveryperiod'].astype(str) == p].copy()
            valid = pd.to_numeric(sub['afrrofferedvolumeupmw'], errors='coerce').notnull() & \
                    pd.to_numeric(sub['priceupmwh'], errors='coerce').notnull()
            sub = sub[valid]
            if sub.empty:
                data_up.append(([], [], [], [], [], [], p))
                continue
            sub['afrrofferedvolumeupmw'] = sub['afrrofferedvolumeupmw'].astype(float)
            sub['priceupmwh'] = sub['priceupmwh'].astype(float)
            sub['selectedbyoptimizer'] = sub['selectedbyoptimizer'].astype(str).str.lower() == "true"
            sub = sub.sort_values(by='priceupmwh')
            cum = sub['afrrofferedvolumeupmw'].cumsum()
            prc = sub['priceupmwh']
            sel = sub['selectedbyoptimizer']
            data_up.append((
                cum, prc, cum[sel], prc[sel], cum[~sel], prc[~sel], p
            ))
            if not cum.empty:
                xmax, ymax = max(xmax, cum.max()), max(ymax, prc.max())
                xmin, ymin = min(xmin, cum.min()), min(ymin, prc.min())
        fig1, axs1 = plt.subplots(3, 2, figsize=(9, 9))
        axs1 = axs1.flatten()
        for i, (cum, prc, cs, ps, cnc, pnc, p) in enumerate(data_up):
            ax = axs1[i]
            if len(cum): ax.step(cum, prc, where='post', color='gray', linewidth=1)
            if len(cnc): ax.scatter(cnc, pnc, color='yellow', edgecolor='black', s=20, label='Not selected')
            if len(cs): ax.scatter(cs, ps, color='red', edgecolor='black', s=20, label='Selected')
            ax.set_title(f"Period {p}", fontsize=10)
            ax.set(xlabel="Cum. offered volume up (MW)", ylabel="Price up (€/MWh)", xlim=(xmin, xmax), ylim=(ymin, ymax_up))
            ax.grid(True)
            h, l = ax.get_legend_handles_labels()
            if l: ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
        for j in range(len(periods), len(axs1)):
            fig1.delaxes(axs1[j])
        plt.tight_layout()
        plt.suptitle("AFRR Up Merit Order by Delivery Period", fontsize=14, y=1.05)
        st.pyplot(fig1)

        # ---------------- aFRR Down Merit Order Section ----------------
        st.markdown("---")
        st.markdown("## AFRR Down Merit Order by Delivery Period")
        st.write("These charts show the cumulative downward offered volumes sorted by price for each 4-hour delivery period. "
                 "Selected bids are highlighted. You can set the maximum Y (price) value for all plots below.")
        default_ymax_down = 500  # adjust as needed
        ymax_down = st.number_input("Set y-max (€/MWh) for AFRR Down Merit Order plots:", min_value=10, max_value=2000, value=default_ymax_down, step=10,
                                    key='ymax_down')

        # Prepare down merit order plots for all periods
        xmaxd = ymaxd = 0
        xmind = ymind = 1e20
        data_down = []
        for p in periods:
            sub = df[df['capacitybiddeliveryperiod'].astype(str) == p].copy()
            valid = pd.to_numeric(sub['afrrofferedvolumedownmw'], errors='coerce').notnull() & \
                    pd.to_numeric(sub['pricedownmwh'], errors='coerce').notnull()
            sub = sub[valid]
            if sub.empty:
                data_down.append(([], [], [], [], [], [], p))
                continue
            sub['afrrofferedvolumedownmw'] = sub['afrrofferedvolumedownmw'].astype(float)
            sub['pricedownmwh'] = sub['pricedownmwh'].astype(float)
            sub['selectedbyoptimizer'] = sub['selectedbyoptimizer'].astype(str).str.lower() == "true"
            sub = sub.sort_values(by='pricedownmwh')
            cum = sub['afrrofferedvolumedownmw'].cumsum()
            prc = sub['pricedownmwh']
            sel = sub['selectedbyoptimizer']
            data_down.append((
                cum, prc, cum[sel], prc[sel], cum[~sel], prc[~sel], p
            ))
            if not cum.empty:
                xmaxd, ymaxd = max(xmaxd, cum.max()), max(ymaxd, prc.max())
                xmind, ymind = min(xmind, cum.min()), min(ymind, prc.min())
        fig2, axs2 = plt.subplots(3, 2, figsize=(9, 9))
        axs2 = axs2.flatten()
        for i, (cum, prc, cs, ps, cnc, pnc, p) in enumerate(data_down):
            ax = axs2[i]
            if len(cum): ax.step(cum, prc, where='post', color='gray', linewidth=1)
            if len(cnc): ax.scatter(cnc, pnc, color='yellow', edgecolor='black', s=20, label='Not selected')
            if len(cs): ax.scatter(cs, ps, color='red', edgecolor='black', s=20, label='Selected')
            ax.set_title(f"Period {p}", fontsize=10)
            ax.set(xlabel="Cum. offered volume down (MW)", ylabel="Price down (€/MWh)",
                   xlim=(xmind, xmaxd), ylim=(ymind, ymax_down))
            ax.grid(True)
            h, l = ax.get_legend_handles_labels()
            if l: ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
        for j in range(len(periods), len(axs2)):
            fig2.delaxes(axs2[j])
        plt.tight_layout()
        plt.suptitle("AFRR Down Merit Order by Delivery Period", fontsize=14, y=1.05)
        st.pyplot(fig2)
